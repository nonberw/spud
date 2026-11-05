"""`spud bc`: learn a policy from recorded demos (behavior cloning), then run it in the sim.

The network sees the low-dimensional observation and predicts the next `chunk` actions at once
(action chunking, as in ACT). At run time every step gets a fresh chunk and the overlapping
predictions for the current step are averaged with exponentially decaying weights (temporal
ensembling), which smooths out the jitter a single-step policy would have.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .manip import OBS_KEYS


def load_demos(path, keys=OBS_KEYS, max_demos: int | None = None, only_success: bool = True):
    import h5py

    eps = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        meta = json.loads(f["data"].attrs.get("env_args", "{}"))
        for n in names:
            g = f["data"][n]
            if only_success and not bool(g.attrs.get("success", True)):
                continue
            obs = np.concatenate([g["obs"][k][()].reshape(len(g["actions"]), -1) for k in keys], 1)
            eps.append((obs.astype(np.float32), g["actions"][()].astype(np.float32)))
            if max_demos and len(eps) >= max_demos:
                break
    return eps, meta


def _windows(eps, chunk):
    """(obs_t, actions_t..t+chunk-1); past the end the last action repeats (the robot holds still)."""
    X, Y = [], []
    for obs, act in eps:
        pad = np.concatenate([act, np.repeat(act[-1:], chunk, 0)])
        idx = np.arange(len(act))[:, None] + np.arange(chunk)[None]
        X.append(obs)
        Y.append(pad[idx])
    return np.concatenate(X), np.concatenate(Y)


class ChunkPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk, hidden=(512, 512, 512)):
        super().__init__()
        self.obs_dim, self.act_dim, self.chunk = obs_dim, act_dim, chunk
        self.register_buffer("mu", torch.zeros(obs_dim))
        self.register_buffer("sd", torch.ones(obs_dim))
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, act_dim * chunk))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net((obs - self.mu) / self.sd).view(-1, self.chunk, self.act_dim)


def train_bc(demos_path, out_path, keys=OBS_KEYS, chunk: int = 10, epochs: int = 300, max_demos: int | None = None,
             batch: int = 512, lr: float = 1e-3, device: str | None = None, seed: int = 0, log=print) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    eps, meta = load_demos(demos_path, keys, max_demos)
    if not eps:
        raise ValueError(f"no successful demos in {demos_path}")
    X, Y = _windows(eps, chunk)
    pol = ChunkPolicy(X.shape[1], Y.shape[2], chunk).to(device)
    pol.mu[:] = torch.as_tensor(X.mean(0))
    pol.sd[:] = torch.as_tensor(X.std(0) + 1e-3)
    Xt, Yt = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * len(X) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.perf_counter()
    for s in range(steps):
        i = torch.randint(0, len(X), (batch,), device=device)
        loss = ((pol(Xt[i]) - Yt[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (s + 1) % max(1, steps // 5) == 0:
            log(f"  step {s + 1}/{steps}  loss {loss.item():.4f}")
    torch.save({"model": pol.state_dict(), "obs_dim": X.shape[1], "act_dim": Y.shape[2], "chunk": chunk,
                "keys": list(keys), "env": meta, "demos": len(eps), "samples": len(X)}, out_path)
    return {"demos": len(eps), "samples": len(X), "seconds": time.perf_counter() - t0, "final_loss": loss.item()}


class BCAgent:
    """Runs a trained ChunkPolicy with temporal ensembling. Same `act(obs_dict)` as the expert."""

    def __init__(self, path, device="cpu", ensemble_k: float = 0.1):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.pol = ChunkPolicy(ck["obs_dim"], ck["act_dim"], ck["chunk"]).to(device)
        self.pol.load_state_dict(ck["model"])
        self.pol.eval()
        self.keys, self.chunk, self.device = ck["keys"], ck["chunk"], device
        self.k = ensemble_k
        self.reset()

    def reset(self):
        self.history = []  # (step it was predicted at, chunk)
        self.t = 0

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        x = np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in self.keys])[None]
        chunk = self.pol(torch.as_tensor(x, device=self.device))[0].cpu().numpy()
        self.history = [(t0, c) for t0, c in self.history if self.t - t0 < self.chunk] + [(self.t, chunk)]
        preds = np.array([c[self.t - t0] for t0, c in self.history])
        ages = np.array([self.t - t0 for t0, _ in self.history])
        w = np.exp(-self.k * (ages.max() - ages))  # ACT's weighting: the oldest prediction counts most
        self.t += 1
        a = (preds * w[:, None]).sum(0) / w.sum()
        a[4] = 1.0 if a[4] > 0 else -1.0  # the gripper is a switch
        return a


def evaluate_bc(model_path, episodes: int = 100, seed: int = 50_000, log=None) -> dict:
    from .manip import PickEnv, collect

    agent = BCAgent(model_path)
    env = PickEnv(seed=seed)
    res = collect(env, agent, None, episodes, seed=seed, log=log)
    env.close()
    return res
