"""`spud train` / `spud play`: PPO on any task file, then run what it learned.

Physics runs on all CPU cores (MuJoCo's rollout thread pool), the network and the PPO update on
the GPU if there is one. PPO itself is ZOOMIES' from-scratch implementation.

A run folder holds everything needed to reuse the policy:

    runs/reach-0412-1530/
      task.yaml        the task, exactly as trained
      progress.csv     iteration, env steps, seconds, return, success rate, ...
      policy.pt        actor-critic weights + observation normalizer + sizes
      policy.onnx      the actor alone (obs -> action), for other runtimes
      curve.png        learning curve
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from .tasks import make_env


def _ppo(obs_dim, act_dim, n, device, overrides=None):
    from zoomies.ppo import PPO, PPOConfig

    cfg = PPOConfig(**(overrides or {}))
    if "hidden" in (overrides or {}):
        cfg.hidden = tuple(cfg.hidden)
    return PPO(obs_dim, act_dim, n, cfg, device=device)


def train(task: dict, out: Path, iters: int | None = None, seed: int = 0, device: str | None = None,
          num_envs: int | None = None, log=print) -> dict:
    import yaml

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = log or (lambda *a: None)
    iters = int(iters or task.get("iters", 200))
    torch.manual_seed(seed)
    env = make_env(task, seed=seed, num_envs=num_envs)
    algo = _ppo(env.obs_dim, env.act_dim, env.n, device, task.get("ppo"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "task.yaml").write_text(yaml.safe_dump({k: v for k, v in task.items() if not k.startswith("_")},
                                                  sort_keys=False), encoding="utf-8")
    f = open(out / "progress.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["iter", "env_steps", "seconds", "ep_return", "success", "steps_per_s"])
    rets, succ = [], []
    obs = torch.as_tensor(env.observe(), device=device)
    t0 = time.perf_counter()
    best = {"success": -1.0}
    try:
        for it in range(1, iters + 1):
            for _ in range(algo.cfg.horizon):
                act, logp, val = algo.ac.act(obs)
                nobs, rew, done, timeout, info = env.step(act.cpu().numpy())
                r = torch.as_tensor(rew, device=device)
                if info["terminal_obs"] is not None:
                    r = algo.bootstrap_timeouts(r, torch.as_tensor(timeout, device=device, dtype=torch.float32),
                                                torch.as_tensor(info["terminal_obs"], device=device))
                algo.store(obs, act, logp, val, r, torch.as_tensor(done, device=device, dtype=torch.float32))
                obs = torch.as_tensor(nobs, device=device)
                if done.any():
                    rets += info["ep_return"][done].tolist()
                    succ += info["success"][done].tolist()
            stats = algo.update(obs)
            if it % 5 == 0 or it == 1 or it == iters:
                el = time.perf_counter() - t0
                steps = it * algo.cfg.horizon * env.n
                ret = float(np.mean(rets[-1000:])) if rets else float("nan")
                sr = float(np.mean(succ[-1000:])) if succ else float("nan")
                w.writerow([it, steps, round(el, 1), round(ret, 4), round(sr, 4), round(steps / el)])
                f.flush()
                log(f"it {it:4d} | {steps / 1e6:6.2f}M steps | {el / 60:5.1f} min | return {ret:8.3f} | "
                    f"success {sr:6.1%} | std {stats['std']:.2f} | {steps / el:,.0f} steps/s")
                if sr >= best["success"]:
                    best = {"success": sr, "iter": it}
                    save_policy(out / "policy.pt", algo.ac, env, task, it)
    finally:
        f.close()
        env.close()
    save_policy(out / "last.pt", algo.ac, env, task, iters)
    _export_onnx(out, algo.ac, env.obs_dim)
    _plot(out)
    return {"best_success": best["success"], "best_iter": best.get("iter"), "seconds": time.perf_counter() - t0,
            "env_steps": iters * algo.cfg.horizon * env.n}


def save_policy(path, ac, env, task, it):
    torch.save({"model": ac.state_dict(), "obs_dim": env.obs_dim, "act_dim": env.act_dim,
                "hidden": list(_hidden(ac)), "task": {k: v for k, v in task.items() if not k.startswith("_")},
                "iter": it}, path)


def _hidden(ac):
    return [m.out_features for m in ac.actor if isinstance(m, torch.nn.Linear)][:-1]


def load_policy(path, device="cpu"):
    """-> (actor-critic in eval mode, task dict)."""
    from zoomies.ppo import ActorCritic, PPOConfig

    ck = torch.load(path, map_location=device, weights_only=False)
    ac = ActorCritic(ck["obs_dim"], ck["act_dim"], PPOConfig(hidden=tuple(ck["hidden"]))).to(device)
    ac.load_state_dict(ck["model"])
    ac.eval()
    return ac, ck["task"]


class _Actor(torch.nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.norm, self.actor = ac.norm, ac.actor

    def forward(self, obs):
        return self.actor(self.norm(obs))


def _export_onnx(out, ac, obs_dim):
    import warnings

    try:
        actor = _Actor(ac).cpu().eval()
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
        torch.onnx.export(actor, torch.zeros(1, obs_dim), str(out / "policy.onnx"), input_names=["obs"],
                          output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                          dynamo=False)
    except Exception as e:  # onnx export is a bonus, never a failure
        (out / "policy.onnx.txt").write_text(f"ONNX export skipped: {e}\n", encoding="utf-8")
    finally:
        ac.to(next(ac.parameters()).device)


def _plot(out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(open(out / "progress.csv")))
    if not rows:
        return
    x = [int(r["env_steps"]) / 1e6 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(x, [float(r["ep_return"]) for r in rows], color="#7c4dff")
    ax[0].set(xlabel="env steps (M)", ylabel="episode return")
    ax[1].plot(x, [100 * float(r["success"]) for r in rows], color="#00bfa5")
    ax[1].set(xlabel="env steps (M)", ylabel="success %", ylim=(0, 100))
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- play --

@torch.no_grad()
def evaluate(policy_path, episodes: int = 256, seed: int = 123, randomize: bool = True) -> dict:
    """Run the deterministic policy on fresh envs (new seed) for exactly one episode each."""
    ac, task = load_policy(policy_path)
    task = dict(task)
    if not randomize:
        task.pop("randomize", None)
    env = make_env(task, seed=seed, num_envs=episodes)
    if hasattr(env, "steps"):
        env._reset(np.arange(env.n))  # everyone starts at step 0
    elif hasattr(env, "env"):
        env.env._reset(np.arange(env.n))
    finished = np.zeros(env.n, bool)
    success = np.zeros(env.n, bool)
    dist = np.full(env.n, np.nan)
    obs = env.observe()
    for _ in range(100000):
        a = ac.act(torch.as_tensor(obs), deterministic=True).numpy()
        obs, _, done, _, info = env.step(a)
        new = done & ~finished
        success[new] = info["success"][new]
        if "dist" in info:
            dist[new] = info["dist"][new]
        finished |= done
        if finished.all():
            break
    env.close()
    out = {"episodes": int(env.n), "success": float(success.mean())}
    if np.isfinite(dist).any():
        out["final_dist_mm_median"] = float(np.nanmedian(dist) * 1000)
    return out


@torch.no_grad()
def render_gif(policy_path, out_path, seconds: float = 8.0, width: int = 360, height: int = 270, seed: int = 7):
    """One robot, one camera, the policy driving. For reach, a new target every episode."""
    import mujoco
    from PIL import Image

    from .tasks import ReachEnv

    ac, task = load_policy(policy_path)
    if task["task"] != "reach":
        raise ValueError("GIFs of walk policies: use `spud world --policy <run>/policy.pt --gif out.gif`")
    env = ReachEnv(dict(task, randomize={}), seed=seed, num_envs=1, nthread=1)
    env._reset(np.arange(1))
    m = env.base
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    pool = env.targets_pool  # frame the camera on where this robot can reach
    cam.lookat[:] = [0, 0, float(np.median(pool[:, 2])) * 0.8]
    cam.distance = 1.6 * float(np.percentile(np.linalg.norm(pool, axis=1), 95)) + 0.2
    cam.elevation, cam.azimuth = -22, 135
    frames = []
    obs = env.observe()
    for k in range(int(seconds / env.dt)):
        obs, *_ = env.step(ac.act(torch.as_tensor(obs), deterministic=True).numpy())
        if k % 3:
            continue
        mujoco.mj_setState(m, d, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.mocap_pos[0] = env.target[0]
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        frames.append(Image.fromarray(r.render()))
    r.close()
    env.close()
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=60, loop=0)
    return len(frames)
