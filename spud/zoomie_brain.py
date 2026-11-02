"""Glue between SPUD's mass simulator and ZOOMIES: observations, actions, and two kinds of brain.

* `TrainedBrain`: the real trained ZOOMIES policy (~380k params). Robots actually walk.
* `BigBrain`: obs -> embed -> N streamed transformer blocks -> actions. Random weights: a systems
  benchmark for "a model too big for the GPU drives thousands of robots", not a skill.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from zoomies.env import ACTION_SCALE, Q_HIGH, Q_LOW, quat_to_mat
from zoomies.robot import DECIMATION, DEFAULT_Q, STAND_HEIGHT, load_model

from .mass import MassSim
from .models import iter_blocks
from .stream import StreamedBlocks, resident_copy

NQ, NV = 19, 18


class ZoomieMass:
    """N independent Zoomies on MassSim, with the same 48-number observation ZOOMIES trained on."""

    def __init__(self, n: int, nthread: int | None = None, ram_budget_gib: float | None = None):
        self.sim = MassSim(load_model(), n, nthread, ram_budget_gib, substeps=DECIMATION)
        self.n = n
        q = self.sim.qpos()
        q[:, 2] = STAND_HEIGHT
        q[:, 3] = 1.0
        q[:, 7:] = DEFAULT_Q
        self.commands = np.zeros((n, 3))
        self.actions = np.zeros((n, 12))

    def observe(self) -> np.ndarray:
        qpos, qvel = self.sim.qpos(), self.sim.qvel()
        R = quat_to_mat(qpos[:, 3:7])
        lin = np.einsum("nji,nj->ni", R, qvel[:, 0:3])
        return np.concatenate([lin * 2.0, qvel[:, 3:6] * 0.25, -R[:, 2, :], self.commands * np.array([2.0, 2.0, 0.25]),
                               qpos[:, 7:] - DEFAULT_Q, qvel[:, 6:] * 0.05, self.actions], 1).astype(np.float32)

    def step(self, actions: np.ndarray):
        self.actions = np.clip(actions, -5.0, 5.0)
        self.sim.step(np.clip(DEFAULT_Q + ACTION_SCALE * self.actions, Q_LOW, Q_HIGH))

    def upright(self) -> np.ndarray:
        q = self.sim.qpos()
        return (-quat_to_mat(q[:, 3:7])[:, 2, 2] < -0.5) & (q[:, 2] > 0.12)


class TrainedBrain:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        from zoomies.policy import Policy

        self.policy = Policy(checkpoint, device=device)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.policy(obs)


class BigBrain(nn.Module):
    def __init__(self, depth: int, d: int, heads: int, device: str = "cuda", store: str = "ram",
                 micro_batch: int | None = None, n_slots: int = 2, resident: bool = False):
        super().__init__()
        dt = torch.float16 if device == "cuda" else torch.float32
        self.dtype, self.device = dt, device
        self.embed = nn.Linear(48, d).to(device, dt)
        self.head = nn.Linear(d, 12).to(device, dt)
        if resident:  # fits the card: no reason to pay for streaming
            self.trunk = resident_copy(list(iter_blocks(depth, d, heads, dtype=dt)), device)
        else:
            self.trunk = StreamedBlocks(iter_blocks(depth, d, heads, dtype=dt), device=device, store=store,
                                        micro_batch=micro_batch, n_slots=n_slots)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = self.embed(torch.as_tensor(obs, device=self.device, dtype=self.dtype))[:, None, :]
        return self.head(self.trunk(x)[:, 0]).float().cpu().numpy()
