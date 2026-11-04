"""GPU physics: thousands of worlds on the graphics card with MuJoCo Warp, policy included.

Physics state lives in Warp arrays on the GPU. `wp.to_torch` gives zero-copy PyTorch views of
qpos / qvel / ctrl, so observations, the policy forward pass and the control write-back all stay
on the card. Nothing crosses PCIe inside the loop. One control step (4 physics substeps) is
captured as a CUDA graph and replayed.

Needs `mujoco-warp` and a CUDA build of `warp-lang` that matches your driver. Warp 1.19's default
wheel wants CUDA 13; on a CUDA 12 driver install the `+cu12` wheel from NVIDIA's releases.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


def available() -> tuple[bool, str]:
    try:
        import mujoco_warp  # noqa: F401
        import warp as wp
    except ImportError as e:
        return False, f"not installed ({e.name}): pip install mujoco-warp"
    wp.config.quiet = True
    wp.init()
    if not wp.is_cuda_available():
        return False, "Warp sees no CUDA device (driver too old for this Warp build? see spud/gpu.py)"
    return True, "ok"


def quat_to_mat_torch(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(*q.shape[:-1], 3, 3)


class GpuZoomies:
    """N Zoomies on MuJoCo Warp, driven by the trained ZOOMIES policy on the same GPU."""

    def __init__(self, n: int, policy_path: str, nconmax: int = 12, njmax: int = 64):
        import mujoco_warp as mjw
        import warp as wp
        from zoomies.env import ACTION_SCALE, Q_HIGH, Q_LOW
        from zoomies.policy import Policy
        from zoomies.robot import DECIMATION, DEFAULT_Q, load_model

        ok, why = available()
        if not ok:
            raise RuntimeError(why)
        self.wp, self.mjw, self.n = wp, mjw, n
        mjm = load_model()
        mjd = mujoco.MjData(mjm)
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
        mujoco.mj_forward(mjm, mjd)
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self.d.qpos)  # (n, 19), zero-copy
        self.qvel = wp.to_torch(self.d.qvel)  # (n, 18)
        self.ctrl = wp.to_torch(self.d.ctrl)  # (n, 12)
        dev = self.qpos.device
        self.default_q = torch.as_tensor(DEFAULT_Q, dtype=torch.float32, device=dev)
        self.q_low = torch.as_tensor(Q_LOW, dtype=torch.float32, device=dev)
        self.q_high = torch.as_tensor(Q_HIGH, dtype=torch.float32, device=dev)
        self.scale = ACTION_SCALE
        self.commands = torch.zeros(n, 3, device=dev)
        self.actions = torch.zeros(n, 12, device=dev)
        self.policy = Policy(policy_path, device=str(dev)).net
        self.decimation = DECIMATION
        self._graph = None

    def observe(self) -> torch.Tensor:
        R = quat_to_mat_torch(self.qpos[:, 3:7])
        lin = torch.einsum("nji,nj->ni", R, self.qvel[:, 0:3])
        cmd = self.commands * torch.tensor([2.0, 2.0, 0.25], device=self.commands.device)
        return torch.cat([lin * 2.0, self.qvel[:, 3:6] * 0.25, -R[:, 2, :], cmd,
                          self.qpos[:, 7:] - self.default_q, self.qvel[:, 6:] * 0.05, self.actions], 1).float()

    def _physics(self):
        for _ in range(self.decimation):
            self.mjw.step(self.m, self.d)

    @torch.no_grad()
    def step(self):
        a = self.policy.act(self.observe(), deterministic=True).clamp(-5, 5)
        self.actions = a
        self.ctrl[:] = torch.clamp(self.default_q + self.scale * a, self.q_low, self.q_high)
        if self._graph is None:  # first call compiles kernels; then capture the substeps once
            self._physics()
            with self.wp.ScopedCapture() as cap:
                self._physics()
            self._graph = cap.graph
        else:
            self.wp.capture_launch(self._graph)

    def upright(self) -> float:
        R = quat_to_mat_torch(self.qpos[:, 3:7])
        return float(((-R[:, 2, 2] < -0.5) & (self.qpos[:, 2] > 0.12)).float().mean())
