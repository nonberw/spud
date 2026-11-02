"""Many small worlds: tens of thousands of independent robots, stepped on every CPU core.

When robots don't interact, one big MuJoCo world is the slow way to simulate them: a single world
steps on a single core. `MassSim` keeps N independent copies as rows of one state array and
advances them with MuJoCo's C thread pool (`mujoco.rollout`), in chunks sized to a RAM budget.
It can checkpoint to disk and resume, so a long run on a flaky laptop survives a crash.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


class MassSim:
    def __init__(self, model: mujoco.MjModel, n: int, nthread: int | None = None,
                 ram_budget_gib: float | None = None, substeps: int = 1):
        self.model, self.n, self.substeps = model, n, substeps
        self.nthread = max(1, nthread or os.cpu_count() or 1)
        self.nstate = mujoco.mj_stateSize(model, STATE)
        d = mujoco.MjData(model)
        mujoco.mj_resetData(model, d)
        mujoco.mj_forward(model, d)
        s0 = np.zeros(self.nstate)
        mujoco.mj_getState(model, d, s0, STATE)
        self.s0 = s0
        self.state = np.tile(s0, (n, 1))
        self.sensordata = np.zeros((n, model.nsensordata))
        # Per env and chunk we hold: output states and sensors for `substeps`, plus the control.
        per_env = 8 * (substeps * (self.nstate + model.nsensordata) + substeps * model.nu)
        budget = (ram_budget_gib or 1.0) * 2**30
        self.chunk = int(max(1, min(n, budget // per_env)))
        self._datas = [mujoco.MjData(model) for _ in range(self.nthread)]
        self._models = [model] * self.chunk  # skip_checks wants one model reference per env
        self._runner = rollout.Rollout(nthread=self.nthread)
        self._out_state = np.zeros((self.chunk, substeps, self.nstate))
        self._out_sens = np.zeros((self.chunk, substeps, model.nsensordata))
        self.steps = 0

    @property
    def memory_gib(self) -> float:
        held = self.state.nbytes + self.sensordata.nbytes + self._out_state.nbytes + self._out_sens.nbytes
        return held / 2**30

    def step(self, ctrl: np.ndarray, sensor_reduce=np.max):
        """Advance every env by `substeps` physics steps with constant `ctrl` (n, nu).
        `sensordata` becomes `sensor_reduce` over the substeps (max suits touch sensors)."""
        for a in range(0, self.n, self.chunk):
            b = min(self.n, a + self.chunk)
            m = b - a
            control = np.ascontiguousarray(np.repeat(ctrl[a:b, None, :], self.substeps, axis=1))
            self._runner.rollout(self._models[:m], self._datas, np.ascontiguousarray(self.state[a:b]), control,
                                 nstep=self.substeps, skip_checks=True,
                                 state=self._out_state[:m], sensordata=self._out_sens[:m])
            self.state[a:b] = self._out_state[:m, -1]
            if self.model.nsensordata:
                self.sensordata[a:b] = sensor_reduce(self._out_sens[:m], axis=1)
        self.steps += 1

    # The state vector is [time, qpos, qvel, act, ...]; these views make it easy to read and edit.
    def qpos(self) -> np.ndarray:
        return self.state[:, 1:1 + self.model.nq]

    def qvel(self) -> np.ndarray:
        return self.state[:, 1 + self.model.nq:1 + self.model.nq + self.model.nv]

    def save(self, path: str | os.PathLike, **extra):
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(tmp, state=self.state, steps=self.steps, **extra)
        os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken checkpoint

    def load(self, path: str | os.PathLike) -> dict:
        z = np.load(path)
        if z["state"].shape != self.state.shape:
            raise ValueError(f"checkpoint has {z['state'].shape}, this sim is {self.state.shape}")
        self.state[:] = z["state"]
        self.steps = int(z["steps"])
        return {k: z[k] for k in z.files if k not in ("state", "steps")}

    def close(self):
        self._runner.close()
