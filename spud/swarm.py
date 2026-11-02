"""One big world: K trained Zoomies (github.com/nonberw/zoomies) in a single MuJoCo scene,
all driven by the same policy in one batched forward pass.

Robots are copied into the scene with `MjSpec.attach`, so every robot's state is a contiguous
slice: qpos (7 + 12), qvel (6 + 12), ctrl (12), touch sensors (4).
"""

from __future__ import annotations

import mujoco
import numpy as np
from zoomies.env import ACTION_SCALE, Q_HIGH, Q_LOW, quat_to_mat
from zoomies.robot import DECIMATION, DEFAULT_Q, STAND_HEIGHT, build_xml

NQ_R, NV_R, NU_R, NS_R = 19, 18, 12, 4


def build_swarm(k: int, spacing: float = 1.0) -> mujoco.MjModel:
    world = mujoco.MjSpec.from_string(build_xml())
    # Strip the robot out of the base scene, keeping floor, lights, assets and options.
    world.delete(world.body("torso"))
    for a in list(world.actuators):
        world.delete(a)
    for s in list(world.sensors):
        world.delete(s)
    for key in list(world.keys):
        world.delete(key)
    half = max(50.0, spacing * np.sqrt(k))
    world.geom("floor").size = [half, half, 0.1]
    side = int(np.ceil(np.sqrt(k)))
    for i in range(k):
        r, c = divmod(i, side)
        robot = mujoco.MjSpec.from_string(build_xml())  # attach moves bodies, so one fresh copy each
        for key in list(robot.keys):
            robot.delete(key)
        # attach() brings the robot's whole worldbody: drop its floor and light, or every foot
        # touches k infinite planes and contacts grow as k^2.
        robot.delete(robot.geom("floor"))
        for light in list(robot.lights):
            robot.delete(light)
        frame = world.worldbody.add_frame(pos=[c * spacing, r * spacing, 0.0])
        world.attach(robot, prefix=f"r{i}_", frame=frame)  # brings actuators and sensors along
    return world.compile()


class Swarm:
    def __init__(self, k: int, spacing: float = 1.0, seed: int = 0):
        self.k = k
        self.model = build_swarm(k, spacing)
        self.data = mujoco.MjData(self.model)
        m = self.model
        assert m.nq == k * NQ_R and m.nv == k * NV_R and m.nu == k * NU_R, (m.nq, m.nv, m.nu)
        self.rng = np.random.default_rng(seed)
        self.commands = np.zeros((k, 3))
        self.actions = np.zeros((k, 12))
        self.reset()

    def reset(self):
        q = self.data.qpos.reshape(self.k, NQ_R)
        # Free-joint qpos is in world coordinates, so take each robot's spot from qpos0 (set by its
        # attach frame). Zeroing it would stack every robot on the origin.
        q[:, 0:2] = self.model.qpos0.reshape(self.k, NQ_R)[:, 0:2]
        q[:, 2] = STAND_HEIGHT
        q[:, 3] = 1.0
        q[:, 4:7] = 0.0
        q[:, 7:] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = np.tile(DEFAULT_Q, self.k)
        mujoco.mj_forward(self.model, self.data)
        self.actions[:] = 0.0

    def observe(self) -> np.ndarray:
        """Same 48-number observation as ZoomieEnv, without sensor noise."""
        qpos = self.data.qpos.reshape(self.k, NQ_R)
        qvel = self.data.qvel.reshape(self.k, NV_R)
        R = quat_to_mat(qpos[:, 3:7])
        lin = np.einsum("nji,nj->ni", R, qvel[:, 0:3])
        grav = -R[:, 2, :]
        obs = np.concatenate([lin * 2.0, qvel[:, 3:6] * 0.25, grav, self.commands * np.array([2.0, 2.0, 0.25]),
                              qpos[:, 7:] - DEFAULT_Q, qvel[:, 6:] * 0.05, self.actions], 1)
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        self.actions = np.clip(actions, -5.0, 5.0)
        ctrl = np.clip(DEFAULT_Q + ACTION_SCALE * self.actions, Q_LOW, Q_HIGH)
        self.data.ctrl[:] = ctrl.ravel()
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    def upright(self) -> np.ndarray:
        q = self.data.qpos.reshape(self.k, NQ_R)
        R = quat_to_mat(q[:, 3:7])
        return (-R[:, 2, 2] < -0.5) & (q[:, 2] > 0.12)

    def positions(self) -> np.ndarray:
        """World xy of every torso."""
        return np.array([self.data.xpos[self.model.body(f"r{i}_torso").id][:2] for i in range(self.k)])
