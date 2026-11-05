"""Pick-and-place you can drive: an arm, a cube, a goal, and Cartesian controls.

The action is what a human operator thinks in, not joint angles:

    [vx, vy, vz, yaw_rate, grip]   each in [-1, 1]; grip > 0 closes the fingers

The controller integrates the velocity into a target gripper pose (pointing down) and solves
ARMageddon's IK every control tick, warm-started from the last solution. The same action space is
used by keyboard teleop, phone teleop, the scripted expert and the learned (BC) policies, so a
dataset recorded with any of them trains a policy for all of them.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

CUBE_HALF = 0.02
SPAWN = np.array([[0.30, -0.18], [0.50, 0.18]])
WORKSPACE = np.array([[0.15, -0.35, CUBE_HALF - 0.008], [0.62, 0.35, 0.35]])
OPEN, CLOSED = 0.04, 0.0
OBS_KEYS = ("ee_pos", "ee_yaw", "gripper", "cube_pos", "cube_yaw", "goal_pos", "cube_rel", "goal_rel")


def gripper_down(yaw: float) -> np.ndarray:
    from armageddon.kinematics import rotation

    return rotation(np.array([0.0, 0.0, 1.0]), yaw) @ rotation(np.array([0.0, 1.0, 0.0]), np.pi)


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def scene_xml(camera_size=(128, 128)) -> str:
    from armageddon import default_arm
    from armageddon.mjcf import ARM_MATERIALS, arm_actuators, arm_xml

    chain = default_arm()
    return f"""<mujoco model="spud_pick">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="{max(640, camera_size[0])}" offheight="{max(480, camera_size[1])}"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.09 0.32" rgb2="0.01 0.01 0.04" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.12 0.2" rgb2="0.08 0.08 0.13" width="256" height="256" mark="edge" markrgb="0.45 0.3 0.9"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.1"/>
{ARM_MATERIALS}
  </asset>
  <default><joint damping="3" armature="0.01"/><geom contype="1" conaffinity="0"/></default>
  <worldbody>
    <light pos="0.6 -0.6 2.2" dir="-0.3 0.3 -1" diffuse="0.9 0.85 1" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="floor" contype="0" conaffinity="1" condim="3" friction="1 0.02 0.002"/>
    <camera name="front" pos="1.05 0 0.62" xyaxes="0 1 0 -0.5 0 0.866" fovy="48"/>
    <camera name="top" pos="0.4 0 1.1" xyaxes="0 1 0 -1 0 0" fovy="45"/>
{arm_xml(chain, gripper="parallel")}
    <body name="goal" mocap="true" pos="0.4 0.1 0.001">
      <geom type="cylinder" size="0.03 0.001" rgba="0.1 1 0.6 0.5" contype="0" conaffinity="0"/>
    </body>
    <body name="cube" pos="0.4 0 {CUBE_HALF}">
      <freejoint/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" rgba="1 0.45 0.1 1" mass="0.05"
            contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" solref="0.005 1"/>
    </body>
  </worldbody>
  <actuator>
{arm_actuators(chain, gripper="parallel")}
  </actuator>
</mujoco>"""


class PickEnv:
    MAX_V, MAX_W, DT = 0.25, 1.5, 0.02

    def __init__(self, seed: int = 0, episode_seconds: float = 15.0, camera: str | None = None,
                 camera_size=(96, 96)):
        from armageddon import default_arm

        self.chain = default_arm()
        self.model = mujoco.MjModel.from_xml_string(scene_xml(camera_size))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.n = self.chain.dof
        self.substeps = round(self.DT / self.model.opt.timestep)
        self.max_steps = int(episode_seconds / self.DT)
        self.cube_jnt = self.model.joint(self.model.body("cube").jntadr[0])
        self.ee_site = self.model.site("ee").id
        self.finger_q = [self.model.joint(f"finger_{s}").qposadr[0] for s in ("l", "r")]
        self.camera, self.camera_size = camera, camera_size
        self._renderer = None
        self.meta = {"env_name": "spud/pick", "type": 1, "env_kwargs": {"episode_seconds": episode_seconds,
                                                                       "camera": camera, "camera_size": list(camera_size)},
                     "action": "vx vy vz yaw_rate grip, each in [-1, 1]; grip > 0 closes"}
        self.reset()

    # ---------------------------------------------------------------- reset --
    def reset(self, seed: int | None = None) -> dict:
        from armageddon import HOME, solve

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        cube = self.rng.uniform(*SPAWN)
        while True:
            goal = self.rng.uniform(*SPAWN)
            if np.linalg.norm(goal - cube) > 0.12:
                break
        yaw = self.rng.uniform(-np.pi / 4, np.pi / 4)
        adr = self.cube_jnt.qposadr[0]
        self.data.qpos[adr:adr + 7] = [*cube, CUBE_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.model.body("goal").pos[:] = [*goal, 0.001]
        self.data.mocap_pos[0] = [*goal, 0.001]
        self.goal = np.array([*goal, CUBE_HALF])

        start = np.array([0.32 + self.rng.uniform(-0.03, 0.03), self.rng.uniform(-0.05, 0.05), 0.22])
        self.p_des, self.yaw_des = start, 0.0
        seed_q = HOME.copy()
        res = solve(self.chain, start, gripper_down(0.0), q0=seed_q, restarts=10, q_rest=seed_q,
                    nullspace_gain=0.05, rng=np.random.default_rng(0))
        self.q_des = res.q
        self.data.qpos[: self.n] = self.q_des
        self.grip = -1.0
        self.data.qpos[self.finger_q] = OPEN
        self.data.ctrl[: self.n] = self.q_des
        self.data.ctrl[self.n:self.n + 2] = OPEN
        mujoco.mj_forward(self.model, self.data)
        self.steps, self.placed_for = 0, 0
        return self.obs()

    # ----------------------------------------------------------------- step --
    def step(self, action) -> tuple[dict, float, bool, dict]:
        from armageddon import solve

        a = np.clip(np.asarray(action, float), -1, 1)
        self.p_des = np.clip(self.p_des + a[:3] * self.MAX_V * self.DT, *WORKSPACE)
        self.yaw_des = float(np.clip(self.yaw_des + a[3] * self.MAX_W * self.DT, -2.6, 2.6))
        res = solve(self.chain, self.p_des, gripper_down(self.yaw_des), q0=self.q_des, max_iters=20)
        self.q_des = res.q
        self.grip = float(a[4])
        self.data.ctrl[: self.n] = self.q_des
        self.data.ctrl[self.n:self.n + 2] = CLOSED if self.grip > 0 else OPEN
        for _ in range(self.substeps):
            self.data.qfrc_applied[: self.n] = self.data.qfrc_bias[: self.n]  # gravity compensation
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        placed = self.is_placed()
        self.placed_for = self.placed_for + 1 if placed else 0
        success = self.placed_for >= 10
        done = success or self.steps >= self.max_steps
        return self.obs(), float(success), done, {"success": success, "timeout": self.steps >= self.max_steps}

    def cube_pose(self):
        adr = self.cube_jnt.qposadr[0]
        q = self.data.qpos[adr:adr + 7]
        w, x, y, z = q[3:]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return q[:3].copy(), float(yaw)

    def is_placed(self) -> bool:
        pos, _ = self.cube_pose()
        opening = float(np.mean(self.data.qpos[self.finger_q]))
        return (np.linalg.norm(pos[:2] - self.goal[:2]) < 0.03 and pos[2] < CUBE_HALF + 0.008 and opening > 0.025)

    # ------------------------------------------------------------------ obs --
    def obs(self) -> dict:
        ee = self.data.site_xpos[self.ee_site].copy()
        R = self.data.site_xmat[self.ee_site].reshape(3, 3)
        ee_yaw = np.arctan2(-R[0, 1], R[1, 1])  # gripper_down(yaw) has its y axis at (-sin, cos, 0)
        cube, cyaw = self.cube_pose()
        o = {
            "ee_pos": ee, "ee_yaw": np.array([np.sin(ee_yaw), np.cos(ee_yaw)]),
            "gripper": np.array([np.mean(self.data.qpos[self.finger_q]) / OPEN]),
            "joint_pos": self.data.qpos[: self.n].copy(),
            "cube_pos": cube, "cube_yaw": np.array([np.sin(4 * cyaw), np.cos(4 * cyaw)]),  # a cube looks the same every 90 deg
            "goal_pos": self.goal.copy(), "cube_rel": cube - ee, "goal_rel": self.goal - cube,
        }
        o = {k: v.astype(np.float32) for k, v in o.items()}
        if self.camera:
            o["rgb"] = self.render(self.camera, *self.camera_size)
        return o

    def state(self) -> np.ndarray:
        s = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, self.data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return s

    def render(self, camera="front", width=320, height=240) -> np.ndarray:
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ----------------------------------------------------------------- expert --

class ScriptedExpert:
    """A tireless operator that knows where everything is. Emits the same commands a human would.

    It is memoryless on purpose: the action is a smooth function of the observation only (no hidden
    phase counter). A phase-based script made behavior cloning stall at phase boundaries, hovering
    over the goal forever, because the learner can't see the phase. This one also recovers: if the
    cube slips, the observation says so and it goes back for it.
    """

    def __init__(self, env: PickEnv):
        self.env = env

    def reset(self):
        pass

    @staticmethod
    def _v(err, gain=5.0):
        return np.clip(gain * np.asarray(err), -PickEnv.MAX_V, PickEnv.MAX_V) / PickEnv.MAX_V

    def act(self, obs) -> np.ndarray:
        ee, cube, goal = obs["ee_pos"], obs["cube_pos"], obs["goal_pos"]
        opening = float(obs["gripper"][0])  # 1 = open, ~0.5 = squeezing the cube, 0 = closed on air
        near = np.linalg.norm(cube - ee) < 0.03
        d_goal = float(np.linalg.norm(goal[:2] - cube[:2]))
        a = np.zeros(5)

        if d_goal < 0.015 and cube[2] < CUBE_HALF + 0.008:  # on the goal: let go, then back off upward
            a[4] = -1.0
            if opening > 0.9:
                a[2] = self._v(0.15 - ee[2])
            return a

        if near and opening < 0.9:  # holding it
            a[4] = 1.0
            if opening > 0.62:  # fingers still closing: wait for the squeeze
                return a
            s = np.clip(1 - (d_goal - 0.006) / 0.03, 0, 1)  # descend smoothly as the cube comes over the goal
            z_t = 0.15 * (1 - s) + CUBE_HALF * s
            # Lift first, then travel, so it never drags the cube; close to the goal, steer freely.
            lift = 1.0 if d_goal < 0.04 else np.clip((ee[2] - 0.06) / 0.05, 0, 1)
            a[:2] = self._v(goal[:2] - cube[:2], 8.0) * lift
            a[2] = self._v(z_t - ee[2], 8.0)
            return a

        if opening < 0.9 and not near:  # closed on air or dropped it: open and rise a little
            a[4] = -1.0
            a[2] = self._v(0.12 - ee[2])
            return a

        # Reach for the cube: align in xy and yaw high up, then come straight down, then close.
        ee_yaw = float(np.arctan2(obs["ee_yaw"][0], obs["ee_yaw"][1]))
        cube_yaw = float(np.arctan2(obs["cube_yaw"][0], obs["cube_yaw"][1])) / 4  # known mod 90 deg
        options = cube_yaw + np.pi / 2 * np.arange(-3, 4)
        dyaw = float(options[np.argmin(np.abs(options - ee_yaw))] - ee_yaw)
        xy_err = cube[:2] - ee[:2]
        s = (np.clip(1 - (np.linalg.norm(xy_err) - 0.004) / 0.02, 0, 1)
             * np.clip(1 - (abs(dyaw) - 0.03) / 0.25, 0, 1))
        grasp = cube - [0, 0, 0.004]
        z_t = 0.10 * (1 - s) + grasp[2] * s
        a[:2] = self._v(xy_err, 8.0)
        a[2] = self._v(z_t - ee[2], 8.0)
        a[3] = np.clip(4.0 * dyaw, -PickEnv.MAX_W, PickEnv.MAX_W) / PickEnv.MAX_W
        a[4] = 1.0 if np.linalg.norm(grasp - ee) < 0.007 else -1.0
        return a


# ------------------------------------------------------------ the dataset --

class DemoWriter:
    """Episodes into one HDF5 file, robomimic layout: data/demo_k/{obs/<key>, actions, rewards, dones, states}."""

    def __init__(self, path, env_meta: dict, append: bool = True):
        import h5py

        self.f = h5py.File(path, "a" if append else "w")
        self.data = self.f.require_group("data")
        self.data.attrs["env_args"] = json.dumps(env_meta)
        self.count = len([k for k in self.data.keys() if k.startswith("demo_")])
        self.total = int(self.data.attrs.get("total", 0))

    def add(self, obs: list[dict], actions, rewards, dones, states, **attrs) -> str:
        name = f"demo_{self.count}"
        g = self.data.create_group(name)
        og = g.create_group("obs")
        for k in obs[0]:
            arr = np.stack([o[k] for o in obs])
            og.create_dataset(k, data=arr, compression="gzip" if arr.dtype == np.uint8 else None)
        g.create_dataset("actions", data=np.asarray(actions, np.float32))
        g.create_dataset("rewards", data=np.asarray(rewards, np.float32))
        g.create_dataset("dones", data=np.asarray(dones, np.uint8))
        g.create_dataset("states", data=np.asarray(states, np.float64))
        g.attrs["num_samples"] = len(actions)
        for k, v in attrs.items():
            g.attrs[k] = v
        self.count += 1
        self.total += len(actions)
        self.data.attrs["total"] = self.total
        self.f.flush()
        return name

    def close(self):
        self.f.close()


def collect(env: PickEnv, policy, writer: DemoWriter | None, episodes: int, seed: int = 0, noise: float = 0.0,
            keep_failures: bool = False, log=None) -> dict:
    """Roll out `policy(obs) -> action` and record. With noise > 0 the *executed* action gets Gaussian
    noise on the motion channels but the *recorded* action stays clean (DART), so the data shows how
    to recover from small mistakes."""
    succ, lengths = [], []
    act = policy.act if hasattr(policy, "act") else policy
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        O, A, R, D, S = [], [], [], [], []
        done = False
        info = {"success": False}
        while not done:
            a = np.asarray(act(obs), float)
            run = a.copy()
            if noise > 0:
                run[:4] = np.clip(run[:4] + env.rng.normal(0, noise, 4), -1, 1)
            O.append(obs)
            S.append(env.state())
            obs, r, done, info = env.step(run)
            A.append(a)
            R.append(r)
            D.append(done)
        succ.append(info["success"])
        lengths.append(len(A))
        if writer is not None and (info["success"] or keep_failures):
            writer.add(O, A, R, D, S, success=bool(info["success"]), seed=seed + ep)
        if log and (ep + 1) % max(1, episodes // 10) == 0:
            log(f"  episode {ep + 1}/{episodes}: success so far {np.mean(succ):.1%}")
    return {"episodes": episodes, "success": float(np.mean(succ)), "mean_seconds": float(np.mean(lengths) * env.DT)}
