"""Training tasks: thousands of copies of one robot, stepped on every CPU core, batched in NumPy.

A task file says which robot, what it has to learn, and how much the world gets randomized:

    task: reach                      # reach | walk
    robot: {type: arm, gripper: fixed}
    # robot: {type: urdf, file: my_arm.urdf, ee: tool_link, ee_offset: [0, 0, 0.05], kp: 80}
    envs: 1024
    episode_seconds: 4
    randomize: {damping: [0.7, 1.3], kp: [0.8, 1.2], mass: [0.8, 1.2]}

`reach`: any fixed-base robot learns to put its end effector on a random target. Works for the
ARMageddon arm and for your own URDF / MJCF arm: if the file has no motors, every hinge and
slide joint gets a position servo, and gravity is compensated so weak servos don't sag.

`walk`: the ZOOMIES quadruped learns to walk from scratch (ZOOMIES' own env, rewards and kicks).

Every env has the same interface as ZOOMIES' ZoomieEnv, so one PPO loop trains all of them:
`observe()`, `step(actions) -> obs, reward, done, timeout, info` with `info["terminal_obs"]`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

TASKS = ("reach", "walk")


def load_task(src) -> dict:
    from .scene import load_config

    cfg = dict(load_config(src))
    if cfg.get("task") not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {cfg.get('task')!r}")
    rc = cfg.get("robot", {})
    if "file" in rc and not Path(rc["file"]).is_absolute():
        # Pin the robot file now: the run folder's task.yaml must still work from anywhere later.
        cfg["robot"] = dict(rc, file=str((Path(cfg.get("_base_dir", ".")) / rc["file"]).resolve()))
    return cfg


# ------------------------------------------------------------ robot model --

def build_robot_model(rc: dict, base_dir: str = ".", timestep: float = 0.004, target_marker: bool = True):
    """One fixed-base robot on a floor, with servos, gravity compensation and an `ee` site.

    Returns (model, info) where info has the actuated joints' qpos/dof addresses, ctrl ranges,
    the home pose and the id of the end-effector position sensor.
    """
    from .scene import _robot_spec

    kind = rc["type"]
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.13, 0.12, 0.2], rgb2=[0.08, 0.08, 0.13], width=256, height=256,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.45, 0.3, 0.9])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.16, 0.09, 0.32], rgb2=[0.01, 0.01, 0.04], width=256, height=256)
    mat = spec.add_material(name="ground", reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [4, 4]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], material="ground",
                            contype=0, conaffinity=1)
    spec.worldbody.add_light(pos=[0.6, -0.6, 2.2], dir=[-0.3, 0.3, -1], diffuse=[0.9, 0.85, 1], castshadow=True)

    child = _robot_spec(kind, rc, base_dir)
    if kind in ("urdf", "mjcf"):
        if "ee" not in rc:
            raise ValueError("a urdf/mjcf robot needs `ee: <body name>` (the link whose tip should reach)")
        child.body(rc["ee"]).add_site(name="ee", pos=rc.get("ee_offset", [0, 0, 0]), size=[0.01, 0, 0])
        if not list(child.actuators):  # URDFs carry no motors: give every 1-dof joint a position servo
            for j in child.joints:
                if j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    a = child.add_actuator(name=f"act_{j.name}", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                    a.set_to_position(kp=float(rc.get("kp", 50.0)), kv=float(rc.get("kv", 0.0)))
                    if j.range[1] > j.range[0]:
                        a.ctrlrange, a.ctrllimited = list(j.range), True
                    j.damping = np.maximum(np.asarray(j.damping, float), float(rc.get("damping", 1.0)))
    for b in child.bodies:
        if b.name != "world":
            b.gravcomp = 1.0  # servos fight the task, not gravity
    spec.attach(child, prefix="r_", frame=spec.worldbody.add_frame(pos=rc.get("pos", [0, 0, 0])))
    spec.add_sensor(name="ee_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="r_ee")
    if target_marker:
        t = spec.worldbody.add_body(name="target", mocap=True, pos=[0.4, 0, 0.3])
        t.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[1, 0.25, 0.5, 0.9],
                   contype=0, conaffinity=0)
    model = spec.compile()

    acts = list(range(model.nu))
    if not acts:
        raise ValueError("this robot has no actuated joints")
    jnt = model.actuator_trnid[acts, 0]
    lo, hi = model.actuator_ctrlrange[acts].T.copy()
    unlimited = ~model.actuator_ctrllimited[acts].astype(bool)
    lo[unlimited], hi[unlimited] = -np.pi, np.pi
    if kind == "arm":
        from armageddon import HOME

        home = np.asarray(HOME, float)[: len(acts)]
    else:
        home = np.clip(np.asarray(rc.get("home", np.zeros(len(acts))), float), lo, hi)
    info = dict(qadr=model.jnt_qposadr[jnt], dadr=model.jnt_dofadr[jnt], lo=lo, hi=hi, home=home,
                ee_sensor=model.sensor("ee_pos").adr[0])
    return model, info


# ------------------------------------------------------------------ reach --

class ReachEnv:
    """N copies of a fixed-base robot learning to touch random targets. Actions nudge the servo targets."""

    def __init__(self, cfg: dict, seed: int = 0, nthread: int | None = None, num_envs: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n = int(num_envs or cfg.get("envs", 1024))
        timestep = float(cfg.get("timestep", 0.004))
        self.dt = float(cfg.get("control_dt", 0.02))
        self.decimation = max(1, round(self.dt / timestep))
        base_dir = cfg.get("_base_dir", ".")
        self.base, info = build_robot_model(cfg["robot"], base_dir, timestep)
        self.qadr, self.dadr, self.lo, self.hi, self.home = (info[k] for k in ("qadr", "dadr", "lo", "hi", "home"))
        self.ee_adr = info["ee_sensor"]
        self.nu = self.base.nu
        self.action_scale = float(cfg.get("action_scale", 0.05))  # rad (or m) of servo target per step
        self.tol = float(cfg.get("success_radius", 0.02))
        self.obs_noise = float(cfg.get("obs_noise", 1.0))
        self.max_steps = int(float(cfg.get("episode_seconds", 4.0)) / self.dt)
        self.obs_dim, self.act_dim = 3 * self.nu + 9, self.nu

        self.models = self._variants(int(cfg.get("variants", 16)), cfg.get("randomize", {}))
        self.env_model = [self.models[i % len(self.models)] for i in range(self.n)]
        self.nthread = nthread or 16
        self.datas = [mujoco.MjData(self.base) for _ in range(self.nthread)]
        self.runner = rollout.Rollout(nthread=self.nthread)
        self._scratch = mujoco.MjData(self.base)
        self.nstate = mujoco.mj_stateSize(self.base, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.QP, self.QV = 1, 1 + self.base.nq
        mujoco.mj_forward(self.base, self._scratch)
        self.s0 = np.zeros(self.nstate)
        mujoco.mj_getState(self.base, self._scratch, self.s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        self.targets_pool = self._reachable_points(int(cfg.get("target_pool", 8192)))
        self.state = np.tile(self.s0, (self.n, 1))
        self._out_state = np.zeros((self.n, self.decimation, self.nstate))
        self._out_sens = np.zeros((self.n, self.decimation, self.base.nsensordata))
        self.ctrl = np.tile(self.home, (self.n, 1))
        self.actions = np.zeros((self.n, self.nu))
        self.ee = np.zeros((self.n, 3))
        self.target = np.zeros((self.n, 3))
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.ep_return = np.zeros(self.n)
        self._reset(np.arange(self.n))
        self.steps[:] = self.rng.integers(0, self.max_steps, self.n)  # desynchronize resets

    # ---------------------------------------------------------------- setup --
    def _variants(self, k, rand):
        if not rand or k <= 1:
            return [self.base]
        robot_bodies = [b for b in range(self.base.nbody) if self.base.body(b).name.startswith("r_")]
        out = []
        for _ in range(k):
            m = copy.copy(self.base)
            if "damping" in rand:
                m.dof_damping[:] = self.base.dof_damping * self.rng.uniform(*rand["damping"], self.base.nv)
            if "kp" in rand:
                s = self.rng.uniform(*rand["kp"], self.base.nu)
                m.actuator_gainprm[:, 0] = self.base.actuator_gainprm[:, 0] * s
                m.actuator_biasprm[:, 1] = self.base.actuator_biasprm[:, 1] * s
            if "mass" in rand:
                m.body_mass[robot_bodies] = self.base.body_mass[robot_bodies] * self.rng.uniform(*rand["mass"], len(robot_bodies))
            out.append(m)
        return out

    def _ee_of(self, q_rows):
        """End-effector positions for joint configurations (forward kinematics only)."""
        d = self._scratch
        out = np.empty((len(q_rows), 3))
        for i, q in enumerate(q_rows):
            d.qpos[:] = self.base.qpos0
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.base, d)
            mujoco.mj_sensorPos(self.base, d)
            out[i] = d.sensordata[self.ee_adr:self.ee_adr + 3]
        return out

    def _reachable_points(self, k):
        """Targets the robot can actually reach: end effector positions of random joint configurations."""
        margin = 0.1 * (self.hi - self.lo)
        q = self.rng.uniform(self.lo + margin, self.hi - margin, (k * 2, self.nu))
        pts = self._ee_of(q)
        pts = pts[pts[:, 2] > float(self.cfg.get("min_target_height", 0.05))]
        if len(pts) < 16:
            raise ValueError("could not find reachable targets above the floor; check the robot's joint ranges")
        return pts[:k]

    def _reset(self, idx):
        if len(idx) == 0:
            return
        n = len(idx)
        s = np.tile(self.s0, (n, 1))
        span = self.hi - self.lo
        q = np.clip(self.home + self.rng.uniform(-0.1, 0.1, (n, self.nu)) * span, self.lo, self.hi)
        ee = self._ee_of(q)
        for _ in range(20):  # a random start can put the hand in the floor: redraw those (found by a test)
            bad = ee[:, 2] < 0.1
            if not bad.any():
                break
            q[bad] = np.clip(self.home + self.rng.uniform(-0.1, 0.1, (bad.sum(), self.nu)) * span, self.lo, self.hi)
            ee[bad] = self._ee_of(q[bad])
        q[ee[:, 2] < 0.1] = self.home
        ee[ee[:, 2] < 0.1] = self._ee_of(self.home[None])[0]
        s[:, self.QP + self.qadr] = q
        self.state[idx] = s
        self.ctrl[idx] = q
        self.actions[idx] = 0.0
        self.ee[idx] = ee
        self.target[idx] = self.targets_pool[self.rng.integers(0, len(self.targets_pool), n)]
        self.steps[idx] = 0
        self.ep_return[idx] = 0.0

    # -------------------------------------------------------------- stepping --
    def observe(self) -> np.ndarray:
        q = self.state[:, self.QP + self.qadr]
        qd = self.state[:, self.QV + self.dadr]
        k = self.obs_noise
        span = self.hi - self.lo
        qn = 2 * (q - self.lo) / span - 1 + k * self.rng.normal(0, 0.005, q.shape)
        return np.concatenate([qn, 0.1 * qd, (self.ctrl - q) / span * 10, self.ee, self.target,
                               (self.target - self.ee) * 5], 1).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(np.asarray(actions, float), -1.0, 1.0)
        prev = self.actions
        self.actions = a
        self.ctrl = np.clip(self.ctrl + self.action_scale * a * (self.hi - self.lo) / 2, self.lo, self.hi)
        control = np.ascontiguousarray(np.repeat(self.ctrl[:, None, :], self.decimation, axis=1))
        self.runner.rollout(self.env_model, self.datas, np.ascontiguousarray(self.state), control,
                            nstep=self.decimation, skip_checks=True, state=self._out_state, sensordata=self._out_sens)
        self.state = self._out_state[:, -1].copy()
        self.ee = self._out_sens[:, -1, self.ee_adr:self.ee_adr + 3].copy()
        self.steps += 1

        dist = np.linalg.norm(self.target - self.ee, axis=1)
        qd = self.state[:, self.QV + self.dadr]
        rew = (-dist + 0.5 * np.exp(-dist / self.tol) - 0.02 * ((a - prev) ** 2).sum(1)
               - 0.002 * (qd ** 2).sum(1)) * self.dt * 10
        self.ep_return += rew
        timeout = self.steps >= self.max_steps
        done = timeout.copy()
        info = {"timeout": timeout, "success": dist < self.tol, "dist": dist.copy(),
                "ep_return": self.ep_return.copy(), "ep_steps": self.steps.copy(),
                "terminal_obs": self.observe() if timeout.any() else None}
        self._reset(np.nonzero(done)[0])
        return self.observe(), rew.astype(np.float32), done, timeout, info

    def close(self):
        self.runner.close()


# ------------------------------------------------------------------- walk --

class WalkEnv:
    """The ZOOMIES quadruped learning to walk from scratch. A thin adapter over zoomies.env.ZoomieEnv."""

    def __init__(self, cfg: dict, seed: int = 0, nthread: int | None = None, num_envs: int | None = None):
        from zoomies.env import ACT_DIM, OBS_DIM, EnvConfig, ZoomieEnv

        keys = {"push", "randomize", "episode_seconds", "push_speed", "obs_noise"}
        kw = {k: cfg[k] for k in keys if k in cfg}
        self.env = ZoomieEnv(EnvConfig(num_envs=int(num_envs or cfg.get("envs", 1024)), **kw), seed=seed,
                             nthread=nthread)
        self.n, self.obs_dim, self.act_dim, self.dt = self.env.n, OBS_DIM, ACT_DIM, 0.02

    def observe(self):
        return self.env.observe()

    def step(self, actions):
        obs, rew, done, timeout, info = self.env.step(actions)
        info["success"] = ~info["fell"]  # an episode that ends by time-out without falling
        return obs, rew, done, timeout, info

    def close(self):
        self.env.close()


def make_env(cfg: dict, seed: int = 0, num_envs: int | None = None, nthread: int | None = None):
    cls = {"reach": ReachEnv, "walk": WalkEnv}[cfg["task"]]
    return cls(cfg, seed=seed, num_envs=num_envs, nthread=nthread)


def default_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "tasks" / name
