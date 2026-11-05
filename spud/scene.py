"""Scenes from a config file, Isaac-style: robots, objects, terrain, lights and sensors in one YAML.

    name: yard
    terrain: {type: rough, size: [16, 16], height: 0.08}
    light: day
    robots:
      - {type: zoomie, count: 16, layout: grid, spacing: 1.2, command: [0.6, 0, 0]}
      - {type: arm, pos: [3, 0, 0]}
      - {type: urdf, file: my_robot.urdf, pos: [0, 4, 0.5], free: true}
    objects:
      - {type: box, size: [0.06, 0.06, 0.06], count: 30, region: [[-3, -3], [3, 3]], rgba: random}
    sensors:
      - {type: camera, name: top, pos: [0, -6, 5], lookat: [0, 0, 0], width: 320, height: 240}
      - {type: lidar, robot: zoomie0, rays: 180, range: 8}

Robot types: `zoomie` (trained quadruped from ZOOMIES, walks with its policy), `arm`
(ARMageddon 6-DoF arm, holds its home pose), `urdf` / `mjcf` (your own file). Every robot gets
a name prefix: `zoomie0_`, `zoomie1_`, ..., `arm0_`, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .quality import Quality, pick_quality

ZOOMIE_DT = 0.005  # the ZOOMIES policy was trained at this physics step (4 substeps = 50 Hz control)


# ------------------------------------------------------------------ config --

def load_config(src) -> dict:
    if isinstance(src, dict):
        return src
    import yaml

    path = Path(src)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("_base_dir", str(path.parent))
    return cfg


# ----------------------------------------------------------------- terrain --

def _smooth_noise(n: int, rng, octaves=4) -> np.ndarray:
    """Cheap fractal noise in [0, 1]: sum of upsampled random grids."""
    out = np.zeros((n, n))
    for o in range(octaves):
        cells = 2 ** (o + 2)
        coarse = rng.random((cells + 1, cells + 1))
        xs = np.linspace(0, cells, n)
        i = np.minimum(xs.astype(int), cells - 1)
        t = xs - i
        t = t * t * (3 - 2 * t)  # smoothstep
        rows = coarse[i] * (1 - t)[:, None] + coarse[i + 1] * t[:, None]
        grid = rows[:, i] * (1 - t)[None, :] + rows[:, i + 1] * t[None, :]
        out += grid / 2**o
    out -= out.min()
    return out / max(out.max(), 1e-9)


def _add_terrain(spec: mujoco.MjSpec, cfg: dict, rng) -> float:
    """Returns the highest terrain point, so robots spawn above it."""
    t = cfg.get("terrain", {"type": "flat"})
    kind = t.get("type", "flat")
    sx, sy = t.get("size", [20, 20])
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.13, 0.12, 0.2], rgb2=[0.08, 0.08, 0.13], width=256, height=256,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.45, 0.3, 0.9])
    mat = spec.add_material(name="ground", reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [sx / 2, sy / 2]
    if kind == "rough":
        n = int(t.get("resolution", 128))
        h = float(t.get("height", 0.08))
        spec.add_hfield(name="terrain", nrow=n, ncol=n, size=[sx / 2, sy / 2, h, 0.1],
                        userdata=list(_smooth_noise(n, rng).ravel()))
        spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                material="ground", contype=0, conaffinity=1)
        # Flat ground beyond the patch: robots that walk off its edge keep walking instead of falling
        # out of the world (found by the ROS 2 CI run, where dogs walked for minutes).
        spec.worldbody.add_geom(name="ground_beyond", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1],
                                material="ground", contype=0, conaffinity=1)
        return h
    spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[sx / 2, sy / 2, 0.1],
                            material="ground", contype=0, conaffinity=1)
    if kind == "stairs":
        step_h, depth, count = t.get("step_height", 0.05), t.get("step_depth", 0.4), t.get("steps", 8)
        x0, width = t.get("start", 2.0), t.get("width", 4.0)
        for k in range(count):
            hgt = step_h * (k + 1)
            spec.worldbody.add_geom(name=f"step{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=[x0 + depth * (k + 0.5), 0, hgt / 2], size=[depth / 2, width / 2, hgt / 2],
                                    rgba=[0.3, 0.28, 0.4, 1], contype=0, conaffinity=1)
        return 0.0
    if kind != "flat":
        raise ValueError(f"unknown terrain type {kind!r} (flat, rough, stairs)")
    return 0.0


LIGHTS = {
    "day": [dict(pos=[0, 0, 6], dir=[0.2, 0.3, -1], diffuse=[0.9, 0.88, 0.85], castshadow=True)],
    "studio": [dict(pos=[3, -3, 5], dir=[-0.5, 0.5, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True),
               dict(pos=[-3, -3, 4], dir=[0.5, 0.5, -1], diffuse=[0.4, 0.4, 0.45], castshadow=False),
               dict(pos=[0, 4, 4], dir=[0, -0.6, -1], diffuse=[0.3, 0.3, 0.35], castshadow=False)],
    "night": [dict(pos=[0, 0, 5], dir=[0, 0, -1], diffuse=[0.25, 0.25, 0.4], castshadow=True)],
}


def _add_lights(spec, cfg, quality: Quality):
    preset = cfg.get("light", "day")
    for i, l in enumerate(LIGHTS[preset] if isinstance(preset, str) else preset):
        spec.worldbody.add_light(name=f"light{i}", pos=l["pos"], dir=l["dir"], diffuse=l["diffuse"],
                                 castshadow=bool(l.get("castshadow", False)) and quality.shadows)


# ------------------------------------------------------------------ robots --

def _strip_world(child: mujoco.MjSpec, keep_floor=False):
    """Remove the parts of a robot file that would duplicate the scene's own world."""
    for key in list(child.keys):
        child.delete(key)
    for light in list(child.lights):
        child.delete(light)
    if not keep_floor:
        for g in list(child.worldbody.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:  # a second infinite floor = k^2 contacts
                child.delete(g)


def _robot_spec(kind: str, rc: dict, base_dir: str) -> mujoco.MjSpec:
    if kind == "zoomie":
        from zoomies.robot import build_xml

        child = mujoco.MjSpec.from_string(build_xml())
    elif kind == "arm":
        from armageddon import default_arm
        from armageddon.mjcf import ARM_MATERIALS, arm_actuators, arm_xml

        chain = default_arm()
        grip = rc.get("gripper", "parallel")
        xml = f"""<mujoco><compiler angle="radian" autolimits="true"/>
          <default><joint damping="3" armature="0.01"/><geom contype="1" conaffinity="0"/></default>
          <asset>{ARM_MATERIALS}</asset>
          <worldbody>{arm_xml(chain, gripper=grip)}</worldbody>
          <actuator>{arm_actuators(chain, gripper=grip)}</actuator></mujoco>"""
        child = mujoco.MjSpec.from_string(xml)
    elif kind in ("urdf", "mjcf"):
        path = Path(rc["file"])
        if not path.is_absolute():
            path = Path(base_dir) / path
        child = mujoco.MjSpec.from_file(str(path))
        if rc.get("free"):
            root = child.worldbody.first_body()
            if root is not None and not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints):
                root.add_freejoint()
    else:
        raise ValueError(f"unknown robot type {kind!r} (zoomie, arm, urdf, mjcf)")
    _strip_world(child)
    return child


def _layout(count, layout, spacing, origin, rng, region=None):
    origin = np.asarray(origin, float)
    if layout == "grid":
        side = int(np.ceil(np.sqrt(count)))
        pts = [(i % side, i // side) for i in range(count)]
        offs = np.array(pts, float) * spacing
        return origin[None, :2] + offs - offs.mean(0)
    if layout == "line":
        return origin[None, :2] + np.stack([np.arange(count) * spacing, np.zeros(count)], 1)
    if layout == "random":
        lo, hi = np.asarray(region or [[-5, -5], [5, 5]], float)
        out = []
        while len(out) < count:
            p = rng.uniform(lo, hi)
            if all(np.linalg.norm(p - q) > spacing for q in out):
                out.append(p)
        return np.array(out)
    raise ValueError(f"unknown layout {layout!r} (grid, line, random)")


@dataclass
class RobotGroup:
    kind: str
    names: list[str]
    controller: str
    command: np.ndarray
    qpos_idx: np.ndarray | None = None
    qvel_idx: np.ndarray | None = None
    ctrl_idx: np.ndarray | None = None
    actions: np.ndarray | None = None
    home: np.ndarray | None = None
    gravity_dofs: np.ndarray | None = None  # dofs that get gravity compensation (arms)


@dataclass
class ObjectInfo:
    name: str
    cls: str
    body: int = -1
    geom: int = -1


@dataclass
class Scene:
    cfg: dict
    quality: Quality
    model: mujoco.MjModel
    data: mujoco.MjData
    groups: list[RobotGroup]
    objects: list[ObjectInfo]
    substeps: int
    policy: object = None
    sensors: list = field(default_factory=list)

    # ---------------------------------------------------------------- step --
    def step(self):
        """One control step (50 Hz): run every robot controller, then the physics substeps."""
        for g in self.groups:
            if g.controller == "policy":
                obs = _zoomie_obs(self.data, g)
                act = self.policy(obs)
                g.actions = np.clip(act, -5.0, 5.0)
                from zoomies.env import ACTION_SCALE, Q_HIGH, Q_LOW
                from zoomies.robot import DEFAULT_Q

                self.data.ctrl[g.ctrl_idx] = np.clip(DEFAULT_Q + ACTION_SCALE * g.actions, Q_LOW, Q_HIGH)
            elif g.controller == "hold" and g.home is not None:
                self.data.ctrl[g.ctrl_idx] = g.home
        comp = [g.gravity_dofs for g in self.groups if g.gravity_dofs is not None]
        comp = np.concatenate(comp) if comp else None
        for _ in range(self.substeps):
            if comp is not None:  # feed forward what gravity costs, so position servos don't sag
                self.data.qfrc_applied[comp] = self.data.qfrc_bias[comp]
            mujoco.mj_step(self.model, self.data)

    def robot_body(self, name: str) -> int:
        """Root body id of a robot by its name (`zoomie0`, `arm1`, ...)."""
        for g in self.groups:
            if name in g.names:
                prefix = name + "_"
                for b in range(self.model.nbody):
                    if self.model.body(b).name.startswith(prefix) and self.model.body_parentid[b] == 0:
                        return b
        raise KeyError(name)

    def upright(self) -> dict:
        out = {}
        for g in self.groups:
            if g.kind == "zoomie":
                q = self.data.qpos[g.qpos_idx]
                from zoomies.env import quat_to_mat

                R = quat_to_mat(q[:, 3:7])
                out[g.kind] = float(((-R[:, 2, 2] < -0.5) & (q[:, 2] > 0.12)).mean())
        return out


def _zoomie_obs(data, g: RobotGroup) -> np.ndarray:
    from zoomies.env import quat_to_mat
    from zoomies.robot import DEFAULT_Q

    qpos, qvel = data.qpos[g.qpos_idx], data.qvel[g.qvel_idx]
    R = quat_to_mat(qpos[:, 3:7])
    lin = np.einsum("nji,nj->ni", R, qvel[:, 0:3])
    return np.concatenate([lin * 2.0, qvel[:, 3:6] * 0.25, -R[:, 2, :], g.command * np.array([2.0, 2.0, 0.25]),
                           qpos[:, 7:] - DEFAULT_Q, qvel[:, 6:] * 0.05, g.actions], 1).astype(np.float32)


# ----------------------------------------------------------------- objects --

PRIMITIVES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}


def _add_objects(spec, cfg, rng, ground_z) -> list[ObjectInfo]:
    out = []
    for oc in cfg.get("objects", []):
        kind = oc.get("type", "box")
        cls = oc.get("class", kind)
        count = int(oc.get("count", 1))
        if "pos" in oc and count == 1:
            spots = np.array([oc["pos"][:2]])
        else:
            spots = _layout(count, "random", oc.get("min_gap", 0.15), [0, 0], rng, oc.get("region"))
        for k, xy in enumerate(spots):
            name = f"{cls}{len([o for o in out if o.cls == cls])}"
            z = (oc["pos"][2] if "pos" in oc and len(oc["pos"]) > 2 else ground_z + oc.get("drop_height", 0.3))
            body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], z],
                                           quat=_yaw_quat(rng.uniform(-np.pi, np.pi)) if oc.get("random_yaw", True) else [1, 0, 0, 0])
            if not oc.get("static", False):
                body.add_freejoint()
            rgba = oc.get("rgba", [0.8, 0.5, 0.2, 1])
            if rgba == "random":
                rgba = [*rng.uniform(0.15, 0.95, 3), 1.0]
            geom_kw = dict(name=name, rgba=rgba, contype=1, conaffinity=1, condim=3)
            if kind == "mesh":
                mesh_name = f"mesh_{name}"
                path = Path(oc["file"])
                if not path.is_absolute():
                    path = Path(cfg.get("_base_dir", ".")) / path
                spec.add_mesh(name=mesh_name, file=str(path), scale=oc.get("scale", [1, 1, 1]))
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, mass=oc.get("mass", 0.2), **geom_kw)
            else:
                size = oc.get("size", [oc.get("radius", 0.05)] * 3)
                body.add_geom(type=PRIMITIVES[kind], size=list(size) + [0] * (3 - len(size)),
                              mass=oc.get("mass", 0.2), **geom_kw)
            out.append(ObjectInfo(name, cls))
    return out


def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


# -------------------------------------------------------------------- build --

def build_scene(src, quality: str | Quality | None = None, seed: int | None = None) -> Scene:
    cfg = load_config(src)
    rng = np.random.default_rng(cfg.get("seed", 0) if seed is None else seed)
    q = quality if isinstance(quality, Quality) else pick_quality(quality or cfg.get("quality", "auto"))
    spec = mujoco.MjSpec()
    spec.modelname = cfg.get("name", "spud_scene")
    spec.compiler.degree = False
    spec.option.timestep = ZOOMIE_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = q.solver_iterations
    spec.visual.quality.shadowsize = q.shadow_size
    spec.visual.global_.offwidth = q.max_render[0]
    spec.visual.global_.offheight = q.max_render[1]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.33]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.42]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.16, 0.09, 0.32], rgb2=[0.01, 0.01, 0.04], width=256, height=256)
    ground = _add_terrain(spec, cfg, rng)
    _add_lights(spec, cfg, q)

    base_dir = cfg.get("_base_dir", ".")
    groups = []
    for rc in cfg.get("robots", []):
        kind = rc["type"]
        count = int(rc.get("count", 1))
        n_before = sum(len(g.names) for g in groups if g.kind == kind)
        spots = (np.array([rc["pos"][:2]] * count) if "pos" in rc and count == 1 else
                 _layout(count, rc.get("layout", "grid"), rc.get("spacing", 1.0), rc.get("origin", [0, 0]), rng,
                         rc.get("region")))
        names = []
        for k in range(count):
            name = f"{kind}{n_before + k}"
            z = rc["pos"][2] if "pos" in rc and len(rc["pos"]) > 2 else (ground if kind == "zoomie" else 0.0)
            frame = spec.worldbody.add_frame(pos=[spots[k][0], spots[k][1], z])
            spec.attach(_robot_spec(kind, rc, base_dir), prefix=name + "_", frame=frame)
            names.append(name)
        ctrl = rc.get("controller", "policy" if kind == "zoomie" else "hold")
        groups.append(RobotGroup(kind, names, ctrl, np.tile(np.asarray(rc.get("command", [0, 0, 0]), float), (count, 1))))
    objects = _add_objects(spec, cfg, rng, ground)

    model = spec.compile()
    data = mujoco.MjData(model)
    policy = None
    for g in groups:
        _bind_group(model, data, g)
        if g.controller == "policy":
            if g.kind != "zoomie":
                raise ValueError("controller: policy is only available for zoomie robots")
            if policy is None:
                from .zoomie_brain import TrainedBrain

                policy = TrainedBrain(str(Path(__file__).with_name("assets") / "zoomies_policy.pt"))
    for o in objects:
        o.body = model.body(o.name).id
        o.geom = model.geom(o.name).id
    mujoco.mj_forward(model, data)
    scene = Scene(cfg, q, model, data, groups, objects, round(0.02 / model.opt.timestep), policy)
    from .sensors import build_sensors

    scene.sensors = build_sensors(scene, cfg.get("sensors", []))
    return scene


def _bind_group(model, data, g: RobotGroup):
    """Index arrays into qpos/qvel/ctrl for every robot of a group, and its starting pose."""
    if g.kind == "zoomie":
        from zoomies.robot import DEFAULT_Q, STAND_HEIGHT

        qa, va, ca = [], [], []
        joints = [f"{leg}_{j}" for leg in ("FL", "FR", "HL", "HR") for j in ("roll", "pitch", "knee")]
        for name in g.names:
            root = model.body(f"{name}_torso")
            j0 = root.jntadr[0]
            qa.append(model.jnt_qposadr[j0] + np.arange(19))
            va.append(model.jnt_dofadr[j0] + np.arange(18))
            ca.append([model.actuator(f"{name}_{j}").id for j in joints])
        g.qpos_idx, g.qvel_idx, g.ctrl_idx = np.array(qa), np.array(va), np.array(ca)
        q = data.qpos
        for r in range(len(g.names)):
            idx = g.qpos_idx[r]
            q[idx[2]] += STAND_HEIGHT
            q[idx[3:7]] = [1, 0, 0, 0]
            q[idx[7:]] = DEFAULT_Q
        g.actions = np.zeros((len(g.names), 12))
        data.ctrl[g.ctrl_idx] = DEFAULT_Q
    elif g.kind == "arm":
        from armageddon import HOME

        ca, qa = [], []
        for name in g.names:
            acts = [a for a in range(model.nu) if model.actuator(a).name.startswith(name + "_")]
            ca.append(acts)
            qa.append([model.jnt_qposadr[model.actuator_trnid[a][0]] for a in acts])
        g.ctrl_idx = np.array(ca)
        g.gravity_dofs = np.array([model.jnt_dofadr[model.actuator_trnid[a][0]] for row in ca for a in row[:6]])
        home = np.concatenate([HOME, [0.04, 0.04]])
        for r in range(len(g.names)):
            data.qpos[qa[r]] = home[: len(qa[r])]
        g.home = np.tile(home[: g.ctrl_idx.shape[1]], (len(g.names), 1))
        data.ctrl[g.ctrl_idx] = g.home
    else:
        acts = [[a for a in range(model.nu) if model.actuator(a).name.startswith(n + "_")] for n in g.names]
        if acts and all(acts) and len({len(a) for a in acts}) == 1:
            g.ctrl_idx = np.array(acts)
            g.home = np.array([[data.qpos[model.jnt_qposadr[model.actuator_trnid[a][0]]] for a in row] for row in acts])
        else:
            g.controller = "none"
