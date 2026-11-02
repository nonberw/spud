"""Sensors for SPUD scenes: cameras (RGB / depth / segmentation), lidar, IMU and contacts.

    sensors:
      - {type: camera, name: top, pos: [0, -6, 5], lookat: [0, 0, 0], width: 320, height: 240}
      - {type: camera, name: dogcam, robot: zoomie0, offset: [0.25, 0, 0.05]}   # rides on the robot
      - {type: lidar, name: scan, robot: zoomie0, rays: 360, channels: 8, fov: [-15, 15], range: 10}
      - {type: imu, name: imu, robot: zoomie0}
      - {type: contacts, name: touch, robot: zoomie0}

Every sensor has `read() -> dict` of numpy arrays.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _look(src, dst):
    """MjvCamera angles that put the camera at `src` looking at `dst`."""
    v = np.asarray(dst, float) - np.asarray(src, float)
    dist = np.linalg.norm(v)
    azimuth = np.degrees(np.arctan2(v[1], v[0]))
    elevation = np.degrees(np.arcsin(v[2] / max(dist, 1e-9)))
    return dist, azimuth, elevation


class Camera:
    _renderers: dict = {}

    def __init__(self, scene, cfg):
        self.scene, self.name = scene, cfg.get("name", "camera")
        q = scene.quality
        w, h = cfg.get("width", q.camera_size[0]), cfg.get("height", q.camera_size[1])
        self.width, self.height = min(w, q.max_render[0]), min(h, q.max_render[1])
        self.modes = cfg.get("modes", ["rgb", "depth", "segmentation"])
        self.robot = cfg.get("robot")
        self.offset = np.asarray(cfg.get("offset", [0.25, 0.0, 0.05]), float)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if self.robot is None:
            pos, lookat = cfg.get("pos", [0, -5, 4]), cfg.get("lookat", [0, 0, 0])
            self.cam.lookat[:] = lookat
            self.cam.distance, self.cam.azimuth, self.cam.elevation = _look(pos, lookat)
        else:
            self.body = scene.robot_body(self.robot)
        self.renderer = self._renderer(scene.model, msaa=True)
        # Segmentation encodes object ids as colors, so anti-aliasing at edges blends two ids into
        # a third, wrong one. Masks get their own renderer with MSAA off; RGB keeps it.
        self.seg_renderer = self._renderer(scene.model, msaa=False) if "segmentation" in self.modes else None
        # geom id -> object index (or -1 for robots / terrain), for instance masks
        self.geom_to_obj = np.full(scene.model.ngeom + 1, -1, np.int32)
        for k, o in enumerate(scene.objects):
            self.geom_to_obj[o.geom] = k

    def _renderer(self, model, msaa: bool):
        key = (id(model), self.width, self.height, msaa)
        if key not in Camera._renderers:
            samples = model.vis.quality.offsamples
            if not msaa:
                model.vis.quality.offsamples = 0  # read when the GL context is created
            Camera._renderers[key] = mujoco.Renderer(model, self.height, self.width)
            model.vis.quality.offsamples = samples
        return Camera._renderers[key]

    def _follow(self):
        d = self.scene.data
        R = d.xmat[self.body].reshape(3, 3)
        eye = d.xpos[self.body] + R @ self.offset
        ahead = eye + R @ np.array([1.0, 0.0, -0.15])
        self.cam.lookat[:] = ahead
        self.cam.distance, self.cam.azimuth, self.cam.elevation = _look(eye, ahead)

    def read(self) -> dict:
        if self.robot is not None:
            self._follow()
        r, d, out = self.renderer, self.scene.data, {}
        if "rgb" in self.modes:
            r.update_scene(d, self.cam)
            out["rgb"] = r.render().copy()
        if "depth" in self.modes:
            r.enable_depth_rendering()
            r.update_scene(d, self.cam)
            out["depth"] = r.render().copy()
            r.disable_depth_rendering()
        if "segmentation" in self.modes:
            if self.seg_renderer is None:
                self.seg_renderer = self._renderer(self.scene.model, msaa=False)
            sr = self.seg_renderer
            sr.enable_segmentation_rendering()
            sr.update_scene(d, self.cam)
            seg = sr.render().copy()
            sr.disable_segmentation_rendering()
            geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
            out["geom_id"] = geom
            out["instance"] = np.where(geom >= 0, self.geom_to_obj[np.clip(geom, 0, None)], -1)
        return out


class Lidar:
    def __init__(self, scene, cfg):
        self.scene, self.name = scene, cfg.get("name", "lidar")
        self.body = scene.robot_body(cfg["robot"]) if "robot" in cfg else 0
        self.offset = np.asarray(cfg.get("offset", [0.0, 0.0, 0.12]), float)
        self.range = float(cfg.get("range", 10.0))
        n, ch = int(cfg.get("rays", 360)), int(cfg.get("channels", 1))
        lo, hi = cfg.get("fov", [0.0, 0.0])
        az = np.linspace(-np.pi, np.pi, n, endpoint=False)
        el = np.radians(np.linspace(lo, hi, ch)) if ch > 1 else np.radians([lo])
        A, E = np.meshgrid(az, el)
        self.dirs_local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.shape = (ch, n)
        self._geom = np.zeros(len(self.dirs_local), np.int32)
        self._dist = np.zeros(len(self.dirs_local))

    def read(self) -> dict:
        m, d = self.scene.model, self.scene.data
        R = d.xmat[self.body].reshape(3, 3)
        origin = d.xpos[self.body] + R @ self.offset
        vec = (self.dirs_local @ R.T).ravel()
        mujoco.mj_multiRay(m, d, origin, vec, None, 1, self.body, self._geom, self._dist, None,
                           len(self.dirs_local), self.range)
        dist = np.where(self._dist < 0, np.inf, self._dist)
        hit = np.isfinite(dist)
        points = origin + vec.reshape(-1, 3) * np.where(hit, dist, 0.0)[:, None]
        return {"distance": dist.reshape(self.shape), "points": points[hit], "geom_id": self._geom.reshape(self.shape),
                "origin": origin}


class IMU:
    def __init__(self, scene, cfg):
        self.scene, self.name = scene, cfg.get("name", "imu")
        self.body = scene.robot_body(cfg["robot"])
        self._acc = np.zeros(6)

    def read(self) -> dict:
        m, d = self.scene.model, self.scene.data
        mujoco.mj_rnePostConstraint(m, d)
        # World-frame results rotated by hand: checked against MuJoCo's own <accelerometer>.
        # cacc already contains the -gravity pseudo-acceleration, so this *is* specific force.
        mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_BODY, self.body, self._acc, 0)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.body, vel, 0)
        R = d.xmat[self.body].reshape(3, 3)
        return {"quat": d.xquat[self.body].copy(), "gyro": R.T @ vel[:3], "accel": R.T @ self._acc[3:]}


class Contacts:
    def __init__(self, scene, cfg):
        self.scene, self.name = scene, cfg.get("name", "contacts")
        root = scene.robot_body(cfg["robot"])
        m = scene.model
        self.bodies = {b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[root]}

    def read(self) -> dict:
        m, d = self.scene.model, self.scene.data
        f = np.zeros(6)
        rows = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.bodies or b2 in self.bodies:
                mujoco.mj_contactForce(m, d, i, f)
                mine, other = (c.geom1, c.geom2) if b1 in self.bodies else (c.geom2, c.geom1)
                rows.append((m.geom(mine).name or str(mine), m.geom(other).name or str(other), float(f[0])))
        return {"contacts": rows, "total_normal_force": sum(r[2] for r in rows)}


KINDS = {"camera": Camera, "lidar": Lidar, "imu": IMU, "contacts": Contacts}


def build_sensors(scene, cfgs) -> list:
    out = []
    for c in cfgs or []:
        if c["type"] not in KINDS:
            raise ValueError(f"unknown sensor {c['type']!r} ({', '.join(KINDS)})")
        out.append(KINDS[c["type"]](scene, c))
    return out
