"""`spud view scene.yaml`: an interactive window onto a scene, with live sensors.

Keys: arrows = drive the dogs (forward/back, turn), space = pause, C = save every camera to PNG,
L = toggle lidar points. Physics runs at real time when the machine can keep up; the title line
in the terminal says how fast it actually goes.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

UP, DOWN, LEFT, RIGHT, SPACE = 265, 264, 263, 262, 32


def _draw_points(scn, points, rgba=(1.0, 0.3, 0.4, 1.0), size=0.015, limit=3000):
    scn.ngeom = 0
    for p in points[:limit]:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                            p, np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1


def view(scene, out_dir: str | Path = "captures"):
    from .sensors import Camera, Lidar

    state = {"paused": False, "lidar": True, "capture": False}
    dogs = [g for g in scene.groups if g.controller == "policy"]

    def on_key(key):
        for g in dogs:
            if key == UP:
                g.command[:, 0] = np.minimum(g.command[:, 0] + 0.2, 1.2)
            elif key == DOWN:
                g.command[:, 0] = np.maximum(g.command[:, 0] - 0.2, -0.6)
            elif key == LEFT:
                g.command[:, 2] = np.minimum(g.command[:, 2] + 0.3, 1.0)
            elif key == RIGHT:
                g.command[:, 2] = np.maximum(g.command[:, 2] - 0.3, -1.0)
        if key == SPACE:
            state["paused"] = not state["paused"]
        elif key == ord("C"):
            state["capture"] = True
        elif key == ord("L"):
            state["lidar"] = not state["lidar"]

    lidars = [s for s in scene.sensors if isinstance(s, Lidar)]
    cams = [s for s in scene.sensors if isinstance(s, Camera)]
    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=on_key) as v:
        v.cam.distance = 8.0
        v.cam.elevation = -30
        clock, steps, t_report = time.perf_counter(), 0, time.perf_counter()
        while v.is_running():
            if not state["paused"]:
                scene.step()
                steps += 1
            if state["lidar"] and lidars:
                pts = np.concatenate([l.read()["points"] for l in lidars]) if lidars else np.zeros((0, 3))
                with v.lock():
                    _draw_points(v.user_scn, pts)
            elif v.user_scn.ngeom:
                with v.lock():
                    v.user_scn.ngeom = 0
            if state["capture"]:
                state["capture"] = False
                _capture(cams, Path(out_dir))
            v.sync()
            clock += 0.02
            time.sleep(max(0.0, clock - time.perf_counter()))
            if time.perf_counter() - t_report > 2:
                cmd = dogs[0].command[0] if dogs else None
                print(f"\r{steps * 0.02 / (time.perf_counter() - t_report):.2f}x real time"
                      + (f" | dogs: vx {cmd[0]:+.1f} yaw {cmd[2]:+.1f}" if cmd is not None else "")
                      + f" | upright {scene.upright()}   ", end="", flush=True)
                steps, t_report = 0, time.perf_counter()
                clock = time.perf_counter()


def _capture(cams, out: Path):
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for c in cams:
        r = c.read()
        if "rgb" in r:
            Image.fromarray(r["rgb"]).save(out / f"{stamp}_{c.name}_rgb.png")
        if "depth" in r:
            d = r["depth"]
            d8 = (255 * (1 - np.clip(d / max(np.percentile(d, 99), 1e-6), 0, 1))).astype(np.uint8)
            Image.fromarray(d8).save(out / f"{stamp}_{c.name}_depth.png")
        if "instance" in r:
            Image.fromarray(colorize(r["instance"])).save(out / f"{stamp}_{c.name}_seg.png")
    print(f"\nsaved {len(cams)} camera(s) to {out}/")


def colorize(instance: np.ndarray) -> np.ndarray:
    """Stable random color per instance id; background (-1) black."""
    ids = instance.astype(np.int64)
    rng = np.random.default_rng(12345)
    lut = rng.integers(40, 255, (max(int(ids.max()) + 2, 2), 3), dtype=np.uint8)
    img = lut[np.clip(ids, -1, None) + 1]
    img[ids < 0] = 0
    return img
