"""Synthetic datasets, Isaac-Replicator-style, on a laptop: domain-randomized frames with labels.

For every frame: objects are re-scattered (pose, yaw) and dropped so they settle physically,
recolored, the lights and the camera viewpoint are jittered, then RGB, depth and an instance
mask are rendered. Labels are COCO JSON: categories, per-object boxes, areas and visible pixels.

    dataset/
      images/000000.png   depth/000000.npy (float16 meters)   masks/000000.png (uint16 instance id + 1)
      annotations.json    (COCO: images, annotations with bbox [x, y, w, h], categories)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .sensors import Camera

MIN_PIXELS = 20


def _scatter(scene, rng):
    m, d = scene.model, scene.data
    regions = {}
    for oc in scene.cfg.get("objects", []):
        regions[oc.get("class", oc.get("type", "box"))] = np.asarray(oc.get("region", [[-1, -1], [1, 1]]), float)
    for o in scene.objects:
        jnt = m.body_jntadr[o.body]
        if jnt < 0:
            continue  # static object
        a = m.jnt_qposadr[jnt]
        lo, hi = regions.get(o.cls, np.array([[-1, -1], [1, 1]]))
        xy = rng.uniform(lo, hi)
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[a:a + 3] = (xy[0], xy[1], 0.15 + rng.uniform(0, 0.25))
        d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        d.qvel[m.jnt_dofadr[jnt]:m.jnt_dofadr[jnt] + 6] = 0
        m.geom_rgba[o.geom, :3] = rng.uniform(0.1, 0.95, 3)


def _jitter_lights_and_camera(scene, cam: Camera, base, rng):
    m = scene.model
    for i in range(m.nlight):
        m.light_dir[i] = base["light_dir"][i] + rng.normal(0, 0.35, 3)
        m.light_diffuse[i] = np.clip(base["light_diffuse"][i] * rng.uniform(0.5, 1.4), 0, 1)
    m.vis.headlight.ambient[:] = rng.uniform(0.1, 0.45)
    cam.cam.azimuth = base["azimuth"] + rng.uniform(-15, 15)
    cam.cam.elevation = base["elevation"] + rng.uniform(-8, 8)
    cam.cam.distance = base["distance"] * rng.uniform(0.85, 1.15)


def generate(scene, out, frames: int = 100, camera: str | None = None, settle: int = 40, seed: int = 0):
    out = Path(out)
    for sub in ("images", "depth", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cams = [s for s in scene.sensors if isinstance(s, Camera)]
    if not cams:
        raise ValueError("the scene needs at least one camera sensor")
    cam = next((c for c in cams if c.name == camera), cams[0]) if camera else cams[0]
    cam.modes = ["rgb", "depth", "segmentation"]
    rng = np.random.default_rng(seed)
    m, d = scene.model, scene.data
    base = {"light_dir": m.light_dir.copy(), "light_diffuse": m.light_diffuse.copy(),
            "azimuth": cam.cam.azimuth, "elevation": cam.cam.elevation, "distance": cam.cam.distance}
    classes = sorted({o.cls for o in scene.objects})
    cat_id = {c: i + 1 for i, c in enumerate(classes)}
    coco = {"info": {"description": f"SPUD synthetic dataset from scene '{scene.cfg.get('name', '?')}'", "seed": seed},
            "categories": [{"id": cat_id[c], "name": c} for c in classes], "images": [], "annotations": []}
    ann_id, t0, t_render = 1, time.perf_counter(), 0.0
    for f in range(frames):
        _scatter(scene, rng)
        mujoco.mj_forward(m, d)
        for _ in range(settle):  # let them fall and come to rest: physically plausible piles
            mujoco.mj_step(m, d)
        _jitter_lights_and_camera(scene, cam, base, rng)
        tr = time.perf_counter()
        r = cam.read()
        t_render += time.perf_counter() - tr
        name = f"{f:06d}"
        Image.fromarray(r["rgb"]).save(out / "images" / f"{name}.png")
        np.save(out / "depth" / f"{name}.npy", r["depth"].astype(np.float16))
        inst = r["instance"]
        Image.fromarray((inst + 1).astype(np.uint16)).save(out / "masks" / f"{name}.png")
        coco["images"].append({"id": f + 1, "file_name": f"images/{name}.png", "width": cam.width, "height": cam.height})
        for k in np.unique(inst):
            if k < 0:
                continue
            ys, xs = np.nonzero(inst == k)
            if len(xs) < MIN_PIXELS:
                continue
            o = scene.objects[k]
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            coco["annotations"].append({"id": ann_id, "image_id": f + 1, "category_id": cat_id[o.cls],
                                        "bbox": [x0, y0, x1 - x0, y1 - y0], "area": int(len(xs)),
                                        "instance": o.name, "mask_value": int(k) + 1, "iscrowd": 0})
            ann_id += 1
    (out / "annotations.json").write_text(json.dumps(coco))
    dt = time.perf_counter() - t0
    print(f"wrote {frames} frames, {ann_id - 1} labeled objects ({(ann_id - 1) / max(frames, 1):.1f}/frame) to {out}/ "
          f"in {dt:.1f}s ({frames / dt:.1f} frames/s incl. physics settling; rendering alone {frames / max(t_render, 1e-9):.0f} frames/s)")
    return coco
