"""Photoreal frames: the same MuJoCo scene, path-traced with Mitsuba 3.

Every visible geom is exported in its current pose (boxes, spheres, cylinders, capsules, planes,
heightfields, meshes), with a rough-plastic material in its color, a checkered ground, the scene's
lights as directional emitters plus sky light, and the *same* camera as the rasterizer. So a
photoreal RGB frame lines up pixel-for-pixel with MuJoCo's depth and segmentation masks.

Variant choice: cuda_ad_rgb (GPU) > llvm_ad_rgb (vectorized CPU) > scalar_rgb (always works).
On this laptop only scalar_rgb runs: DrJit refuses NVIDIA drivers 565-569 for OptiX, and the LLVM
variant needs LLVM-C.dll. A newer driver (>= 570) enables the GPU path.

It is slow next to the rasterizer (seconds per frame, not milliseconds): use it to make datasets
look less like a video game, not for real-time viewing.
"""

from __future__ import annotations

import os
import tempfile

import mujoco
import numpy as np

_VARIANT = None


def variant() -> str:
    global _VARIANT
    if _VARIANT:
        return _VARIANT
    import mitsuba as mi

    for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        if v not in mi.variants():
            continue
        try:
            mi.set_variant(v)
            mi.load_dict({"type": "scene", "s": {"type": "sphere"}})  # forces backend init
            _VARIANT = v
            return v
        except Exception:
            continue
    raise RuntimeError("no usable Mitsuba variant")


def _mat4(pos, R, scale=(1, 1, 1)):
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(scale)
    M[:3, 3] = pos
    return M


def _bsdf(rgb, rough=0.35):
    return {"type": "roughplastic", "distribution": "ggx", "alpha": rough,
            "diffuse_reflectance": {"type": "rgb", "value": [float(c) for c in rgb]}}


def _ground_bsdf():
    return {"type": "roughplastic", "alpha": 0.6, "diffuse_reflectance": {
        "type": "checkerboard", "color0": {"type": "rgb", "value": [0.13, 0.12, 0.2]},
        "color1": {"type": "rgb", "value": [0.09, 0.09, 0.14]},
        "to_uv": None}}


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def _hfield_mesh(m, hid):
    nr, nc = m.hfield_nrow[hid], m.hfield_ncol[hid]
    sx, sy, sz, _ = m.hfield_size[hid]
    z = m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nr * nc].reshape(nr, nc) * sz
    xs, ys = np.linspace(-sx, sx, nc), np.linspace(-sy, sy, nr)
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X.ravel(), Y.ravel(), z.ravel()], 1)
    idx = np.arange(nr * nc).reshape(nr, nc)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return verts, faces


def camera_pose(cam: mujoco.MjvCamera):
    """MjvCamera (free: lookat, distance, azimuth, elevation) -> eye position, target, up."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    target = np.asarray(cam.lookat, float)
    return target - cam.distance * fwd, target, np.array([0.0, 0.0, 1.0])


def scene_dict(model: mujoco.MjModel, data: mujoco.MjData, cam: mujoco.MjvCamera, width: int, height: int,
               spp: int = 64, max_depth: int = 6, tmp: str | None = None) -> dict:
    import mitsuba as mi

    variant()
    T = mi.ScalarTransform4f
    tmp = tmp or tempfile.mkdtemp(prefix="spud_mi_")
    eye, target, up = camera_pose(cam)
    d = {"type": "scene",
         "integrator": {"type": "path", "max_depth": max_depth},
         "sensor": {"type": "perspective", "fov": float(model.vis.global_.fovy), "fov_axis": "y",
                    "to_world": T().look_at(origin=eye.tolist(), target=target.tolist(), up=up.tolist()),
                    "film": {"type": "hdrfilm", "width": width, "height": height, "rfilter": {"type": "gaussian"}},
                    "sampler": {"type": "independent", "sample_count": spp}},
         "sky": {"type": "constant", "radiance": {"type": "rgb", "value": [0.2, 0.19, 0.26]}}}
    for i in range(model.nlight):
        dirn = data.light_xdir[i] if hasattr(data, "light_xdir") else model.light_dir[i]
        d[f"sun{i}"] = {"type": "directional", "direction": [float(x) for x in dirn],
                        "irradiance": {"type": "rgb", "value": [float(3.5 * c) for c in model.light_diffuse[i]]}}
    G = mujoco.mjtGeom
    for g in range(model.ngeom):
        rgba = model.geom_rgba[g]
        mat = model.geom_matid[g]
        if mat >= 0:
            rgba = model.mat_rgba[mat]
        if rgba[3] < 0.05 or model.geom_group[g] > 2:
            continue
        typ, size = model.geom_type[g], model.geom_size[g]
        pos, R = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        bsdf = _bsdf(rgba[:3])
        key = f"g{g}"
        if typ == G.mjGEOM_BOX:
            d[key] = {"type": "cube", "to_world": T(_mat4(pos, R, size)), "bsdf": bsdf}
        elif typ == G.mjGEOM_SPHERE:
            d[key] = {"type": "sphere", "center": pos.tolist(), "radius": float(size[0]), "bsdf": bsdf}
        elif typ in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
            r, h = float(size[0]), float(size[1])
            axis = R[:, 2]
            p0, p1 = pos - axis * h, pos + axis * h
            d[key] = {"type": "cylinder", "p0": p0.tolist(), "p1": p1.tolist(), "radius": r, "bsdf": bsdf}
            for k, (p, n) in enumerate(((p0, -axis), (p1, axis))):
                if typ == G.mjGEOM_CAPSULE:
                    d[f"{key}_{k}"] = {"type": "sphere", "center": p.tolist(), "radius": r, "bsdf": bsdf}
                else:
                    Rd = _frame(n)
                    d[f"{key}_{k}"] = {"type": "disk", "to_world": T(_mat4(p, Rd, (r, r, 1))), "bsdf": bsdf}
        elif typ == G.mjGEOM_PLANE:
            half = size[:2] if size[0] > 0 else (50, 50)
            d[key] = {"type": "rectangle", "to_world": T(_mat4(pos, R, (half[0], half[1], 1))), "bsdf": _ground_bsdf()}
            d[key]["bsdf"]["diffuse_reflectance"]["to_uv"] = mi.ScalarTransform3f().scale([half[0], half[1]])
        elif typ == G.mjGEOM_HFIELD:
            v, f = _hfield_mesh(model, model.geom_dataid[g])
            path = os.path.join(tmp, f"hf{g}.obj")
            _write_obj(path, v, f)
            d[key] = {"type": "obj", "filename": path, "to_world": T(_mat4(pos, R)), "bsdf": _bsdf([0.16, 0.15, 0.22], 0.7)}
        elif typ == G.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            path = os.path.join(tmp, f"mesh{mid}.obj")
            if not os.path.exists(path):
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                _write_obj(path, model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn])
            d[key] = {"type": "obj", "filename": path, "to_world": T(_mat4(pos, R)), "bsdf": bsdf}
    return d


def _frame(n):
    """Rotation whose z axis is n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(n, x), n], 1)


def render(model, data, cam, width: int, height: int, spp: int = 64, max_depth: int = 6) -> np.ndarray:
    """Path-traced uint8 RGB (H, W, 3), aligned with MuJoCo's own render of the same camera."""
    import mitsuba as mi

    sc = mi.load_dict(scene_dict(model, data, cam, width, height, spp, max_depth))
    img = np.asarray(mi.render(sc))
    srgb = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(np.clip(img, 0, None), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
