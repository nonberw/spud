"""GIF of one big world: a crowd of trained Zoomies following a changing command."""

from __future__ import annotations

import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (start second, vx, vy, yaw, caption)
SCRIPT = [
    (0.0, 0.0, 0.0, 0.0, "standing by"),
    (1.0, 0.8, 0.0, 0.0, "everyone: forward"),
    (4.0, 0.0, 0.0, 1.0, "everyone: spin"),
    (6.5, 0.0, 0.4, 0.0, "everyone: strafe left"),
    (8.5, 0.6, 0.0, 0.0, "forward again"),
]


def _font(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _save_gif(frames, path, fps):
    sample = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0, disposal=1)


def render_world_gif(swarm, brain, path, seconds=10.0, width=480, height=300, fps=14):
    m, d = swarm.model, swarm.data
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    xy = swarm.positions()
    center = xy.mean(0)
    span = np.ptp(xy, axis=0).max() + 2
    cam.distance, cam.azimuth, cam.elevation = 1.15 * span, 135, -35
    big, small = _font(16), _font(12)
    frames, caption = [], ""
    every = max(1, round(1 / (0.02 * fps)))
    t_physics = 0.0
    for k in range(int(seconds / 0.02)):
        t = k * 0.02
        for start, vx, vy, yaw, text in SCRIPT:
            if t >= start:
                swarm.commands[:] = (vx, vy, yaw)
                caption = text
        t0 = time.perf_counter()
        swarm.step(brain(swarm.observe()))
        t_physics += time.perf_counter() - t0
        if k % every:
            continue
        target = swarm.positions().mean(0)
        xy_cam = 0.9 * cam.lookat[:2] + 0.1 * target if k else center
        cam.lookat[:] = [xy_cam[0], xy_cam[1], 0.2]
        renderer.update_scene(d, cam)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.text((12, 10), f"{swarm.k} trained Zoomies, one world, {m.nv:,} DoF", fill=(235, 230, 255), font=big)
        dr.text((12, 32), f"{caption}   t={t:4.1f}s   upright {swarm.upright().mean():.0%}", fill=(200, 190, 240), font=small)
        frames.append(np.asarray(img))
    renderer.close()
    _save_gif(frames, path, fps)
    print(f"wrote {path}: {len(frames)} frames; simulated {seconds:.0f}s in {t_physics:.1f}s of compute "
          f"({seconds / t_physics:.2f}x real time), upright at the end {swarm.upright().mean():.0%}")


def render_scene_gif(scene, path, seconds=8.0, width=520, height=320, fps=14):
    """GIF of a YAML scene: overview camera, lidar hits drawn in the world, a robot-camera inset."""
    from .sensors import Camera, Lidar

    m, d = scene.model, scene.data
    cams = [s for s in scene.sensors if isinstance(s, Camera)]
    lidars = [s for s in scene.sensors if isinstance(s, Lidar)]
    overview = next((c for c in cams if c.robot is None), None)
    onboard = next((c for c in cams if c.robot is not None), None)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    if overview is not None:
        cam.lookat[:] = overview.cam.lookat
        cam.distance, cam.azimuth, cam.elevation = overview.cam.distance, overview.cam.azimuth, overview.cam.elevation
    else:
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.azimuth, cam.elevation = 8.0, 135, -30
    big, small = _font(15), _font(12)
    frames, every = [], max(1, round(1 / (0.02 * fps)))
    t0 = time.perf_counter()
    for k in range(int(seconds / 0.02)):
        scene.step()
        if k % every:
            continue
        renderer.update_scene(d, cam)
        hits = 0
        for lid in lidars:
            pts = lid.read()["points"]
            hits += len(pts)
            scn = renderer.scene
            for p in pts[::2]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                                    p, np.eye(3).ravel(), np.array([1, 0.25, 0.4, 1], np.float32))
                scn.ngeom += 1
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        if onboard is not None:
            inset = Image.fromarray(onboard.read()["rgb"]).resize((176, 132))
            img.paste(inset, (width - 184, 8))
            dr.rectangle((width - 185, 7, width - 7, 141), outline=(180, 150, 255), width=2)
            dr.text((width - 182, 144), f"{onboard.name} ({onboard.robot})", fill=(220, 210, 255), font=small)
        n_robots = sum(len(g.names) for g in scene.groups)
        dr.text((10, 8), f"{scene.cfg.get('name', 'scene')}: {n_robots} robots, {len(scene.objects)} objects, quality={scene.quality.name}",
                fill=(235, 230, 255), font=big)
        up = ", ".join(f"{kind} {v:.0%} upright" for kind, v in scene.upright().items())
        dr.text((10, 28), f"t={k * 0.02:4.1f}s   lidar hits {hits}   {up}", fill=(200, 190, 240), font=small)
        frames.append(np.asarray(img))
    renderer.close()
    _save_gif(frames, path, fps)
    print(f"wrote {path}: {len(frames)} frames in {time.perf_counter() - t0:.1f}s")
