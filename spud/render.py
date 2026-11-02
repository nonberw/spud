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
