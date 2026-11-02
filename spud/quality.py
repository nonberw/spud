"""Quality presets picked from the hardware: the part Isaac Sim doesn't do for you.

A preset decides how expensive the *pretty* parts are (shadows, render size, anti-aliasing) and
how many robots SPUD suggests per world, so a scene that looks fine on a desktop still runs on a
2-core laptop with integrated graphics. Physics accuracy is not traded away: the timestep stays
what the robots were trained at.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quality:
    name: str
    shadows: bool
    shadow_size: int
    max_render: tuple[int, int]     # offscreen framebuffer, also the camera-sensor size cap
    camera_size: tuple[int, int]    # default sensor camera resolution
    solver_iterations: int
    robots_per_world: int           # above this, prefer MassSim's many small worlds
    viewer_fps: int


PRESETS = {
    "potato": Quality("potato", False, 512, (640, 480), (160, 120), 30, 64, 30),
    "laptop": Quality("laptop", True, 2048, (1280, 720), (320, 240), 50, 256, 60),
    "desktop": Quality("desktop", True, 4096, (1920, 1080), (640, 480), 100, 1024, 60),
}


def pick_quality(name: str = "auto", hw=None) -> Quality:
    if name != "auto":
        if name not in PRESETS:
            raise ValueError(f"quality must be auto or one of {list(PRESETS)}")
        return PRESETS[name]
    if hw is None:
        from .hw import detect

        hw = detect(measure=False)
    return PRESETS[auto_preset(hw)]


def auto_preset(hw) -> str:
    """Rules of thumb, written down so they can be argued with:
    no discrete GPU or <= 4 CPU threads -> potato; < 6 GB VRAM or < 12 threads -> laptop; else desktop."""
    if not hw.gpu or hw.cpu_threads <= 4 or hw.ram_total < 6:
        return "potato"
    if hw.vram_total < 6 or hw.cpu_threads < 12:
        return "laptop"
    return "desktop"
