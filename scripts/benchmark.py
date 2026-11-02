"""Reproduce SPUD's headline numbers on your machine -> benchmarks/results.json + benchmarks/spud.png

    python scripts/benchmark.py            # ~10 min on a laptop; needs ~4.5 GB free RAM for the 2B part
    python scripts/benchmark.py --quick    # skip the 2B model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spud.cli import POLICY  # noqa: E402
from spud.hw import detect  # noqa: E402
from spud.models import PRESETS, iter_blocks, make_blocks, params_of  # noqa: E402
from spud.stream import StreamedBlocks, resident_copy  # noqa: E402
from spud.zoomie_brain import TrainedBrain, ZoomieMass  # noqa: E402


def timed(fn, x, reps=3):
    fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, y


def sweep_1b(batches):
    depth, d, h = PRESETS["1b"]
    out = {"resident_ms": {}, "streamed_ms": {}, "bit_exact": {}}
    res = resident_copy(make_blocks(depth, d, h))
    refs = {}
    for b in batches:
        x = torch.randn(b, 1, d, dtype=torch.float16, device="cuda", generator=torch.Generator("cuda").manual_seed(b))
        t, refs[b] = timed(res, x)
        out["resident_ms"][b] = 1e3 * t
    torch.cuda.reset_peak_memory_stats()
    out["resident_peak_gib"] = None
    del res
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    st = StreamedBlocks(iter_blocks(depth, d, h))
    for b in batches:
        x = torch.randn(b, 1, d, dtype=torch.float16, device="cuda", generator=torch.Generator("cuda").manual_seed(b))
        t, y = timed(st, x)
        out["streamed_ms"][b] = 1e3 * t
        out["bit_exact"][b] = bool(torch.equal(y, refs[b]))
    out["streamed_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def two_b(batch=4096):
    depth, d, h = PRESETS["2b"]
    x = torch.randn(batch, 1, d, dtype=torch.float16, device="cuda")
    out = {"params_b": params_of(depth, d) / 1e9, "batch": batch}
    torch.cuda.reset_peak_memory_stats()
    st = StreamedBlocks(iter_blocks(depth, d, h))
    out["streamed_ms"], _ = timed(st, x)
    out["streamed_ms"] *= 1e3
    out["streamed_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del st
    torch.cuda.empty_cache()
    blocks = [b.cuda() for b in iter_blocks(depth, d, h)]  # the driver spills what doesn't fit to system RAM

    def fwd(z):
        with torch.no_grad():
            for b in blocks:
                z = b(z)
        return z

    t, _ = timed(fwd, x, reps=2)
    out["resident_spill_ms"] = 1e3 * t
    del blocks
    torch.cuda.empty_cache()
    return out


def mass(sizes, threads):
    brain = TrainedBrain(str(POLICY))
    out = []
    for n in sizes:
        for th in threads:
            zm = ZoomieMass(n, nthread=th)
            zm.commands[:] = (0.6, 0, 0)
            zm.step(brain(zm.observe()))
            t0 = time.perf_counter()
            for _ in range(30):
                zm.step(brain(zm.observe()))
            dt = time.perf_counter() - t0
            out.append({"robots": n, "threads": th, "robot_steps_per_s": n * 30 / dt, "upright": float(zm.upright().mean())})
            zm.sim.close()
    return out


def plot(r, path):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=110)
    for ax in axes:
        ax.set_facecolor("#0d0b1a")
        ax.grid(alpha=0.15)
    s = r["sweep_1b"]
    b = sorted(int(k) for k in s["resident_ms"])
    ratio = [s["streamed_ms"][k] / s["resident_ms"][k] for k in b]
    axes[0].plot(b, ratio, "o-", color="#2de2e6", lw=2.5)
    axes[0].axhline(1, color="#888", ls="--", lw=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("robots per forward pass")
    axes[0].set_ylabel("streamed time / resident time")
    axes[0].set_title("1B brain: streaming cost vanishes with batch")
    if "two_b" in r:
        t = r["two_b"]
        bars = axes[1].bar(["driver spill\n(\"just load it\")", "SPUD stream"], [t["resident_spill_ms"], t["streamed_ms"]],
                           color=["#ff5fa2", "#2de2e6"])
        for bar, v in zip(bars, [t["resident_spill_ms"], t["streamed_ms"]]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v / 1e3:.1f} s", ha="center", va="bottom")
        axes[1].set_ylabel("ms per step")
        axes[1].set_title(f"{t['params_b']:.1f}B brain, {t['batch']} robots, 4 GB GPU")
    m = r["mass"]
    for th, c in ((max(x["threads"] for x in m), "#2de2e6"), (min(x["threads"] for x in m), "#ffd166")):
        rows = [x for x in m if x["threads"] == th]
        axes[2].plot([x["robots"] for x in rows], [x["robot_steps_per_s"] / 1e3 for x in rows], "o-", color=c, lw=2.5,
                     label=f"{th} CPU threads")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("trained robots simulated")
    axes[2].set_ylabel("thousand robot-steps / s")
    axes[2].set_title("Many small worlds")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="#0d0b1a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    hw = detect()
    r = {"hardware": hw.as_dict()}
    print("1B sweep...", flush=True)
    r["sweep_1b"] = sweep_1b([64, 256, 1024, 4096, 8192])
    if not args.quick:
        print("2B: streamed vs driver spill...", flush=True)
        r["two_b"] = two_b()
    print("mass simulation...", flush=True)
    r["mass"] = mass([1024, 4096, 16384, 65536], [hw.cpu_threads, 2])
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(r, indent=2))
    plot(r, out / "spud.png")
    print(json.dumps({k: v for k, v in r.items() if k != "hardware"}, indent=1))


if __name__ == "__main__":
    main()
