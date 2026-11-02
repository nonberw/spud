"""spud: run big simulations and big models on a potato.

    spud doctor                                  what is this machine, really
    spud plan  --model 2b --robots 4096          will it fit, how fast, which tier
    spud bench --model 1b                        resident vs streamed: speed, VRAM, bit-exactness
    spud swarm --robots 65536                    many small worlds of trained Zoomies, all cores
    spud world --robots 256 [--gif out.gif]      one big world of trained Zoomies
    spud brain --model 2b --robots 4096          a too-big-for-your-GPU brain drives a swarm
    spud view  examples/yard.yaml                interactive scene from a YAML file, live sensors
    spud scene examples/yard.yaml --gif out.gif  run a scene headless: speed, sensor rates, GIF
    spud gen   examples/blocks.yaml --frames 200 synthetic dataset: RGB, depth, masks, boxes (JSON)

Every command takes --vram/--ram/--threads/--cpu-only to pretend the machine is weaker.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

POLICY = Path(__file__).with_name("assets") / "zoomies_policy.pt"


def _hw(args, measure=True):
    from .hw import apply_limits, detect

    hw = detect(measure=measure)
    if any(v is not None for v in (args.vram, args.ram, args.threads)) or args.cpu_only:
        hw = apply_limits(hw, vram=args.vram, ram=args.ram, threads=args.threads, cpu_only=args.cpu_only)
    return hw


def _print_hw(hw):
    tag = "  (limited: pretending to be weaker)" if hw.limited else ""
    print(f"CPU   {hw.cpu_threads} threads, {hw.cpu_gflops or 0:.0f} GFLOPS fp32{tag}")
    print(f"RAM   {hw.ram_free:.1f} GiB free of {hw.ram_total:.1f} GiB")
    if hw.gpu:
        print(f"GPU   {hw.gpu}: {hw.vram_free:.2f} GiB free of {hw.vram_total:.2f} GiB, "
              f"{hw.gpu_tflops or 0:.1f} TFLOPS fp16, host->GPU {hw.h2d_gbps or 0:.1f} GiB/s")
    else:
        print("GPU   none (CPU mode)")


def cmd_doctor(args):
    _print_hw(_hw(args))


def _model_numbers(name):
    from .models import PRESETS, params_of

    depth, d, heads = PRESETS[name]
    p = params_of(depth, d)
    return depth, d, heads, p, 2 * p, 2 * p // depth


def _calibrated_plan(hw, model, robots):
    import torch

    from .models import Block
    from .plan import plan_model
    from .stream import time_one_block

    depth, d, heads, p, wbytes, bbytes = _model_numbers(model)
    tb = None
    if hw.gpu:
        try:  # calibrate on one real block (needs ~one block of VRAM); scale to the full batch
            mb = min(robots, 2048)
            tb = time_one_block(Block(d, heads), torch.randn(mb, 1, d, dtype=torch.float16, device="cuda")) * robots / mb
        except torch.OutOfMemoryError:
            tb = None
        torch.cuda.empty_cache()
    return plan_model(hw, p, wbytes, bbytes, d, robots, block_seconds=tb, depth=depth)


def cmd_plan(args):
    hw = _hw(args)
    _print_hw(hw)
    depth, d, heads, p, wbytes, bbytes = _model_numbers(args.model)
    pl = _calibrated_plan(hw, args.model, args.robots)
    print(f"\nmodel {args.model}: {p / 1e9:.2f}B params, {wbytes / 2**30:.2f} GiB fp16, {depth} blocks of {bbytes / 2**20:.0f} MiB")
    print(f"plan  tier={pl.tier}  fits={pl.fits}  step={pl.step_seconds * 1e3:.0f} ms "
          f"({args.robots / pl.step_seconds:,.0f} robots/s)  bound by {pl.bound}")
    print(f"      peak VRAM {pl.peak_vram_gib:.2f} GiB, host store {pl.host_store_gib:.2f} GiB"
          + (f", micro-batch {pl.micro_batch:,}" if pl.micro_batch else ""))
    for n in pl.notes:
        print(f"      - {n}")


def cmd_bench(args):
    import torch

    from .models import iter_blocks, make_blocks
    from .stream import StreamedBlocks, resident_copy

    depth, d, heads, p, wbytes, _ = _model_numbers(args.model)
    x = torch.randn(args.robots, 1, d, dtype=torch.float16, device="cuda")

    def timeit(m):
        m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            y = m(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.reps, y

    print(f"model {args.model} ({p / 1e9:.2f}B params, {wbytes / 2**30:.2f} GiB), batch {args.robots}")
    ref = None
    if not args.skip_resident:
        torch.cuda.reset_peak_memory_stats()
        res = resident_copy(make_blocks(depth, d, heads))
        t, ref = timeit(res)
        print(f"resident  {t * 1e3:8.0f} ms   peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del res
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    st = StreamedBlocks(iter_blocks(depth, d, heads), store=args.store, n_slots=args.slots, micro_batch=args.micro_batch)
    t, y = timeit(st)
    print(f"streamed  {t * 1e3:8.0f} ms   peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB   store={args.store}")
    if ref is not None and not args.micro_batch:
        print(f"bit-exact {torch.equal(y, ref)}")


def cmd_swarm(args):
    from .zoomie_brain import TrainedBrain, ZoomieMass

    hw = _hw(args, measure=False)
    brain = TrainedBrain(str(args.policy), device="cpu")
    zm = ZoomieMass(args.robots, nthread=hw.cpu_threads, ram_budget_gib=args.sim_ram)
    zm.commands[:] = (args.vx, 0.0, args.yaw)
    if args.resume and Path(args.resume).exists():
        zm.actions = zm.sim.load(args.resume)["actions"]
        print(f"resumed from {args.resume} at step {zm.sim.steps}")
    x0 = zm.sim.qpos()[:, 0].copy()
    t0 = time.perf_counter()
    for k in range(args.steps):
        zm.step(brain(zm.observe()))
        if args.resume and (k + 1) % 50 == 0:
            zm.sim.save(args.resume, actions=zm.actions)
    dt = time.perf_counter() - t0
    print(f"{args.robots:,} trained Zoomies x {args.steps} steps on {hw.cpu_threads} threads: "
          f"{args.robots * args.steps / dt:,.0f} robot-steps/s, sim RAM {zm.sim.memory_gib:.2f} GiB (chunk {zm.sim.chunk:,})")
    print(f"upright {zm.upright().mean():.1%}, mean forward speed "
          f"{(zm.sim.qpos()[:, 0] - x0).mean() / (args.steps * 0.02):.2f} m/s (commanded {args.vx})")
    zm.sim.close()


def cmd_world(args):
    from .swarm import Swarm
    from .zoomie_brain import TrainedBrain

    _hw(args, measure=False)
    t0 = time.perf_counter()
    sw = Swarm(args.robots, spacing=args.spacing)
    print(f"built one world with {args.robots} robots ({sw.model.nv:,} DoF) in {time.perf_counter() - t0:.1f}s")
    brain = TrainedBrain(str(args.policy), device="cpu")
    if args.gif:
        from .render import render_world_gif

        render_world_gif(sw, brain, args.gif, seconds=args.seconds)
        return
    sw.commands[:] = (args.vx, 0.0, args.yaw)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        sw.step(brain(sw.observe()))
    dt = time.perf_counter() - t0
    print(f"{args.robots * args.steps / dt:,.0f} robot-steps/s ({args.steps / dt:.1f} control steps/s, real time is 50), "
          f"upright {sw.upright().mean():.0%}")


def cmd_brain(args):
    import numpy as np
    import torch

    from .zoomie_brain import BigBrain, ZoomieMass

    hw = _hw(args)
    depth, d, heads, p, wbytes, bbytes = _model_numbers(args.model)
    pl = _calibrated_plan(hw, args.model, args.robots)
    device = "cpu" if pl.tier == "cpu" else "cuda"
    store = "disk" if pl.tier == "disk" or (pl.tier == "cpu" and pl.host_store_gib == 0) else "ram"
    print(f"plan: tier={pl.tier} store={store} micro-batch={pl.micro_batch} (predicted {pl.step_seconds:.2f} s/step)")
    t0 = time.perf_counter()
    brain = BigBrain(depth, d, heads, device=device, store=store, micro_batch=pl.micro_batch,
                     resident=pl.tier == "resident")
    print(f"{p / 1e9:.2f}B-param brain ready in {time.perf_counter() - t0:.0f}s (random weights: a systems test, not a skill)")
    zm = ZoomieMass(args.robots, nthread=hw.cpu_threads)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    brain(zm.observe())
    t0 = time.perf_counter()
    for _ in range(args.steps):
        zm.step(np.nan_to_num(brain(zm.observe())))
    dt = time.perf_counter() - t0
    vram = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device == "cuda" else ""
    print(f"{args.robots:,} robots: {dt / args.steps:.2f} s/step, {args.robots * args.steps / dt:,.0f} robot-steps/s{vram}")
    zm.sim.close()


def _scene(args):
    from .quality import pick_quality
    from .scene import build_scene

    hw = _hw(args, measure=False)
    q = pick_quality(args.quality, hw)
    t0 = time.perf_counter()
    sc = build_scene(args.scene, quality=q)
    print(f"scene '{sc.cfg.get('name', args.scene)}': {sc.model.nbody} bodies, {sc.model.nv:,} DoF, "
          f"{sum(len(g.names) for g in sc.groups)} robots, {len(sc.objects)} objects, {len(sc.sensors)} sensors; "
          f"quality={q.name} ({'shadows' if q.shadows else 'no shadows'}); built in {time.perf_counter() - t0:.1f}s")
    return sc


def cmd_view(args):
    from .view import view

    view(_scene(args), args.out)


def cmd_scene(args):
    sc = _scene(args)
    if args.gif:
        from .render import render_scene_gif

        render_scene_gif(sc, args.gif, seconds=args.seconds)
        return
    t0 = time.perf_counter()
    for _ in range(args.steps):
        sc.step()
    dt = time.perf_counter() - t0
    print(f"physics + controllers: {args.steps * 0.02 / dt:.2f}x real time ({args.steps / dt:.0f} control steps/s), upright {sc.upright()}")
    for s in sc.sensors:
        s.read()
        t0 = time.perf_counter()
        for _ in range(5):
            s.read()
        print(f"  {type(s).__name__:8s} {s.name:10s} {5 / (time.perf_counter() - t0):7.0f} reads/s")


def cmd_gen(args):
    from .replicator import generate

    sc = _scene(args)
    generate(sc, args.out, frames=args.frames, camera=args.camera, settle=args.settle, seed=args.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spud", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        g = p.add_argument_group("pretend to be weaker hardware")
        g.add_argument("--vram", type=float, help="GPU memory cap in GiB")
        g.add_argument("--ram", type=float, help="RAM budget in GiB")
        g.add_argument("--threads", type=int, help="CPU threads")
        g.add_argument("--cpu-only", action="store_true", help="ignore the GPU")
        return p

    add("doctor", cmd_doctor, help="detect and measure this machine")
    p = add("plan", cmd_plan, help="will a model fit, and how fast")
    p.add_argument("--model", default="2b", choices=["tiny", "1b", "2b", "3b"])
    p.add_argument("--robots", type=int, default=4096)
    p = add("bench", cmd_bench, help="resident vs streamed")
    p.add_argument("--model", default="1b", choices=["tiny", "1b", "2b", "3b"])
    p.add_argument("--robots", type=int, default=4096)
    p.add_argument("--store", default="ram", choices=["ram", "disk"])
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--skip-resident", action="store_true")
    for name, fn, helptext in (("swarm", cmd_swarm, "many small worlds of trained Zoomies"),
                               ("world", cmd_world, "one big world of trained Zoomies")):
        p = add(name, fn, help=helptext)
        p.add_argument("--robots", type=int, default=65536 if name == "swarm" else 256)
        p.add_argument("--steps", type=int, default=100)
        p.add_argument("--vx", type=float, default=0.6)
        p.add_argument("--yaw", type=float, default=0.0)
        p.add_argument("--policy", type=Path, default=POLICY)
        if name == "swarm":
            p.add_argument("--sim-ram", type=float, default=1.0, help="RAM budget for the simulator buffers, GiB")
            p.add_argument("--resume", help="checkpoint file: resume from it if present, save to it every 50 steps")
        else:
            p.add_argument("--spacing", type=float, default=1.0)
            p.add_argument("--gif", help="render a GIF instead of benchmarking")
            p.add_argument("--seconds", type=float, default=10.0)
    p = add("brain", cmd_brain, help="a big streamed brain drives a swarm")
    p.add_argument("--model", default="2b", choices=["tiny", "1b", "2b", "3b"])
    p.add_argument("--robots", type=int, default=4096)
    p.add_argument("--steps", type=int, default=3)

    for name, fn, helptext in (("view", cmd_view, "interactive scene window"),
                               ("scene", cmd_scene, "run a scene headless"),
                               ("gen", cmd_gen, "generate a synthetic dataset")):
        p = add(name, fn, help=helptext)
        p.add_argument("scene", help="scene YAML file")
        p.add_argument("--quality", default="auto", choices=["auto", "potato", "laptop", "desktop"])
        if name == "view":
            p.add_argument("--out", default="captures")
        if name == "scene":
            p.add_argument("--steps", type=int, default=250)
            p.add_argument("--gif")
            p.add_argument("--seconds", type=float, default=8.0)
        if name == "gen":
            p.add_argument("--out", default="dataset")
            p.add_argument("--frames", type=int, default=100)
            p.add_argument("--camera", help="camera sensor name (default: the first camera)")
            p.add_argument("--settle", type=int, default=40, help="physics steps before each capture")
            p.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
