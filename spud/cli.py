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
    spud ros   examples/yard.yaml                publish the scene to ROS 2 (rclpy or rosbridge)

  robot learning
    spud train examples/tasks/reach.yaml         PPO on a task file (any URDF arm, or the dog): runs/<name>/
    spud play  runs/reach-.../policy.pt --gif    evaluate a trained policy (PPO or BC), render a GIF
    spud teleop --input phone --record d.hdf5    drive the arm from your phone or keyboard, record demos
    spud demos --episodes 200 --out d.hdf5       or let the scripted expert record them (DART noise)
    spud bc d.hdf5 --out bc.pt                   behavior cloning on the demos, then evaluate it

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
    if args.backend == "gpu":
        return _swarm_gpu(args)
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
    print(f"{args.robots:,} trained Zoomies x {args.steps} steps on {hw.cpu_threads} CPU threads: "
          f"{args.robots * args.steps / dt:,.0f} robot-steps/s, sim RAM {zm.sim.memory_gib:.2f} GiB (chunk {zm.sim.chunk:,})")
    print(f"upright {zm.upright().mean():.1%}, mean forward speed "
          f"{(zm.sim.qpos()[:, 0] - x0).mean() / (args.steps * 0.02):.2f} m/s (commanded {args.vx})")
    zm.sim.close()


def _swarm_gpu(args):
    import torch

    from .gpu import GpuZoomies, available

    ok, why = available()
    if not ok:
        print(f"GPU physics unavailable: {why}. Falling back to the CPU backend.")
        args.backend = "cpu"
        return cmd_swarm(args)
    g = GpuZoomies(args.robots, str(args.policy))
    g.commands[:, 0], g.commands[:, 2] = args.vx, args.yaw
    g.step()
    torch.cuda.synchronize()
    x0 = g.qpos[:, 0].clone()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        g.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{args.robots:,} trained Zoomies x {args.steps} steps on the GPU (MuJoCo Warp, policy on GPU): "
          f"{args.robots * args.steps / dt:,.0f} robot-steps/s")
    print(f"upright {g.upright():.1%}, mean forward speed {(g.qpos[:, 0] - x0).mean().item() / (args.steps * 0.02):.2f} m/s "
          f"(commanded {args.vx})")


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
    generate(sc, args.out, frames=args.frames, camera=args.camera, settle=args.settle, seed=args.seed,
             photoreal=args.photoreal, spp=args.spp)


def cmd_ros(args):
    from .ros import Bridge, make_transport

    sc = _scene(args)
    tx = make_transport(args.transport, args.url)
    print(f"ROS 2 bridge up via {type(tx).__name__}; publishing /clock, /tf, odom, joint_states and sensors; "
          f"send geometry_msgs/Twist to /<robot>/cmd_vel")
    try:
        Bridge(sc, tx, scan_hz=args.scan_hz, camera_hz=args.camera_hz).run(seconds=args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        tx.close()


def cmd_train(args):
    from .rl import evaluate, train
    from .tasks import load_task

    task = load_task(args.task)
    out = Path(args.out) if args.out else Path("runs") / f"{Path(args.task).stem}-{time.strftime('%m%d-%H%M%S')}"
    print(f"training {task['task']} -> {out}/")
    res = train(task, out, iters=args.iters, seed=args.seed, num_envs=args.envs,
                device="cpu" if args.cpu_only else None)
    print(f"done in {res['seconds'] / 60:.1f} min, {res['env_steps'] / 1e6:.1f}M env steps; "
          f"best training success {res['best_success']:.1%} at iteration {res['best_iter']}")
    ev = evaluate(out / "policy.pt", episodes=256)
    print(f"evaluation on 256 fresh episodes: {ev}")
    print(f"saved {out}/policy.pt, policy.onnx, progress.csv, curve.png")


def cmd_play(args):
    import torch

    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    if "chunk" in ck:  # a behavior-cloning policy for the pick task
        from .bc import BCAgent, evaluate_bc

        print(evaluate_bc(args.policy, episodes=args.episodes))
        if args.gif:
            _pick_gif(BCAgent(args.policy), args.gif)
        return
    from .rl import evaluate, render_gif

    print(evaluate(args.policy, episodes=args.episodes))
    if args.gif:
        print(f"{render_gif(args.policy, args.gif)} frames -> {args.gif}")


def _pick_gif(agent, path, episodes=3, seed=777):
    from PIL import Image

    from .manip import PickEnv

    import mujoco

    env = PickEnv(seed=seed)
    cam = mujoco.MjvCamera()  # a three-quarter view: arm, cube and goal all in frame
    cam.lookat[:] = [0.3, 0.0, 0.1]
    cam.distance, cam.elevation, cam.azimuth = 1.25, -28, 145
    frames = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        agent.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step(agent.act(obs))
            if env.steps % 3 == 0:
                frames.append(Image.fromarray(env.render(cam, 360, 270)))
    env.close()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)
    print(f"{len(frames)} frames -> {path}")


def cmd_teleop(args):
    from . import teleop
    from .manip import PickEnv

    env = PickEnv(camera=args.camera)
    if args.input == "phone":
        dev = teleop.PhoneInput(port=args.port)
        print(f"open this on your phone (same Wi-Fi):\n\n    {dev.url}\n")
        print("If it doesn't load, allow Python through the Windows firewall for private networks.")
    else:
        dev = teleop.KeyboardInput()
        print(teleop.__doc__.split("Phone:")[0])
    try:
        n = teleop.run(env, dev, record=args.record, window=not args.no_window)
    except KeyboardInterrupt:
        n = None
    finally:
        if hasattr(dev, "close"):
            dev.close()
        env.close()
    if args.record:
        print(f"demos saved to {args.record}" + (f" ({n} this session)" if n is not None else ""))


def cmd_demos(args):
    from .manip import DemoWriter, PickEnv, ScriptedExpert, collect

    env = PickEnv(seed=args.seed, camera=args.camera)
    w = DemoWriter(args.out, env.meta, append=not args.overwrite)
    t0 = time.perf_counter()
    res = collect(env, ScriptedExpert(env), w, args.episodes, seed=args.seed, noise=args.noise, log=print)
    print(f"{res}; {w.count} demos in {args.out} ({time.perf_counter() - t0:.0f} s)")
    w.close()
    env.close()


def cmd_bc(args):
    from .bc import evaluate_bc, train_bc

    out = args.out or str(Path(args.demos).with_suffix("")) + "_bc.pt"
    res = train_bc(args.demos, out, chunk=args.chunk, epochs=args.epochs, max_demos=args.max_demos,
                   device="cpu" if args.cpu_only else None)
    print(f"trained on {res['demos']} demos ({res['samples']} samples) in {res['seconds']:.0f} s -> {out}")
    print(f"evaluation, {args.episodes} fresh episodes: {evaluate_bc(out, episodes=args.episodes)}")


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
            p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                           help="gpu = MuJoCo Warp on the graphics card (needs mujoco-warp)")
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
                               ("gen", cmd_gen, "generate a synthetic dataset"),
                               ("ros", cmd_ros, "bridge a scene to ROS 2")):
        p = add(name, fn, help=helptext)
        p.add_argument("scene", help="scene YAML file")
        p.add_argument("--quality", default="auto", choices=["auto", "potato", "laptop", "desktop"])
        if name == "view":
            p.add_argument("--out", default="captures")
        if name == "ros":
            p.add_argument("--transport", default="auto", choices=["auto", "rclpy", "rosbridge", "memory"])
            p.add_argument("--url", default="ws://localhost:9090", help="rosbridge_server address")
            p.add_argument("--seconds", type=float, help="stop after this much sim time (default: run forever)")
            p.add_argument("--scan-hz", type=float, default=10.0)
            p.add_argument("--camera-hz", type=float, default=5.0)
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
            p.add_argument("--photoreal", action="store_true", help="path-trace RGB with Mitsuba 3 (slow, pretty)")
            p.add_argument("--spp", type=int, default=64, help="samples per pixel for --photoreal")

    p = add("train", cmd_train, help="train a policy with PPO on a task file")
    p.add_argument("task", help="task YAML (examples/tasks/*.yaml)")
    p.add_argument("--iters", type=int, help="PPO iterations (default: from the task file)")
    p.add_argument("--envs", type=int, help="parallel envs (default: from the task file)")
    p.add_argument("--out", help="run folder (default: runs/<task>-<time>)")
    p.add_argument("--seed", type=int, default=0)
    p = add("play", cmd_play, help="evaluate a trained policy, optionally render a GIF")
    p.add_argument("policy", help="policy.pt from spud train, or a BC model from spud bc")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--gif")
    p = add("teleop", cmd_teleop, help="drive the arm yourself and record demos")
    p.add_argument("--input", default="keyboard", choices=["keyboard", "phone"])
    p.add_argument("--record", help="HDF5 file to append demos to")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--camera", choices=["front", "top"], help="also record this camera's RGB in the demos")
    p.add_argument("--no-window", action="store_true", help="phone only: no MuJoCo window on the laptop")
    p = add("demos", cmd_demos, help="let the scripted expert record demos")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default="demos.hdf5")
    p.add_argument("--noise", type=float, default=0.1, help="DART: noise on executed actions, clean labels")
    p.add_argument("--camera", choices=["front", "top"], help="also record this camera's RGB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p = add("bc", cmd_bc, help="behavior cloning on recorded demos")
    p.add_argument("demos", help="HDF5 from spud teleop / spud demos")
    p.add_argument("--out")
    p.add_argument("--chunk", type=int, default=10, help="actions predicted per step (action chunking)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-demos", type=int)
    p.add_argument("--episodes", type=int, default=100, help="evaluation episodes")

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
