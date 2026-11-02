import numpy as np
import pytest
import torch

from spud.hw import Hardware
from spud.mass import MassSim
from spud.models import PRESETS, iter_blocks, make_blocks, params_of
from spud.plan import plan_model
from spud.stream import StreamedBlocks, resident_copy

CUDA = torch.cuda.is_available()
TINY = PRESETS["tiny"]


def _hw(**kw):
    base = dict(cpu_threads=8, ram_total=16, ram_free=8, gpu="test", vram_total=4, vram_free=3.5,
                h2d_gbps=12.0, os="test", gpu_tflops=15.0, cpu_gflops=200.0)
    base.update(kw)
    return Hardware(**base)


# ------------------------------------------------------------- streaming --

@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("kw", [dict(), dict(n_slots=1), dict(n_slots=3), dict(store="disk")])
def test_streamed_is_bit_exact(kw, tmp_path):
    depth, d, h = TINY
    x = torch.randn(32, 4, d, dtype=torch.float16, device="cuda")
    ref = resident_copy(make_blocks(depth, d, h))(x)
    if kw.get("store") == "disk":
        kw = dict(kw, cache_dir=tmp_path)
    assert torch.equal(StreamedBlocks(iter_blocks(depth, d, h), **kw)(x), ref)


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_micro_batch_matches_resident_on_same_chunks():
    depth, d, h = TINY
    res = resident_copy(make_blocks(depth, d, h))
    x = torch.randn(40, 4, d, dtype=torch.float16, device="cuda")
    out = StreamedBlocks(iter_blocks(depth, d, h), micro_batch=9)(x)
    assert torch.equal(out, torch.cat([res(c) for c in x.split(9)]))


@pytest.mark.parametrize("store", ["ram", "disk"])
def test_cpu_streaming_is_bit_exact(store, tmp_path):
    depth, d, h = TINY
    x = torch.randn(4, 3, d)
    ref = resident_copy(make_blocks(depth, d, h, dtype=torch.float32), "cpu")(x)
    st = StreamedBlocks(iter_blocks(depth, d, h, dtype=torch.float32), device="cpu", store=store, cache_dir=tmp_path)
    assert torch.equal(st(x), ref)


def test_mismatched_blocks_are_rejected():
    blocks = make_blocks(2, 64, 4) + make_blocks(1, 128, 4)
    with pytest.raises(ValueError, match="different structure"):
        StreamedBlocks(blocks, device="cpu")


def test_store_size_matches_parameter_count():
    depth, d, h = TINY
    st = StreamedBlocks(iter_blocks(depth, d, h), device="cpu")
    assert st.stats.store_bytes == 2 * params_of(depth, d)  # fp16, no extra copies


# --------------------------------------------------------------- planner --

def test_planner_tiers():
    p = PRESETS["2b"]
    P = params_of(*p[:2])
    small = plan_model(_hw(), params_of(4, 256), 2 * params_of(4, 256), 2 * params_of(4, 256) // 4, 256, 512)
    assert small.tier == "resident"
    assert plan_model(_hw(), P, 2 * P, 2 * P // p[0], p[1], 4096).tier == "ram"
    assert plan_model(_hw(ram_free=3), P, 2 * P, 2 * P // p[0], p[1], 4096).tier == "disk"
    assert plan_model(_hw(gpu=None, vram_total=0, vram_free=0), P, 2 * P, 2 * P // p[0], p[1], 64).tier == "cpu"


def test_planner_picks_micro_batches_when_activations_dont_fit():
    depth, d, _ = PRESETS["2b"]
    P = params_of(depth, d)
    pl = plan_model(_hw(vram_free=1.0, vram_total=1.0), P, 2 * P, 2 * P // depth, d, 16384)
    assert pl.tier in ("ram", "disk") and pl.micro_batch and pl.micro_batch < 16384 and pl.fits
    assert pl.peak_vram_gib < 0.85 * 1.0


def test_planner_falls_back_to_cpu_when_even_the_batch_cannot_fit():
    depth, d, _ = PRESETS["2b"]
    P = params_of(depth, d)
    assert plan_model(_hw(vram_free=0.5, vram_total=0.5), P, 2 * P, 2 * P // depth, d, 65536).tier == "cpu"


def test_planner_step_is_max_of_transfer_and_compute():
    depth, d, _ = PRESETS["1b"]
    P = params_of(depth, d)
    pl = plan_model(_hw(), P, 2 * P, 2 * P // depth, d, 64, force="ram")
    assert pl.bound == "transfer"
    assert pl.step_seconds == pytest.approx(2 * P / (12.0 * 2**30))


# ------------------------------------------------------------ simulation --

def _zoomie_model():
    from zoomies.robot import load_model

    return load_model()


def test_mass_sim_chunking_does_not_change_results():
    m = _zoomie_model()
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0, 0.3, (10, 37, m.nu))
    a = MassSim(m, 37, nthread=4, ram_budget_gib=1.0, substeps=4)
    b = MassSim(m, 37, nthread=4, ram_budget_gib=5e-6, substeps=4)  # forces tiny chunks
    assert b.chunk < 37
    for t in range(10):
        a.step(ctrl[t])
        b.step(ctrl[t])
    np.testing.assert_array_equal(a.state, b.state)
    a.close()
    b.close()


def test_mass_sim_checkpoint_roundtrip(tmp_path):
    m = _zoomie_model()
    sim = MassSim(m, 8, nthread=2, substeps=4)
    ctrl = np.zeros((8, m.nu))
    for _ in range(5):
        sim.step(ctrl)
    sim.save(tmp_path / "c.npz", tag=np.array([7]))
    other = MassSim(m, 8, nthread=2, substeps=4)
    extra = other.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(other.state, sim.state)
    assert other.steps == 5 and extra["tag"][0] == 7
    sim.close()
    other.close()


def test_one_big_world_has_one_floor_and_separate_robots():
    from spud.swarm import Swarm

    sw = Swarm(9, spacing=1.0)
    floors = [i for i in range(sw.model.ngeom) if sw.model.geom_type[i] == 0]  # mjGEOM_PLANE
    assert len(floors) == 1
    xy = sw.positions()
    assert min(np.linalg.norm(xy[i] - xy[j]) for i in range(9) for j in range(i)) > 0.9


def test_trained_policy_walks_in_mass_sim():
    from spud.cli import POLICY
    from spud.zoomie_brain import TrainedBrain, ZoomieMass

    zm = ZoomieMass(64, nthread=4)
    zm.commands[:] = (0.6, 0, 0)
    brain = TrainedBrain(str(POLICY))
    x0 = zm.sim.qpos()[:, 0].copy()
    for _ in range(100):
        zm.step(brain(zm.observe()))
    assert zm.upright().mean() > 0.95
    assert (zm.sim.qpos()[:, 0] - x0).mean() / 2.0 > 0.3  # 100 steps = 2 s
    zm.sim.close()
