import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from spud.hw import Hardware
from spud.quality import PRESETS, auto_preset
from spud.scene import build_scene

DATA = Path(__file__).parent / "data"


def _hw(**kw):
    base = dict(cpu_threads=20, ram_total=16, ram_free=8, gpu="x", vram_total=4, vram_free=3, h2d_gbps=12, os="t")
    base.update(kw)
    return Hardware(**base)


@pytest.mark.parametrize("terrain", [{"type": "flat"}, {"type": "rough", "height": 0.05, "resolution": 32},
                                     {"type": "stairs", "steps": 4}])
def test_terrains_build(terrain):
    sc = build_scene({"terrain": terrain, "robots": [{"type": "zoomie", "count": 2}]}, quality="potato")
    assert sum(len(g.names) for g in sc.groups) == 2
    for _ in range(10):
        sc.step()
    assert np.isfinite(sc.data.qpos).all()


def test_robot_names_counts_and_objects():
    sc = build_scene({"robots": [{"type": "zoomie", "count": 3}, {"type": "arm", "count": 2, "layout": "line", "spacing": 1.5}],
                      "objects": [{"type": "box", "count": 4, "region": [[-1, -1], [1, 1]], "class": "crate"}]},
                     quality="potato")
    names = [n for g in sc.groups for n in g.names]
    assert names == ["zoomie0", "zoomie1", "zoomie2", "arm0", "arm1"]
    assert [o.name for o in sc.objects] == ["crate0", "crate1", "crate2", "crate3"]
    assert sc.model.ngeom == len({sc.model.geom(i).name for i in range(sc.model.ngeom) if sc.model.geom(i).name}) + \
        sum(1 for i in range(sc.model.ngeom) if not sc.model.geom(i).name)
    planes = [i for i in range(sc.model.ngeom) if sc.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert len(planes) == 1  # robot files' own floors are stripped


def test_dogs_walk_in_a_scene():
    sc = build_scene({"robots": [{"type": "zoomie", "count": 4, "command": [0.6, 0, 0]}]}, quality="potato")
    x0 = sc.data.qpos[sc.groups[0].qpos_idx[:, 0]].copy()
    for _ in range(100):
        sc.step()
    assert sc.upright()["zoomie"] == 1.0
    assert (sc.data.qpos[sc.groups[0].qpos_idx[:, 0]] - x0).mean() / 2.0 > 0.3


def test_arm_holds_home_pose():
    from armageddon import HOME

    sc = build_scene({"robots": [{"type": "arm"}]}, quality="potato")
    for _ in range(50):
        sc.step()
    g = sc.groups[0]
    q = sc.data.qpos[[sc.model.jnt_qposadr[sc.model.actuator_trnid[a][0]] for a in g.ctrl_idx[0][:6]]]
    np.testing.assert_allclose(q, HOME, atol=0.03)


def test_urdf_robot_loads_and_holds():
    sc = build_scene({"robots": [{"type": "urdf", "file": str(DATA / "two_link.urdf"), "pos": [0, 0, 0.05]}]},
                     quality="potato")
    assert any(sc.model.body(b).name == "urdf0_arm" for b in range(sc.model.nbody))
    for _ in range(20):
        sc.step()
    assert np.isfinite(sc.data.qpos).all()


def test_lidar_measures_a_wall():
    sc = build_scene({"terrain": {"type": "flat"}, "robots": [{"type": "zoomie", "pos": [0, 0, 0]}],
                      "objects": [{"type": "box", "size": [0.05, 2, 1], "pos": [2.0, 0, 1.0], "static": True,
                                   "random_yaw": False, "class": "wall"}],
                      "sensors": [{"type": "lidar", "robot": "zoomie0", "rays": 360, "range": 10}]}, quality="potato")
    r = sc.sensors[0].read()
    ahead = r["distance"][0, 180]  # ray 180 of 360 points along +x (azimuth 0)
    torso_x = sc.data.xpos[sc.robot_body("zoomie0")][0]
    assert ahead == pytest.approx(2.0 - 0.05 - torso_x, abs=0.01)


def test_imu_at_rest_reads_gravity():
    sc = build_scene({"robots": [{"type": "zoomie", "command": [0, 0, 0]}], "sensors": [{"type": "imu", "robot": "zoomie0"}]},
                     quality="potato")
    for _ in range(150):
        sc.step()
    r = sc.sensors[0].read()
    assert np.linalg.norm(r["accel"]) == pytest.approx(9.81, abs=0.15)
    assert np.abs(r["gyro"]).max() < 0.1


def test_segmentation_masks_are_tight():
    sc = build_scene({"objects": [{"type": "box", "size": [0.03, 0.03, 0.03], "count": 6, "region": [[-0.3, -0.3], [0.3, 0.3]]}],
                      "sensors": [{"type": "camera", "pos": [0, -1, 1.2], "lookat": [0, 0, 0], "width": 160, "height": 120}]},
                     quality="potato")
    for _ in range(60):
        mujoco.mj_step(sc.model, sc.data)
    inst = sc.sensors[0].read()["instance"]
    seen = [k for k in np.unique(inst) if k >= 0]
    assert len(seen) >= 4
    for k in seen:  # no anti-aliasing ghosts: each instance is one compact blob
        ys, xs = np.nonzero(inst == k)
        assert xs.max() - xs.min() < 30 and ys.max() - ys.min() < 30


def test_replicator_writes_valid_coco(tmp_path):
    from spud.replicator import generate

    sc = build_scene(Path(__file__).parents[1] / "examples" / "blocks.yaml", quality="potato")
    coco = generate(sc, tmp_path, frames=3, settle=20)
    saved = json.loads((tmp_path / "annotations.json").read_text())
    assert saved["images"] and saved["annotations"] and {c["name"] for c in saved["categories"]} == {"cube", "ball", "can"}
    for a in saved["annotations"]:
        x, y, w, h = a["bbox"]
        img = saved["images"][a["image_id"] - 1]
        assert w > 0 and h > 0 and x + w <= img["width"] and y + h <= img["height"] and a["area"] <= w * h
    assert len(list((tmp_path / "images").glob("*.png"))) == 3
    assert coco["images"][0]["width"] == 320  # as the YAML asks; potato only caps the maximum (640x480)


def test_quality_presets():
    assert auto_preset(_hw(gpu=None)) == "potato"
    assert auto_preset(_hw(cpu_threads=4)) == "potato"
    assert auto_preset(_hw()) == "laptop"
    assert auto_preset(_hw(vram_total=12, cpu_threads=24)) == "desktop"
    assert not PRESETS["potato"].shadows and PRESETS["desktop"].shadows


def test_dogs_walk_off_the_rough_patch_and_keep_walking():
    # Regression: found by the live ROS 2 CI run, where dogs walked off the heightfield and fell out of the world.
    sc = build_scene({"terrain": {"type": "rough", "size": [3, 3], "height": 0.04, "resolution": 32},
                      "robots": [{"type": "zoomie", "count": 2, "spacing": 1.0, "command": [0.8, 0, 0]}]}, quality="potato")
    for _ in range(300):  # 6 s at ~0.7 m/s: far past the 1.5 m edge
        sc.step()
    assert sc.data.qpos[sc.groups[0].qpos_idx[:, 0]].min() > 2.0
    assert sc.upright()["zoomie"] == 1.0
