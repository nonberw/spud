"""Noise models, GPU physics, photoreal export and the ROS 2 bridge."""

import numpy as np
import pytest

from spud import noise as nz
from spud.scene import build_scene


# ------------------------------------------------------------------- noise --

def test_imu_noise_matches_its_density():
    rng = np.random.default_rng(0)
    m = nz.ImuNoise(gyro_bias_walk=0.0, accel_bias_walk=0.0)
    out = np.array([m.apply(rng, np.zeros(3), np.zeros(3))[0] for _ in range(5000)])
    assert out.std() == pytest.approx(m.gyro_noise / np.sqrt(m.dt), rel=0.05)


def test_imu_bias_wanders():
    rng = np.random.default_rng(1)
    m = nz.ImuNoise(gyro_noise=0.0, accel_noise=0.0)
    for _ in range(2000):
        m.apply(rng, np.zeros(3), np.zeros(3))
    assert np.abs(m.gyro_bias).max() > 0  # a random walk, not a constant


def test_depth_noise_follows_kinect_model_and_marks_invalid():
    rng = np.random.default_rng(0)
    m = nz.DepthNoise(edge_holes=0.0, quantization=0.0)
    z = np.full((200, 200), 2.0, np.float32)
    z[:, :5] = 20.0  # beyond max_depth
    out = m.apply(rng, z)
    assert (out[:, :5] == 0).all()
    assert (out[:, 10:] - 2.0).std() == pytest.approx(m.a + m.b * (2.0 - m.z0) ** 2, rel=0.08)


def test_lidar_noise_drops_grazing_rays_more():
    rng = np.random.default_rng(0)
    m = nz.LidarNoise(dropout=0.0, grazing_dropout=1.0)
    n = 20000
    dirs = np.tile([1.0, 0, 0], (n, 1))
    head_on = m.apply(rng, np.full(n, 3.0), np.tile([-1.0, 0, 0], (n, 1)), dirs)[0]
    grazing = m.apply(rng, np.full(n, 3.0), np.tile([0.0, 1, 0], (n, 1)), dirs)[0]
    assert np.isfinite(head_on).mean() > 0.99 and np.isfinite(grazing).mean() < 0.01


def test_noise_config_is_validated():
    with pytest.raises(ValueError, match="unknown"):
        nz.make(nz.LidarNoise, {"range_sd": 0.1})
    assert nz.make(nz.LidarNoise, False) is None


def test_noisy_scene_sensors_keep_clean_copies():
    sc = build_scene({"robots": [{"type": "zoomie"}],
                      "sensors": [{"type": "lidar", "robot": "zoomie0", "rays": 90, "noise": True},
                                  {"type": "imu", "robot": "zoomie0", "noise": True},
                                  {"type": "camera", "pos": [0, -2, 1], "width": 64, "height": 48, "noise": {"rgb": True}}]},
                     quality="potato")
    lid, imu, cam = (s.read() for s in sc.sensors)
    assert {"distance", "distance_clean", "intensity"} <= set(lid)
    assert {"gyro_clean", "accel_clean"} <= set(imu)
    assert "rgb_clean" in cam and "depth_clean" not in cam  # only RGB noise was asked for


# --------------------------------------------------------------------- GPU --

def test_gpu_dogs_walk_like_cpu_dogs():
    from spud.gpu import GpuZoomies, available

    ok, why = available()
    if not ok:
        pytest.skip(why)
    import torch

    from spud.cli import POLICY

    g = GpuZoomies(256, str(POLICY))
    g.commands[:, 0] = 0.6
    g.step()
    x0 = g.qpos[:, 0].clone()
    for _ in range(100):
        g.step()
    torch.cuda.synchronize()
    assert g.upright() > 0.95
    assert (g.qpos[:, 0] - x0).mean().item() / 2.0 > 0.3


# ---------------------------------------------------------------- photoreal --

def test_photoreal_camera_matches_mujoco_camera():
    pytest.importorskip("mitsuba")
    import mujoco

    from spud import photoreal

    sc = build_scene({"objects": [{"type": "box", "size": [0.05, 0.05, 0.05], "pos": [0, 0, 0.05], "static": True,
                                   "random_yaw": False, "rgba": [1, 0.2, 0.2, 1]}],
                      "sensors": [{"type": "camera", "pos": [0, -1, 0.6], "lookat": [0, 0, 0.05], "width": 64, "height": 48}]},
                     quality="potato")
    mujoco.mj_forward(sc.model, sc.data)
    cam = sc.sensors[0]
    mask = cam.read()["instance"] == 0
    img = photoreal.render(sc.model, sc.data, cam.cam, 64, 48, spp=16).astype(int)
    redness = img[..., 0] - img[..., 1:].mean(-1)
    # The red cube's pixels in the path-traced frame are where MuJoCo's mask says the cube is.
    assert redness[mask].mean() > redness[~mask].mean() + 40


# ---------------------------------------------------------------------- ROS --

def test_ros_bridge_topics_rates_and_cmd_vel():
    from spud.ros import Bridge, MemoryTransport

    sc = build_scene({"robots": [{"type": "zoomie", "count": 2}],
                      "sensors": [{"type": "lidar", "name": "scan", "robot": "zoomie0", "rays": 120},
                                  {"type": "imu", "name": "imu", "robot": "zoomie0"},
                                  {"type": "camera", "name": "cam", "pos": [0, -3, 2], "width": 32, "height": 24}]},
                     quality="potato")
    tx = MemoryTransport()
    br = Bridge(sc, tx, scan_hz=10, camera_hz=5)
    br.run(seconds=1.0, realtime=False)
    topics = [t for t, _, _ in tx.published]
    assert topics.count("/clock") == 50 and topics.count("/scan/scan") == 10 and topics.count("/cam/image_raw") == 5
    img = next(m for t, _, m in tx.published if t == "/cam/image_raw")
    assert img["encoding"] == "rgb8" and img["step"] == 32 * 3 and len(img["data"]) == 24 * 32 * 3
    scan = next(m for t, _, m in tx.published if t == "/scan/scan")
    assert len(scan["ranges"]) == 120 and scan["header"]["frame_id"] == "zoomie0/scan"
    js = next(m for t, _, m in tx.published if t == "/zoomie1/joint_states")
    assert len(js["name"]) == 12 and "FL_knee" in js["name"]
    tf = next(m for t, _, m in tx.published if t == "/tf")
    assert {x["child_frame_id"] for x in tf["transforms"]} == {"zoomie0/base_link", "zoomie1/base_link"}
    tx.handlers["/zoomie1/cmd_vel"]({"linear": {"x": 5.0, "y": 0.0}, "angular": {"z": -0.5}})
    np.testing.assert_allclose(sc.groups[0].command[1], [1.2, 0.0, -0.5])  # clipped to what the policy learned
