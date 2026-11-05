"""Robot learning: training tasks, PPO, the pick task, demos, behavior cloning, teleop."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from spud.tasks import ReachEnv, load_task

ROOT = Path(__file__).parents[1]


# ------------------------------------------------------------------ tasks --

def test_reach_env_holds_still_without_actions():
    env = ReachEnv({"task": "reach", "robot": {"type": "arm", "gripper": "fixed"}}, num_envs=64, nthread=4)
    env.steps[:] = 0  # no resets during the check
    ee0 = env.ee.copy()
    for _ in range(50):
        env.step(np.zeros((env.n, env.act_dim)))
    assert np.linalg.norm(env.ee - ee0, axis=1).max() < 0.005  # gravity compensated, servos hold
    env.close()


def test_urdf_robot_without_motors_gets_servos():
    cfg = load_task(ROOT / "examples" / "tasks" / "reach_my_robot.yaml")
    env = ReachEnv(cfg, num_envs=32, nthread=4)
    assert env.nu == 3 and env.obs_dim == 18
    env.steps[:] = 0
    q0 = env.state[:, env.QP + env.qadr].copy()
    for _ in range(25):
        env.step(np.ones((env.n, env.act_dim)))  # push every joint target up
    q1 = env.state[:, env.QP + env.qadr]
    assert (q1 - q0).mean() > 0.2  # the added servos actually move the joints
    env.close()


def test_targets_are_reachable_and_above_the_floor():
    env = ReachEnv({"task": "reach", "robot": {"type": "arm", "gripper": "fixed"}}, num_envs=8, nthread=2)
    assert (env.targets_pool[:, 2] > 0.05).all()
    assert np.linalg.norm(env.targets_pool[:, :2], axis=1).max() < 0.95  # inside the arm's reach
    env.close()


def test_ppo_learns_to_reach(tmp_path):
    from spud.rl import evaluate, train

    task = load_task(ROOT / "examples" / "tasks" / "reach_my_robot.yaml")
    res = train(task, tmp_path, iters=40, num_envs=512, device="cpu", log=None)
    assert (tmp_path / "policy.pt").exists() and (tmp_path / "progress.csv").exists()
    ev = evaluate(tmp_path / "policy.pt", episodes=64)
    assert ev["success"] > 0.5, (res, ev)
    if (tmp_path / "policy.onnx").exists():  # the exported actor gives the same actions as torch
        ort = pytest.importorskip("onnxruntime")
        import torch

        from spud.rl import load_policy

        ac, _ = load_policy(tmp_path / "policy.pt")
        x = np.random.default_rng(0).normal(size=(8, ac.norm.mean.shape[0])).astype(np.float32)
        y = ort.InferenceSession(str(tmp_path / "policy.onnx")).run(None, {"obs": x})[0]
        np.testing.assert_allclose(y, ac.act(torch.as_tensor(x), deterministic=True).numpy(), atol=1e-4)


# ------------------------------------------------------------------- pick --

def test_scripted_expert_places_the_cube():
    from spud.manip import PickEnv, ScriptedExpert, collect

    env = PickEnv(seed=0)
    res = collect(env, ScriptedExpert(env), None, 5, seed=123)
    assert res["success"] == 1.0
    env.close()


def test_success_needs_the_cube_released_on_the_goal():
    from spud.manip import CUBE_HALF, PickEnv

    env = PickEnv(seed=1)
    adr = env.cube_jnt.qposadr[0]
    env.data.qpos[adr:adr + 3] = env.goal[:2].tolist() + [CUBE_HALF]
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    assert env.is_placed()
    env.data.qpos[env.finger_q] = 0.0  # still holding it: not placed
    assert not env.is_placed()
    env.close()


def test_demos_roundtrip_and_bc_learns(tmp_path):
    import h5py

    from spud.bc import evaluate_bc, load_demos, train_bc
    from spud.manip import OBS_KEYS, DemoWriter, PickEnv, ScriptedExpert, collect

    env = PickEnv(seed=0)
    w = DemoWriter(tmp_path / "d.hdf5", env.meta, append=False)
    collect(env, ScriptedExpert(env), w, 40, seed=0, noise=0.1)
    w.close()
    with h5py.File(tmp_path / "d.hdf5") as f:
        g = f["data"]["demo_0"]
        assert set(OBS_KEYS) <= set(g["obs"].keys())
        assert g["actions"].shape[1] == 5 and len(g["states"]) == len(g["actions"])
        assert json.loads(f["data"].attrs["env_args"])["env_name"] == "spud/pick"
    eps, _ = load_demos(tmp_path / "d.hdf5")
    assert len(eps) >= 35
    train_bc(tmp_path / "d.hdf5", tmp_path / "bc.pt", epochs=150, log=None)
    assert evaluate_bc(tmp_path / "bc.pt", episodes=20)["success"] >= 0.6


# ----------------------------------------------------------------- teleop --

def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=2).read())


def test_phone_server_token_commands_and_deadman():
    from spud.teleop import PhoneInput

    dev = PhoneInput(port=0, host="127.0.0.1")
    base = f"http://127.0.0.1:{dev.port}"
    try:
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/", timeout=2)  # no token, no page
        page = urllib.request.urlopen(f"{base}/?t={dev.token}", timeout=2).read().decode()
        assert "spud teleop" in page
        _post(f"{base}/cmd?t={dev.token}", {"x": 0.5, "y": -2.0, "z": 0, "yaw": 0.1, "grip": True, "event": "save"})
        a = dev.command()
        np.testing.assert_allclose(a, [0.5, -1.0, 0.0, 0.1, 1.0])  # clipped to [-1, 1]
        assert dev.pop_event() == "save" and dev.pop_event() is None
        time.sleep(PhoneInput.DEADMAN + 0.1)
        a = dev.command()
        assert np.all(a[:4] == 0) and a[4] == 1.0  # stopped moving, but keeps holding
        dev.push_frame(np.zeros((24, 32, 3), np.uint8))
        jpg = urllib.request.urlopen(f"{base}/frame.jpg?t={dev.token}", timeout=2).read()
        assert jpg[:2] == b"\xff\xd8"
    finally:
        dev.close()


def test_teleop_loop_records_what_the_operator_did(tmp_path):
    import h5py

    from spud import teleop
    from spud.manip import PickEnv, ScriptedExpert

    env = PickEnv(seed=0)
    expert = ScriptedExpert(env)

    class Operator:  # stands in for a human on the phone
        def command(self):
            return expert.act(env.obs())

        def pop_event(self):
            return None

    saved = teleop.run(env, Operator(), record=tmp_path / "t.hdf5", window=False, max_episodes=1,
                       log=lambda *a: None, realtime=False)
    assert saved == 1
    with h5py.File(tmp_path / "t.hdf5") as f:
        g = f["data"]["demo_0"]
        assert bool(g.attrs["success"]) and g.attrs["operator"] == "Operator"
    env.close()


def test_walk_task_wraps_zoomies():
    from spud.tasks import make_env

    env = make_env({"task": "walk", "push": False}, num_envs=16, nthread=4)
    obs = env.observe()
    assert obs.shape == (16, env.obs_dim)
    obs, rew, done, timeout, info = env.step(np.zeros((16, env.act_dim)))
    assert "success" in info and rew.shape == (16,)
    env.close()
