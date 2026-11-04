"""ROS 2 bridge: a SPUD scene shows up in ROS like a real robot does.

Published (per robot `<r>`, e.g. `zoomie0`):
    /clock                      rosgraph_msgs/Clock        sim time (use_sim_time: true)
    /tf                         tf2_msgs/TFMessage         odom -> <r>/base_link for every robot
    /<r>/odom                   nav_msgs/Odometry
    /<r>/joint_states           sensor_msgs/JointState
    /<sensor>/scan              sensor_msgs/LaserScan      lidar (first channel)
    /<sensor>/imu               sensor_msgs/Imu
    /<sensor>/image_raw         sensor_msgs/Image (rgb8);  /<sensor>/depth  (32FC1, meters)
Subscribed:
    /<r>/cmd_vel                geometry_msgs/Twist        drives the trained dogs (linear.x/y, angular.z)

Two transports, same messages:
  * `rclpy`: a native ROS 2 node, if ROS 2 is installed and sourced in this Python;
  * `rosbridge`: JSON over WebSocket to a `rosbridge_server` (ROS 2 in Docker/WSL/another machine).
    The realistic route on Windows: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`.
Messages are built once as plain dicts with ROS 2 field names; each transport converts them.
"""

from __future__ import annotations

import base64
import time

import numpy as np

from .sensors import IMU, Camera, Lidar


# ----------------------------------------------------------------- messages --

def stamp(t: float) -> dict:
    sec = int(t)
    return {"sec": sec, "nanosec": int(round((t - sec) * 1e9))}


def header(t: float, frame: str) -> dict:
    return {"stamp": stamp(t), "frame_id": frame}


def _q(wxyz) -> dict:
    w, x, y, z = (float(v) for v in wxyz)
    return {"x": x, "y": y, "z": z, "w": w}


def _v(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def laser_scan(t, frame, lidar: Lidar, reading: dict) -> dict:
    ranges = reading["distance"][0]
    n = len(ranges)
    inc = 2 * np.pi / n
    msg = {"header": header(t, frame), "angle_min": -np.pi, "angle_max": float(-np.pi + inc * (n - 1)),
           "angle_increment": float(inc), "time_increment": 0.0, "scan_time": 0.1,
           "range_min": 0.05, "range_max": float(lidar.range),
           "ranges": [float(r) for r in ranges],  # +inf = no return, as REP 117 asks
           "intensities": [float(v) for v in reading["intensity"][0]] if "intensity" in reading else []}
    return msg


def imu_msg(t, frame, reading: dict, noise=None) -> dict:
    def cov(var):
        return [float(var), 0.0, 0.0, 0.0, float(var), 0.0, 0.0, 0.0, float(var)]

    g_var = (noise.gyro_noise**2 / noise.dt) if noise is not None else 0.0
    a_var = (noise.accel_noise**2 / noise.dt) if noise is not None else 0.0
    return {"header": header(t, frame), "orientation": _q(reading["quat"]), "orientation_covariance": cov(0.0),
            "angular_velocity": _v(reading["gyro"]), "angular_velocity_covariance": cov(g_var),
            "linear_acceleration": _v(reading["accel"]), "linear_acceleration_covariance": cov(a_var)}


def image_msg(t, frame, img: np.ndarray, encoding: str) -> dict:
    h, w = img.shape[:2]
    if encoding == "rgb8":
        raw, step = np.ascontiguousarray(img, np.uint8).tobytes(), w * 3
    else:  # 32FC1 depth in meters
        raw, step = np.ascontiguousarray(img, np.float32).tobytes(), w * 4
    return {"header": header(t, frame), "height": h, "width": w, "encoding": encoding, "is_bigendian": 0,
            "step": step, "data": raw}


def joint_state(t, model, data, robot: str) -> dict:
    names, pos, vel = [], [], []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.startswith(robot + "_") and model.jnt_type[j] in (2, 3):  # slide, hinge
            names.append(name[len(robot) + 1:])
            pos.append(float(data.qpos[model.jnt_qposadr[j]]))
            vel.append(float(data.qvel[model.jnt_dofadr[j]]))
    return {"header": header(t, ""), "name": names, "position": pos, "velocity": vel, "effort": []}


def base_pose(model, data, body: int):
    return data.xpos[body].copy(), data.xquat[body].copy()


def odometry(t, robot, model, data, body) -> dict:
    pos, quat = base_pose(model, data, body)
    import mujoco

    vel = np.zeros(6)
    # World frame, rotated into the body frame by hand (REP 105: odom twist is in child_frame_id).
    # MuJoCo's flg_local=1 result did not match that frame: forward speed came out on z.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
    R = data.xmat[body].reshape(3, 3)
    return {"header": header(t, "odom"), "child_frame_id": f"{robot}/base_link",
            "pose": {"pose": {"position": _v(pos), "orientation": _q(quat)}, "covariance": [0.0] * 36},
            "twist": {"twist": {"linear": _v(R.T @ vel[3:]), "angular": _v(R.T @ vel[:3])}, "covariance": [0.0] * 36}}


def tf_msg(t, items) -> dict:
    return {"transforms": [{"header": header(t, "odom"), "child_frame_id": child,
                            "transform": {"translation": _v(p), "rotation": _q(q)}} for child, p, q in items]}


# --------------------------------------------------------------- transports --

class MemoryTransport:
    """Keeps every message; used by tests and `--dry-run`."""

    def __init__(self):
        self.published, self.handlers = [], {}

    def advertise(self, topic, msg_type):
        pass

    def publish(self, topic, msg_type, msg):
        self.published.append((topic, msg_type, msg))

    def subscribe(self, topic, msg_type, callback):
        self.handlers[topic] = callback

    def close(self):
        pass


class RosbridgeTransport:
    def __init__(self, url: str):
        import roslibpy

        host, port = url.replace("ws://", "").split(":")
        self.ros = roslibpy.Ros(host=host, port=int(port))
        self.ros.run(timeout=10)
        self._roslibpy, self.topics, self._subs = roslibpy, {}, []

    def advertise(self, topic, msg_type):
        t = self._roslibpy.Topic(self.ros, topic, msg_type)
        t.advertise()
        self.topics[topic] = t

    def publish(self, topic, msg_type, msg):
        if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):  # rosbridge wants base64 for uint8[]
            msg = dict(msg, data=base64.b64encode(msg["data"]).decode())
        self.topics[topic].publish(self._roslibpy.Message(msg))

    def subscribe(self, topic, msg_type, callback):
        t = self._roslibpy.Topic(self.ros, topic, msg_type)
        t.subscribe(callback)
        self._subs.append(t)

    def close(self):
        for t in self.topics.values():
            t.unadvertise()
        for t in self._subs:
            t.unsubscribe()
        self.ros.terminate()


class RclpyTransport:
    def __init__(self, node_name: str = "spud"):
        import rclpy
        from rosidl_runtime_py.set_message import set_message_fields
        from rosidl_runtime_py.utilities import get_message

        rclpy.init()
        self.rclpy, self.node = rclpy, rclpy.create_node(node_name)
        self._get, self._set, self.pubs = get_message, set_message_fields, {}

    def _cls(self, msg_type):
        return self._get(msg_type.replace("/", "/msg/") if "/msg/" not in msg_type else msg_type)

    def advertise(self, topic, msg_type):
        self.pubs[topic] = self.node.create_publisher(self._cls(msg_type), topic, 10)

    def publish(self, topic, msg_type, msg):
        m = self._cls(msg_type)()
        if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):
            msg = dict(msg, data=list(msg["data"]))
        self._set(m, msg)
        self.pubs[topic].publish(m)

    def subscribe(self, topic, msg_type, callback):
        from rosidl_runtime_py.convert import message_to_ordereddict

        self.node.create_subscription(self._cls(msg_type), topic, lambda m: callback(message_to_ordereddict(m)), 10)

    def spin_once(self):
        self.rclpy.spin_once(self.node, timeout_sec=0)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def make_transport(kind: str = "auto", url: str = "ws://localhost:9090"):
    if kind in ("auto", "rclpy"):
        try:
            return RclpyTransport()
        except ImportError:
            if kind == "rclpy":
                raise
    if kind in ("auto", "rosbridge"):
        return RosbridgeTransport(url)
    if kind == "memory":
        return MemoryTransport()
    raise ValueError(f"unknown transport {kind!r}")


# ------------------------------------------------------------------- bridge --

class Bridge:
    def __init__(self, scene, transport, scan_hz=10.0, camera_hz=5.0, state_hz=50.0):
        self.scene, self.tx = scene, transport
        self.periods = {"scan": 1 / scan_hz, "camera": 1 / camera_hz, "state": 1 / state_hz}
        self.last = dict.fromkeys(self.periods, -1e9)
        self.robots = [(name, g) for g in scene.groups for name in g.names]
        self.bodies = {name: scene.robot_body(name) for name, _ in self.robots}
        tx = transport
        tx.advertise("/clock", "rosgraph_msgs/Clock")
        tx.advertise("/tf", "tf2_msgs/TFMessage")
        for name, g in self.robots:
            tx.advertise(f"/{name}/odom", "nav_msgs/Odometry")
            tx.advertise(f"/{name}/joint_states", "sensor_msgs/JointState")
            if g.controller == "policy":
                row = g.names.index(name)
                tx.subscribe(f"/{name}/cmd_vel", "geometry_msgs/Twist", self._cmd_handler(g, row))
        for s in scene.sensors:
            if isinstance(s, Lidar):
                tx.advertise(f"/{s.name}/scan", "sensor_msgs/LaserScan")
            elif isinstance(s, IMU):
                tx.advertise(f"/{s.name}/imu", "sensor_msgs/Imu")
            elif isinstance(s, Camera):
                tx.advertise(f"/{s.name}/image_raw", "sensor_msgs/Image")
                tx.advertise(f"/{s.name}/depth", "sensor_msgs/Image")

    @staticmethod
    def _cmd_handler(group, row):
        seen = {"first": True}

        def on_twist(msg):
            if seen["first"]:
                seen["first"] = False
                print(f"cmd_vel received for {group.names[row]}: {msg}", flush=True)
            lin, ang = msg.get("linear", {}), msg.get("angular", {})
            group.command[row] = (np.clip(lin.get("x", 0.0), -0.6, 1.2), np.clip(lin.get("y", 0.0), -0.5, 0.5),
                                  np.clip(ang.get("z", 0.0), -1.0, 1.0))
        return on_twist

    def _frame_of(self, sensor):
        robot = getattr(sensor, "robot", None) or next((n for n, b in self.bodies.items() if b == getattr(sensor, "body", -1)), None)
        return f"{robot}/{sensor.name}" if robot else sensor.name

    def publish(self):
        sc, tx = self.scene, self.tx
        t = float(sc.data.time)
        tx.publish("/clock", "rosgraph_msgs/Clock", {"clock": stamp(t)})
        if t - self.last["state"] >= self.periods["state"] - 1e-9:
            self.last["state"] = t
            tx.publish("/tf", "tf2_msgs/TFMessage",
                       tf_msg(t, [(f"{n}/base_link", *base_pose(sc.model, sc.data, b)) for n, b in self.bodies.items()]))
            for name, _ in self.robots:
                tx.publish(f"/{name}/odom", "nav_msgs/Odometry", odometry(t, name, sc.model, sc.data, self.bodies[name]))
                tx.publish(f"/{name}/joint_states", "sensor_msgs/JointState", joint_state(t, sc.model, sc.data, name))
            for s in sc.sensors:
                if isinstance(s, IMU):
                    tx.publish(f"/{s.name}/imu", "sensor_msgs/Imu", imu_msg(t, self._frame_of(s), s.read(), s.noise))
        if t - self.last["scan"] >= self.periods["scan"] - 1e-9:
            self.last["scan"] = t
            for s in sc.sensors:
                if isinstance(s, Lidar):
                    tx.publish(f"/{s.name}/scan", "sensor_msgs/LaserScan", laser_scan(t, self._frame_of(s), s, s.read()))
        if t - self.last["camera"] >= self.periods["camera"] - 1e-9:
            self.last["camera"] = t
            for s in sc.sensors:
                if isinstance(s, Camera):
                    r = s.read(modes=("rgb", "depth"))  # ROS gets no masks, so don't render them
                    if "rgb" in r:
                        tx.publish(f"/{s.name}/image_raw", "sensor_msgs/Image", image_msg(t, self._frame_of(s), r["rgb"], "rgb8"))
                    if "depth" in r:
                        tx.publish(f"/{s.name}/depth", "sensor_msgs/Image", image_msg(t, self._frame_of(s), r["depth"], "32FC1"))

    def run(self, seconds: float | None = None, realtime: bool = True):
        """Step and publish at 50 Hz. Counts steps instead of comparing float sim time, so 1 s is
        exactly 50 steps (comparing `time < seconds` let rounding sneak in a 51st)."""
        t0, clock = time.perf_counter(), time.perf_counter()
        steps = None if seconds is None else round(seconds / 0.02)
        k = 0
        while steps is None or k < steps:
            k += 1
            self.scene.step()
            self.publish()
            if hasattr(self.tx, "spin_once"):
                self.tx.spin_once()
            if realtime:
                clock += 0.02
                lag = time.perf_counter() - clock
                if lag > 0.1:  # fell behind (startup, a slow render): resync instead of bursting to catch up
                    clock = time.perf_counter()
                time.sleep(max(0.0, clock - time.perf_counter()))
        return time.perf_counter() - t0
