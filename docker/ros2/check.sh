#!/bin/bash
# Runs inside the ROS 2 container while `spud ros examples/yard.yaml --transport rosbridge` publishes.
# Fails (exit 1) unless real ROS 2 sees the right topics, types, rates and data, and cmd_vel moves a dog.
source /opt/ros/jazzy/setup.bash
fail=0
bad() { echo "FAIL: $*"; fail=1; }

echo "== topics"
topics=$(ros2 topic list)
echo "$topics" | sort
for t in /clock /tf /scan/scan /imu/imu /overview/image_raw /overview/depth /zoomie4/odom /zoomie4/joint_states /zoomie4/cmd_vel; do
  echo "$topics" | grep -qx "$t" || bad "missing topic $t"
done

echo "== types"
check_type() { local got; got=$(ros2 topic type "$1"); echo "$1: $got"; [ "$got" = "$2" ] || bad "$1 is $got, expected $2"; }
check_type /scan/scan sensor_msgs/msg/LaserScan
check_type /imu/imu sensor_msgs/msg/Imu
check_type /overview/image_raw sensor_msgs/msg/Image
check_type /zoomie4/odom nav_msgs/msg/Odometry
check_type /tf tf2_msgs/msg/TFMessage
check_type /clock rosgraph_msgs/msg/Clock

echo "== rates (Hz, publisher runs at real time)"
rate() { timeout 8 ros2 topic hz "$1" --window 20 2>/dev/null | grep -m1 "average rate" | awk '{print $3}'; }
check_rate() { local r; r=$(rate "$1"); echo "$1: ${r:-none} (want $2-$3)";
  [ -n "$r" ] && awk -v r="$r" -v lo="$2" -v hi="$3" 'BEGIN{exit !(r>=lo && r<=hi)}' || bad "$1 rate ${r:-none}"; }
check_rate /scan/scan 7 13
check_rate /imu/imu 30 60
check_rate /overview/image_raw 3 7

echo "== data"
ranges=$(timeout 6 ros2 topic echo --once --field ranges /scan/scan 2>/dev/null | head -c 400)
echo "LaserScan ranges: ${ranges:0:120}..."
[ -n "$ranges" ] || bad "no LaserScan received"
az=$(timeout 6 ros2 topic echo --once --field linear_acceleration.z /imu/imu 2>/dev/null | head -1)
echo "Imu linear_acceleration.z: $az (gravity, ~9.8 when upright)"
awk -v a="$az" 'BEGIN{exit !(a>7 && a<13)}' || bad "imu accel.z $az"
img=$(timeout 6 ros2 topic echo --once --no-arr /overview/image_raw 2>/dev/null)
echo "$img" | grep -E "^(height|width|encoding|step):"
echo "$img" | grep -q "encoding: rgb8" || bad "image not rgb8"

echo "== cmd_vel drives zoomie4"
vx() { timeout 6 ros2 topic echo --once --field twist.twist.linear.x /zoomie4/odom 2>/dev/null | head -1; }
v0=$(vx)
timeout 6 ros2 topic pub -r 10 /zoomie4/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.2}}" >/dev/null 2>&1
v1=$(vx)
echo "zoomie4 forward speed: before $v0 m/s (commanded 0.5 in the YAML), after 6 s of cmd_vel x=1.2: $v1 m/s"
awk -v a="$v0" -v b="$v1" 'BEGIN{exit !(b > a + 0.2)}' || bad "cmd_vel did not speed the dog up"

[ $fail = 0 ] && echo "ALL ROS 2 CHECKS PASSED" || echo "SOME ROS 2 CHECKS FAILED"
exit $fail
