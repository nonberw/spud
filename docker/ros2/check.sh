#!/bin/bash
# Runs inside the ROS 2 container while `spud ros examples/ros_demo.yaml --transport rosbridge` publishes.
# Fails (exit 1) unless real ROS 2 sees the right topics, types, rates and data, and cmd_vel moves the dog.
source /opt/ros/jazzy/setup.bash
fail=0
bad() { echo "FAIL: $*"; fail=1; }
num() { grep -E '^-?[0-9]'; }                                   # ROS 2 may print "A message was lost!!!"
mean() { awk '{s+=$1; n++} END {if (n) printf "%.3f", s/n; else print "nan"}'; }
stream() { timeout "$3" ros2 topic echo --field "$2" "$1" 2>/dev/null | num; }  # topic field seconds

echo "== waiting for discovery"
need="/clock /tf /scan/scan /imu/imu /front/image_raw /front/depth /zoomie0/odom /zoomie0/joint_states /zoomie0/cmd_vel"
for i in $(seq 1 30); do
  topics=$(ros2 topic list 2>/dev/null)
  missing=0; for t in $need; do echo "$topics" | grep -qx "$t" || missing=1; done
  [ $missing = 0 ] && break; sleep 1
done
echo "$topics" | sort
for t in $need; do echo "$topics" | grep -qx "$t" || bad "missing topic $t"; done

echo "== types"
check_type() { local got; got=$(ros2 topic type "$1"); echo "$1: $got"; [ "$got" = "$2" ] || bad "$1 is $got, expected $2"; }
check_type /scan/scan sensor_msgs/msg/LaserScan
check_type /imu/imu sensor_msgs/msg/Imu
check_type /front/image_raw sensor_msgs/msg/Image
check_type /zoomie0/odom nav_msgs/msg/Odometry
check_type /tf tf2_msgs/msg/TFMessage
check_type /clock rosgraph_msgs/msg/Clock
check_type /zoomie0/cmd_vel geometry_msgs/msg/Twist

echo "== rates, relative to the sim clock (/clock ticks at 50 Hz of sim time)"
rate() { timeout 8 ros2 topic hz "$1" --window 30 2>/dev/null | grep -m1 "average rate" | awk '{print $3}'; }
clock=$(rate /clock)
echo "/clock: ${clock:-none} Hz wall -> sim runs at $(awk -v c="$clock" 'BEGIN{printf "%.2f", c/50}')x real time here"
awk -v c="$clock" 'BEGIN{exit !(c > 10)}' || bad "sim too slow or no /clock ($clock)"
ratio() { local r; r=$(rate "$1"); local q; q=$(awk -v r="$r" -v c="$clock" 'BEGIN{printf "%.3f", r/c}');
  echo "$1: ${r:-none} Hz wall, ${q} per clock tick (want $2)";
  awk -v q="$q" -v w="$2" 'BEGIN{exit !(q > 0.7*w && q < 1.3*w)}' || bad "$1 ratio $q, want $2"; }
ratio /scan/scan 0.2
ratio /imu/imu 1.0
ratio /front/image_raw 0.1

echo "== data"
ranges=$(timeout 6 ros2 topic echo --once --field ranges /scan/scan 2>/dev/null | head -c 300)
echo "LaserScan ranges: ${ranges:0:110}..."
echo "$ranges" | grep -qE "[0-9]\.[0-9]" || bad "LaserScan has no finite ranges (the walls should be hit)"
az=$(stream /imu/imu linear_acceleration.z 4 | mean)
echo "Imu linear_acceleration.z, mean over 4 s while standing: $az m/s^2 (gravity: 9.81)"
awk -v a="$az" 'BEGIN{exit !(a > 9.3 && a < 10.3)}' || bad "imu mean accel.z $az"
img=$(timeout 6 ros2 topic echo --once --no-arr /front/image_raw 2>/dev/null)
echo "$img" | grep -E "^(height|width|encoding|step):"
echo "$img" | grep -q "encoding: rgb8" || bad "image not rgb8"

echo "== cmd_vel drives zoomie0"
v0=$(stream /zoomie0/odom twist.twist.linear.x 3 | mean)
timeout 12 ros2 topic pub -r 10 /zoomie0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}" >/dev/null 2>&1 &
sleep 6
v1=$(stream /zoomie0/odom twist.twist.linear.x 4 | mean)
wait
echo "zoomie0 mean forward speed: standing $v0 m/s -> with cmd_vel linear.x=1.0: $v1 m/s"
awk -v a="$v0" 'BEGIN{exit !(a < 0.15 && a > -0.15)}' || bad "dog was not standing still ($v0)"
awk -v b="$v1" 'BEGIN{exit !(b > 0.5)}' || bad "cmd_vel did not make the dog walk ($v1)"

[ $fail = 0 ] && echo "ALL ROS 2 CHECKS PASSED" || echo "SOME ROS 2 CHECKS FAILED"
exit $fail
