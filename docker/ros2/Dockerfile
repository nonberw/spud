# ROS 2 Jazzy + rosbridge, for driving SPUD scenes from ROS on Windows/macOS (or anywhere).
#   docker build -t spud-ros2 docker/ros2
#   docker run --rm -p 9090:9090 spud-ros2
# then: spud ros examples/yard.yaml --transport rosbridge --url ws://localhost:9090
FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*
EXPOSE 9090
CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml"]
