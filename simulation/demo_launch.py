"""
PHANTOM-ECHO REVEAL — Gazebo + ROS2 Nav2 Demo Launch Script

Usage:
    python simulation/demo_launch.py

Prerequisites:
    ROS2 Humble + Gazebo 11 + Nav2 installed
    source /opt/ros/humble/setup.bash

What this launches:
    1. Gazebo with hackathon_room.world
    2. robot_state_publisher with phantom_robot.urdf
    3. Nav2 with nav2_params.yaml
    4. PHANTOM-ECHO REVEAL pipeline (writes costmap.npy)
    5. ROS2 bridge (publishes /phantom_echo/global_costmap)

For simulation without ROS2: run `python -m src.main --mode demo`
and open src/edge/ui/viewer.html
"""

import subprocess, sys, os, time
from pathlib import Path

SIM_DIR = Path(__file__).parent
ROOT    = SIM_DIR.parent


def check_ros2():
    try:
        subprocess.run(["ros2", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    if not check_ros2():
        print("ROS2 not found. Running headless Python simulation instead.")
        print("Open src/edge/ui/viewer.html in Chrome after the pipeline completes.")
        os.chdir(ROOT)
        subprocess.run([sys.executable, "-m", "src.main", "--mode", "demo"])
        return

    print("Launching Gazebo simulation...")
    procs = []

    # 1. Gazebo
    procs.append(subprocess.Popen([
        "gz", "sim", str(SIM_DIR / "hackathon_room.world"), "-v", "4"
    ]))
    time.sleep(3)

    # 2. Robot state publisher
    procs.append(subprocess.Popen([
        "ros2", "run", "robot_state_publisher", "robot_state_publisher",
        "--ros-args", "-p", f"robot_description:={open(SIM_DIR/'phantom_robot.urdf').read()}"
    ]))
    time.sleep(1)

    # 3. Nav2
    procs.append(subprocess.Popen([
        "ros2", "launch", "nav2_bringup", "navigation_launch.py",
        f"params_file:={SIM_DIR / 'nav2_params.yaml'}"
    ]))
    time.sleep(5)

    # 4. PHANTOM pipeline
    os.chdir(ROOT)
    os.environ["PHANTOM_SIMULATE"] = "true"
    procs.append(subprocess.Popen([
        sys.executable, "-m", "src.main", "--mode", "demo"
    ]))

    print("\nAll processes launched. Press Ctrl+C to stop.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        print("Simulation stopped.")


if __name__ == "__main__":
    main()
