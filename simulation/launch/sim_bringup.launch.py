"""
PHANTOM-ECHO REVEAL — Simulation Bringup Launch File
sim_bringup.launch.py

Launches:
    1. Gazebo simulator with living_room_occluded.world
    2. TurtleBot3 Phantom URDF spawn
    3. ROS2 Nav2 navigation stack (SLAM + path planning)
    4. PHANTOM-ECHO REVEAL pipeline node
    5. RViz2 visualisation

Usage:
    ros2 launch phantom_echo_reveal sim_bringup.launch.py

Environment:
    export TURTLEBOT3_MODEL=waffle_pi
    source /opt/ros/humble/setup.bash
    source install/setup.bash
"""

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    ExecuteProcess, SetEnvironmentVariable, TimerAction
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share = FindPackageShare("phantom_echo_reveal")

    # ── Launch arguments ──────────────────────────────────────────────
    use_rviz  = LaunchConfiguration("use_rviz",  default="true")
    use_slam  = LaunchConfiguration("use_slam",  default="true")
    world_file = LaunchConfiguration(
        "world",
        default=PathJoinSubstitution([
            pkg_share, "simulation", "worlds", "living_room_occluded.world"
        ])
    )
    robot_x   = LaunchConfiguration("robot_x",   default="2.5")
    robot_y   = LaunchConfiguration("robot_y",   default="0.5")
    robot_yaw = LaunchConfiguration("robot_yaw", default="1.5708")

    declare_rviz  = DeclareLaunchArgument("use_rviz",  default_value="true")
    declare_slam  = DeclareLaunchArgument("use_slam",  default_value="true")
    declare_world = DeclareLaunchArgument("world",     default_value="living_room_occluded.world")
    declare_x     = DeclareLaunchArgument("robot_x",   default_value="2.5")
    declare_y     = DeclareLaunchArgument("robot_y",   default_value="0.5")
    declare_yaw   = DeclareLaunchArgument("robot_yaw", default_value="1.5708")

    # ── Gazebo ────────────────────────────────────────────────────────
    set_gazebo_model_path = SetEnvironmentVariable(
        "GAZEBO_MODEL_PATH",
        PathJoinSubstitution([pkg_share, "simulation", "models"])
    )

    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"
        ]),
        launch_arguments={
            "world": world_file,
            "verbose": "false",
            "pause": "false",
        }.items()
    )

    # ── Robot spawn ───────────────────────────────────────────────────
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "turtlebot3_phantom",
            "-x", robot_x, "-y", robot_y, "-z", "0.01",
            "-Y", robot_yaw,
        ],
        output="screen",
    )

    # ── Robot state publisher ─────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": PathJoinSubstitution([
                pkg_share, "simulation", "urdf", "turtlebot3_phantom.urdf.xacro"
            ]),
            "use_sim_time": True,
        }],
        output="screen",
    )

    # ── SLAM Toolbox ──────────────────────────────────────────────────
    slam_toolbox = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"
        ]),
        launch_arguments={
            "use_sim_time": "true",
            "slam_params_file": PathJoinSubstitution([
                pkg_share, "simulation", "params", "slam_params.yaml"
            ]),
        }.items(),
        condition=IfCondition(use_slam),
    )

    # ── Nav2 ──────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"
        ]),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": PathJoinSubstitution([
                pkg_share, "simulation", "params", "nav2_params.yaml"
            ]),
        }.items(),
    )

    # ── PHANTOM-ECHO REVEAL pipeline node ─────────────────────────────
    # Delayed start — wait for Gazebo + Nav2 to fully initialise
    phantom_echo_node = TimerAction(
        period=5.0,
        actions=[Node(
            package="phantom_echo_reveal",
            executable="phantom_echo_node",
            name="phantom_echo_pipeline",
            output="screen",
            parameters=[{
                "use_sim_time":       True,
                "simulate_audio":     True,
                "simulate_generation":True,
                "output_dir":         "/tmp/phantom_echo_output",
                "voxel_size":         0.05,
                "floor_y":            0.0,
                "ceiling_y":          2.5,
                "lambda_init":        1.0,
                "watchdog_timeout_s": 30.0,
            }],
            remappings=[
                ("/depth/image_raw",     "/camera/depth/image_raw"),
                ("/camera/image_raw",    "/camera/image_raw"),
                ("/imu",                 "/imu"),
                ("/acoustic_ranges",     "/phantom_echo/acoustic_ranges"),
                ("/global_costmap",      "/phantom_echo/global_costmap"),
                ("/manual_fallback",     "/phantom_echo/manual_fallback"),
            ],
        )]
    )

    # ── RViz2 ────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([
            pkg_share, "simulation", "params", "phantom_echo_rviz.rviz"
        ])],
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([
        declare_rviz, declare_slam, declare_world,
        declare_x, declare_y, declare_yaw,
        set_gazebo_model_path,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        slam_toolbox,
        nav2,
        phantom_echo_node,
        rviz_node,
    ])
