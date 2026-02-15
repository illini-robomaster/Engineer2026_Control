# ROS 2 Humble Migration Notes (robotic_arm_v4_urdf)

## What was migrated
- Package build system: `catkin` -> `ament_cmake`
- Launch files: ROS 1 XML -> ROS 2 Python launch
- URDF frame/link names sanitized for ROS tooling compatibility:
  - `J3-J4` -> `J3_J4`
  - `J5-J6` -> `J5_J6`
  - `End Effector` -> `End_Effector`
- Added `ros2_control` interfaces for joints `Joint1..Joint6`
- Added controller config for:
  - `joint_state_broadcaster`
  - `arm_controller` (`joint_trajectory_controller`)

## Launch commands
From a ROS 2 workspace where this package is built and sourced:

```bash
ros2 launch robotic_arm_v4_urdf display.launch.py
```

```bash
ros2 launch robotic_arm_v4_urdf control.launch.py
```

```bash
ros2 launch robotic_arm_v4_urdf moveit_control.launch.py
```
Use `use_moveit_rviz:=false` for headless mode.

```bash
ros2 launch robotic_arm_v4_urdf gazebo.launch.py
```

## Send a trajectory command
```bash
ros2 topic pub /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6'],
  points: [
    {positions: [0.0, 0.3, -0.5, 0.4, 0.2, 0.0], time_from_start: {sec: 3}}
  ]
}"
```

## Multi-DOF control via MoveIt
- `moveit_control.launch.py` starts:
  - `ros2_control_node`
  - `joint_state_broadcaster`
  - `arm_controller` (`FollowJointTrajectory`)
  - `move_group` (OMPL planning pipeline)
  - optional RViz
- In RViz, add the `MotionPlanning` display, set planning group to `arm`, then plan/execute trajectories that move multiple joints together.

## Hardware integration note
`robotic_arm_v4_urdf.urdf` currently uses `mock_components/GenericSystem`.
For real hardware, replace the `<hardware><plugin>...</plugin></hardware>` section in the `ros2_control` block with your hardware interface plugin.
