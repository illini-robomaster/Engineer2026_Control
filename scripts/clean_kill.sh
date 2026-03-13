#!/usr/bin/bash
printenv | egrep '^(ROS|RMW|AMENT|COLCON)'

# 3) stop obvious leftovers
sudo pkill -f 'ros2|rviz2|gz|gazebo|ign|joint_state_publisher|robot_state_publisher|controller_manager'

# 4) reset CLI discovery cache
ros2 daemon stop
ros2 daemon start
