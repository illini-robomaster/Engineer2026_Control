from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arm_teleop'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='TODO@email.com',
    description='Teleoperation input nodes: AprilTag cube and keyboard.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'socket_teleop_node = arm_teleop.socket_teleop_node:main',
            'keyboard_teleop_node = arm_teleop.keyboard_teleop_node:main',
            'joint_monitor_node = arm_teleop.joint_monitor_node:main',
            'ik_teleop_node = arm_teleop.ik_teleop_node:main',
            'moveit_teleop_node = arm_teleop.moveit_teleop_node:main',
        ],
    },
)
