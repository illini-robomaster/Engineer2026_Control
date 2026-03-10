from setuptools import find_packages, setup
from glob import glob

package_name = 'arm_hardware'

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
    description='Hardware output: UART bridge to STM32 controller.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'uart_bridge_node = arm_hardware.uart_bridge_node:main',
            'homing_node = arm_hardware.homing_node:main',
        ],
    },
)
