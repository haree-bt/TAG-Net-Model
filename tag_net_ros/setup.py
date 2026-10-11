import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tag_net_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hareh',
    maintainer_email='hareh@todo.todo',
    description='Task-Aware Robotic Grasping (TAG-Net) in ROS 2 Jazzy & Gazebo Harmonic',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tag_net_node = tag_net_ros.tag_net_node:main',
            'grasp_commander = tag_net_ros.grasp_commander:main',
        ],
    },
)
