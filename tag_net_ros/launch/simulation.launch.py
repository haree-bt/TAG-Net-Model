import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_tag_net = get_package_share_directory('tag_net_ros')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_sdf = os.path.join(pkg_tag_net, 'worlds', 'manipulation_table.sdf')
    bridge_config = os.path.join(pkg_tag_net, 'config', 'ros_gz_bridge.yaml')

    # 1. Gazebo Sim Launch
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {world_sdf}'
        }.items()
    )

    # 2. ROS-Gazebo Topic Bridge
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config
        }],
        output='screen'
    )

    # 3. Static TF Publisher (World -> Camera Link)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.4', '0.0', '1.05', '0.0', '1.5707963', '0.0', 'world', 'camera_link'],
        output='screen'
    )

    # 4. TAG-Net Perception & Grasp Synthesis Node
    tag_net_node = Node(
        package='tag_net_ros',
        executable='tag_net_node',
        name='tag_net_node',
        output='screen'
    )

    # 5. Autonomous Manipulation Supervisor Node
    commander_node = Node(
        package='tag_net_ros',
        executable='grasp_commander',
        name='grasp_commander',
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        ros_gz_bridge,
        static_tf,
        tag_net_node,
        commander_node
    ])
