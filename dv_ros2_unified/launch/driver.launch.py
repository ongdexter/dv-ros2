import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='dvxplorer_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='dv_ros2_unified',
                    plugin='dv_ros2_unified::Capture',
                    name='live_capture_component',
                    parameters=[{
                        'time_increment': 1000,
                        'frames': True,
                        'events': True,
                        'imu': True,
                        'triggers': True,
                        'bias_sensitivity': 2}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='rosbag2_composable_recorder',
                    plugin='rosbag2_composable_recorder::ComposableRecorder',
                    name="recorder",
                    parameters=[{'topics': [
                            "/events",
                            "/imu",
                            ],
                                'storage_id': 'mcap',
                                'record_all': False,
                                'disable_discovery': False,
                                'serialization_format': 'cdr',
                                'start_recording_immediately': False,
                                "bag_prefix": '/home/nvidia/inivation_ws/ev_'}],
                    remappings=[],
                    extra_arguments=[{'use_intra_process_comms': True}],
                ),
            ],
            output='both',
        )
    ])

