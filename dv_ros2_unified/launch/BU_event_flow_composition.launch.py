import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='evflow_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='dv_ros2_unified',
                    plugin='dv_ros2_unified::Capture',
                    name='event_capture',
                    remappings=[('/events', '/events/asdf')],
                    parameters=[{
                        'timeIncrement': 1000,
                        'frames': True,
                        'events': True,
                        'imu': True,
                        'triggers': True,
                        'cameraName': "default_evcam_name",
                        'aedat4FilePath': "",
                        'cameraCalibrationFilePath': "",
                        'cameraFrameName': "camera",
                        'imuFrameName': "imu",
                        'transformImuToCameraFrame': True,
                        'unbiasedImuData': True,
                        'noiseFiltering': False,
                        'noiseBATime': 2000,
                        'waitForSync': False,
                        'globalHold': False,
                        'biasSensitivity': 2}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                # ComposableNode(
                #     package='evflow_node',
                #     plugin='evflow_node::EvFlowNet',
                #     name='event_flow',
                #     remappings=[('/flow_image', '/flow')],
                #     # TODO: configure additional parameters with the re-write
                #     parameters=[{'history': 'keep_last'}],
                #     extra_arguments=[{'use_intra_process_comms': True}])
            ],
            output='both',
        ),
    ])

