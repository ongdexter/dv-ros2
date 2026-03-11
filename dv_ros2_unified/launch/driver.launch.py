import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

driver_args = [        
        DeclareLaunchArgument('camera_calibration_file_path', default_value='/home/dcist/nf_ws/src/dv-ros2/dv_ros2_unified/config/calib_epa.xml'),
        DeclareLaunchArgument('bias_sensitivity', default_value='3'),
        DeclareLaunchArgument('undistort_enable', default_value='true'),
        DeclareLaunchArgument('visualization_enable', default_value='false'),
        DeclareLaunchArgument('recording_enable', default_value='false'),
    ]

def generate_launch_description():
    """Generate launch description with multiple components."""

    ld = LaunchDescription(driver_args)

    ld.add_action(
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
                        'camera_calibration_file_path': LaunchConfiguration('camera_calibration_file_path'),
                        'bias_sensitivity': LaunchConfiguration('bias_sensitivity'),
                        'undistort_events': LaunchConfiguration('undistort_enable')
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    condition=IfCondition(LaunchConfiguration('visualization_enable')),
                    package='dv_ros2_unified',
                    plugin='dv_ros2_unified::Visualizer',
                    name='live_visualizer_component',
                    parameters=[{
                        'image_topic': '/image_event_visualizer',
                        'frame_rate': 30.0,
                        'background_color_r': 0,
                        'background_color_g': 0,
                        'background_color_b': 0,
                        'positive_event_color_r': 255,
                        'positive_event_color_g': 0,
                        'positive_event_color_b': 0,
                        'negative_event_color_r': 0,
                        'negative_event_color_g': 0,
                        'negative_event_color_b': 255,}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    condition=IfCondition(LaunchConfiguration('recording_enable')),
                    package='rosbag2_composable_recorder',
                    plugin='rosbag2_composable_recorder::ComposableRecorder',
                    name="recorder",
                    parameters=[{'topics': [
                            "/events",
                            "/camera_info",
                            "/imu",
                            "/neurofly1/control_odom",
                            "/neurofly1/zed_node/depth/depth_registered",
                            "/neurofly1/zed_node/depth/camera_info",
                            "/neurofly1/zed_node/left/image_rect_color",
                            "/neurofly1/zed_node/left/camera_info",
                            "/neurofly1/zed_node/odom",
                            "/neurofly1/zed_node/pose",
                            "/neurofly1/zed_node/imu",
                            ],
                                'storage_id': 'mcap',
                                'record_all': False,
                                'disable_discovery': False,
                                'serialization_format': 'cdr',
                                'start_recording_immediately': False,
                                "bag_prefix": '/home/dcist/bags/ev_'}],
                    remappings=[],
                    extra_arguments=[{'use_intra_process_comms': True}],
                ),
            ],
            output='both',
        )
    )

    return ld
