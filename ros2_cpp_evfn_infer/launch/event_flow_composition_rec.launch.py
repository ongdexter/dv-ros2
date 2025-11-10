import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='evflow_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='ros2_cpp_evfn_infer',
                    plugin='ros2_cpp_evfn_infer::InferenceNode',
                    name='live_infnode_component',
                    parameters=[{
                        'input_topic': '/events',
                        'flow_image_topic': '/flow/rgb',
                        'flow_raw_topic': '/flow/raw',
                        'frame_rate': 30.0,
                        'engine_path': '/home/nvidia/nfu_ws/src/trt_engines/evfn_python_converted_fp16.trt',
                        'events_kept': 100_000,}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
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
                        # 'camera_name': "default_evcam_name",
                        # 'aedat4_file_path': "",
                        # 'camera_calibration_file_path': "",
                        # 'camera_frame_name': "camera",
                        # 'imu_frame_name': "imu",
                        # 'transform_imu_to_camera_frame': True,
                        # 'unbiased_imu_data': True,
                        # 'noise_filtering': False,
                        # 'noise_ba_time': 2000,
                        # 'wait_for_sync': False,
                        # 'global_hold': False,
                        'bias_sensitivity': 1}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                # ComposableNode(
                #     package='dv_ros2_unified',
                #     plugin='dv_ros2_unified::Visualizer',
                #     name='live_visualizer_component',
                #     parameters=[{
                #         'image_topic': '/image_event_visualizer',
                #         'frame_rate': 60.0,
                #         'background_color_r': 255,
                #         'background_color_g': 255,
                #         'background_color_b': 255,
                #         'positive_event_color_r': 255,
                #         'positive_event_color_g': 0,
                #         'positive_event_color_b': 0,
                #         'negative_event_color_r': 0,
                #         'negative_event_color_g': 0,
                #         'negative_event_color_b': 255,}],
                #     extra_arguments=[{'use_intra_process_comms': True}]),
				ComposableNode(
                    package='rosbag2_composable_recorder',
                    plugin='rosbag2_composable_recorder::ComposableRecorder',
                    name="recorder",
                    parameters=[{'topics': [
                            "/events",
                            "/imu",
							"/flow_vectors",
                            ],
                                'storage_id': 'mcap',
                                'record_all': False,
                                'disable_discovery': False,
                                'serialization_format': 'cdr',
                                'start_recording_immediately': False,
                                "bag_prefix": '/home/shared_external/rosbags/ev_'}],
                    remappings=[],
                    extra_arguments=[{'use_intra_process_comms': True}],
                ),
            ],
            output='both',
        ),
        #Node(
        #    package='rqt_image_view',
        #    executable='rqt_image_view',
        #    name='rqt_image_view',
        #    output='screen',
        #    emulate_tty=True,
        #)
    ])

