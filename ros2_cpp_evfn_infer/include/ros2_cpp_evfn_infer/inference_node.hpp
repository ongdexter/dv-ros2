#pragma once

#include <boost/lockfree/spsc_queue.hpp>
#include <thread>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <tuple>
#include <queue>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/imgproc.hpp>


#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

// dv_ros2_msgs Headers
#include "dv_ros2_msgs/msg/event.hpp"
#include "dv_ros2_msgs/msg/event_array.hpp"
#include "dv_ros2_messaging/messaging.hpp"

#include "trt_engine.hpp"
#include "event_preprocessing.hpp"

namespace ros2_cpp_evfn_infer
{
    struct Params
    {
        /// @brief topic name of the input event array
        std::string input_topic = "events";
        /// @brief topic name of the output inference results
        std::string flow_rgb_topic = "/flow/rgb";
        std::string flow_raw_topic = "/flow/raw";

        /// @brief path to the TensorRT engine file
        std::string engine_path =
            "/home/shared_external/ml_deployments/trt_engines/evfn_python_converted_fp_full.trt";
        float frame_rate = 30.0; // in Hz
        int events_kept = 100'000; // number of events to keep for inference
    };

    class InferenceNode : public rclcpp::Node
    {
        using rclcpp::Node::Node;
    public:
        explicit InferenceNode(const rclcpp::NodeOptions &options);
        void inference_node_ctor(const std::string &t_node_name);
        ~InferenceNode();
        void start();
        void stop();
        bool isRunning() const;
        rcl_interfaces::msg::SetParametersResult paramsCallback(const std::vector<rclcpp::Parameter> &parameters);
    private:

        /// @brief Width and height of the event frame, grabbed at runtime
        int ev_width = 0;
        int ev_height = 0;

        int flow_width = 320;
        int flow_height = 256;

        /// @brief Number of channels in the event representation grid, should match the model
        const int grid_channels = 18; // hard-coded for now, should match the model
        const int inf_bsize = 1; // likely no need to change this

        /// @brief Parameter initialization
        inline void parameterInitilization() const;

        /// @brief Print parameters
        inline void parameterPrinter() const;

        /// @brief Reads the std library variables and ROS2 parameters
        /// @return true if all parameters are read successfully
        inline bool readParameters();

        /// @brief Update configuration for reconfiguration while running
        void updateConfiguration();

        /// @brief Event callback function for populating queue
        /// @param events EventArray message
        void eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events);

        /// @brief Slicer callback function
        void slicerCallback(const dv::EventStore &events);

        /// @brief Inference execution thread
        void execute_inference();

        void startSlicer();

        template <typename Iterable>
        void debug_inspect_iterable(const Iterable& iterable, const size_t min_show=5);

        template <typename EventStorage>
        void eventStoreToVoxelgrid(const EventStorage &events, size_t usable_events);
        Eigen::Tensor<float, 3, Eigen::RowMajor> m_prealloc_volume{this->grid_channels, this->flow_height, this->flow_width};
        std::unique_ptr<Eigen::Tensor<float, 3, Eigen::RowMajor>> m_volume_ptr = nullptr;
        std::vector<bool> m_flow_mask = std::vector<bool>(this->flow_height * this->flow_width, false);
        std::vector<float> m_voxel_grid_flat = std::vector<float>(this->grid_channels * this->flow_height * this->flow_width, 0.0f);
        std::vector<float> m_infer_storage = std::vector<float>(2 * this->flow_height * this->flow_width, 0.0f);
        std::vector<float> m_prev_storage = std::vector<float>(2 * this->flow_height * this->flow_width, 0.0f);

        void publish_flow(const std::vector<float> &flow);
        void flowToHSV(const cv::Mat &flow2ch);

        void publish_flow_vectors(const std::vector<float> &flow);

        long unsigned int m_frame_id = 0;

        /// @brief rclcpp node pointer
        rclcpp::Node::SharedPtr m_node;

        /// @brief Parameters
        Params m_params;

        // Thread related
        std::atomic<bool> m_spin_thread = true;
        std::thread m_inference_thread;

        /// @brief EventArray subscriber
        rclcpp::Subscription<dv_ros2_msgs::msg::EventArray>::SharedPtr m_events_subscriber;

        /// @brief Flow publisher
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_flow_rgb_publisher;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr m_flow_raw_publisher;

        /// @brief Event queue
        boost::lockfree::spsc_queue<dv::EventStore> m_event_queue{1'000};

        void inference_core(const dv::EventStore &events);

        /// @brief Slicer object
        std::unique_ptr<dv::EventStreamSlicer> m_slicer = nullptr;

        /// @brief Inference engine object
        std::unique_ptr<TrtEngine> m_inferencer = nullptr;

        /// @brief Job ID of the slicer, used to stop jobs running in the slicer
        std::optional<int> m_job_id;

        int64_t m_timestamp_offset = -1;

        cv::Mat m_flow2ch = cv::Mat::zeros(this->flow_height, this->flow_width, CV_32FC2);
        cv::Mat m_flow_rgb = cv::Mat::zeros(this->flow_height, this->flow_width, CV_8UC3);
        cv::Mat m_hsv_flow = cv::Mat::zeros(this->flow_height, this->flow_width, CV_8UC3);
    };
} // end namespace ros2_cpp_evfn_infer