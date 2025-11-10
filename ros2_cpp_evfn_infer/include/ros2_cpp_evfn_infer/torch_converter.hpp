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
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"

// dv_ros2_msgs Headers
#include "dv_ros2_msgs/msg/event.hpp"
#include "dv_ros2_msgs/msg/event_array.hpp"
#include "dv_ros2_messaging/messaging.hpp"

#include "trt_engine.hpp"
#include "event_preprocessing.hpp"

#include <torch/torch.h>
// #include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace torch_converter
{
    struct Params
    {
        /// @brief topic name of the input event array
        std::string input_topic = "events";
        /// @brief topic name of the output inference results
        std::string output_topic = "inference_results";
        /// @brief path to the TensorRT engine file
        std::string pt2_path =
            "/path/to/model.pt2";
        float frame_rate = 30.0; // in Hz
        int events_kept = 100'000; // number of events to keep for inference
    };

    class TorchConverter : public rclcpp::Node
    {
        using rclcpp::Node::Node;
    public:
        explicit TorchConverter(const rclcpp::NodeOptions &options);
        void torch_converter_ctor(const std::string &t_node_name);
        ~TorchConverter();
        void start();
        void stop();
        bool isRunning() const;
        rcl_interfaces::msg::SetParametersResult paramsCallback(const std::vector<rclcpp::Parameter> &parameters);
    private:

        /// @brief Width and height of the event frame, grabbed at runtime
        int ev_width = 0;
        int ev_height = 0;

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

        void eventStoreToTensor(const dv::EventStore &event_store, size_t usable_events);

        void startSlicer();

        long unsigned int m_frame_id = 0;

        /// @brief rclcpp node pointer
        rclcpp::Node::SharedPtr m_node;

        /// @brief Parameters
        Params m_params;

        // Thread related
        std::atomic<bool> m_spin_thread = true;
        std::thread m_inference_thread;

        std::unique_ptr<torch::Tensor> m_events_tensor_ptr;

        /// @brief EventArray subscriber
        rclcpp::Subscription<dv_ros2_msgs::msg::EventArray>::SharedPtr m_events_subscriber;

        /// @brief Event queue
        boost::lockfree::spsc_queue<dv::EventStore> m_event_queue{100};

        /// @brief Slicer object
        std::unique_ptr<dv::EventStreamSlicer> m_slicer = nullptr;

        /// @brief Job ID of the slicer, used to stop jobs running in the slicer
        std::optional<int> m_job_id;

        int64_t m_timestamp_offset = -1;
    };
} // end namespace torch_converter