#include "torch_converter.hpp"

namespace torch_converter
{
    TorchConverter::TorchConverter(const rclcpp::NodeOptions &options)
        : Node("torch_converter", options), m_node{this}
    {
        RCLCPP_INFO(m_node->get_logger(), "[TorchConverter::TorchConverter] Initializing...");
        auto node_name = m_node->get_name();
        this->torch_converter_ctor(node_name);
        RCLCPP_INFO(m_node->get_logger(), "[TorchConverter::TorchConverter] Initialization complete!");
        RCLCPP_INFO(m_node->get_logger(), "[TorchConverter::TorchConverter] Beginning spin...");
        this->start();
    }

    void TorchConverter::torch_converter_ctor(const std::string &t_node_name)
    {
        RCLCPP_INFO(m_node->get_logger(), "Constructor is initialized.");
        parameterInitilization();
        if(!readParameters())
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read parameters.");
            rclcpp::shutdown();
            std::exit(EXIT_FAILURE);
        }
        parameterPrinter();

        m_slicer = std::make_unique<dv::EventStreamSlicer>();

        m_events_tensor_ptr = std::make_unique<torch::Tensor>(
            torch::zeros({m_params.events_kept, 4}, torch::kFloat32)
        );

        std::cout << "Tensor shape: " << m_events_tensor_ptr->sizes() << std::endl;

        m_events_subscriber = m_node->
            create_subscription<dv_ros2_msgs::msg::EventArray>(
                m_params.input_topic,
                10,
                std::bind(
                    &TorchConverter::eventCallback,
                    this,
                    std::placeholders::_1
                )
            );

        // TODO: determine publication type!
        // m_flow_publisher = m_node->
        //     create_publisher<sensor_msgs::msg::Image>(m_params.output_topic, 10);

        RCLCPP_INFO(m_node->get_logger(), "Sucessfully launched.");
    }

    void TorchConverter::start()
    {
        // prepare inference engine
        /*** RICHEEK: you can configure your pt2 engine here if you need to  ***/

        // start slicer
        startSlicer();

        // run thread
        m_spin_thread = true;
        m_inference_thread = std::thread(&TorchConverter::execute_inference, this);
        RCLCPP_INFO(m_node->get_logger(), "Inference thread is started.");
    }

    void TorchConverter::stop()
    {
        RCLCPP_INFO(m_node->get_logger(), "Stopping the inference thread...");
        
        // stop the thread first
        if (m_spin_thread)
        {
            m_spin_thread = false;
            m_inference_thread.join();
        }

        /*** RICHEEK: you can destroy your pt2 engine here if you need to ***/
    }

    bool TorchConverter::isRunning() const
    {
        return m_spin_thread.load(std::memory_order_relaxed);
    }

    void TorchConverter::eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events)
    {

        // Grabbing this at runtime!
        this->ev_width = events->width;
        this->ev_height = events->height;

        auto store = dv_ros2_msgs::toEventStore(*events);
        try
        {
            m_slicer->accept(store);
        }
        catch (std::out_of_range &e)
        {
            RCLCPP_WARN_STREAM(m_node->get_logger(), "Event out of range: " << e.what());
        }
    }

    void TorchConverter::slicerCallback(const dv::EventStore &events)
    {
        m_event_queue.push(events);
    }

    void TorchConverter::updateConfiguration()
    {
        if (m_job_id.has_value())
        {
            m_slicer->removeJob(m_job_id.value());
        }
        // if (m_inferencer != nullptr)
        // {
        //     stop();
        //     start();
        // }
        startSlicer();
    }

    void TorchConverter::startSlicer()
    {
        // convert frame_rate to ms (delta time)
        int32_t delta_time = static_cast<int>(1000 / m_params.frame_rate);
        m_job_id = m_slicer->doEveryTimeInterval(dv::Duration(delta_time * 1000LL), std::bind(&TorchConverter::slicerCallback, this, std::placeholders::_1));
    }

    void TorchConverter::eventStoreToTensor(const dv::EventStore &events, size_t usable_events)
    {
        // subsample events to fit into max_events
        int N = events.size();
        float step = static_cast<float>(N) / static_cast<float>(usable_events);
        if (N <= usable_events) {
            step = 1.0f;
        }


        // just to ensure tensor is on CPU for direct memory access
        if (m_events_tensor_ptr->is_cuda()) {
            RCLCPP_WARN(m_node->get_logger(), "Events tensor is on CUDA device; moving to CPU for direct memory access.");
            try {
                auto cpu = m_events_tensor_ptr->to(torch::kCPU);
                m_events_tensor_ptr = std::make_unique<torch::Tensor>(cpu);
            } catch (const c10::Error &e) {
                RCLCPP_ERROR(m_node->get_logger(), "Failed to move tensor to CPU: %s", e.what());
                return;
            }
        }

        // clear tensor as efficiently as possible - CPU ONLY
        auto t_contig = m_events_tensor_ptr->contiguous(); // ensure contiguous layout
        float* data = t_contig.data_ptr<float>();
        std::memset(data, 0, sizeof(float) * m_events_tensor_ptr->numel());

        // transfer data to tensor
        for (auto cnt = 0; cnt < events.size(); cnt += step) // cnt is the original event index
        {
            // idx is the index in the tensor
            auto idx = std::min(
                static_cast<int>(cnt / step),
                static_cast<int>(usable_events) - 1
            );

            // store event data - this API was very slow, not good enough for real-time
            // m_events_tensor_ptr->index_put_({idx, 0}, static_cast<float>(events[cnt].x()));
            // m_events_tensor_ptr->index_put_({idx, 1}, static_cast<float>(events[cnt].y()));
            // m_events_tensor_ptr->index_put_({idx, 2}, static_cast<float>(events[cnt].timestamp()));
            // m_events_tensor_ptr->index_put_({idx, 3}, static_cast<float>(events[cnt].polarity() ? 1.0f : -1.0f));

            // direct memory access for speed (row-major) - CPU ONLY
            auto n_cols = m_events_tensor_ptr->size(1);
            data[idx * n_cols + 0] = static_cast<float>(events[cnt].x());
            data[idx * n_cols + 1] = static_cast<float>(events[cnt].y());
            data[idx * n_cols + 2] = static_cast<float>(events[cnt].timestamp() - m_timestamp_offset);
            data[idx * n_cols + 3] = static_cast<float>(events[cnt].polarity() ? 1.0f : -1.0f);

            // next event
            cnt++;
        }

        std::cout << "Finished transferring: " << events[0].timestamp()
            << " to " << events[events.size() - 1].timestamp()
            << std::endl;
        std::cout << "T-diff on tensor: "
            << m_events_tensor_ptr->index({static_cast<int>(usable_events) - 1, 2}).item<float>()
            - m_events_tensor_ptr->index({0, 2}).item<float>()
            << std::endl;
    }

    void TorchConverter::execute_inference()
    {
        RCLCPP_INFO(m_node->get_logger(), "Starting inference.");
        if (m_spin_thread == false)
        {
            RCLCPP_WARN(m_node->get_logger(), "Inference thread started while spin_thread is false.");
            return;
        }
        while (m_spin_thread)
        {
            m_event_queue.consume_all([&](const dv::EventStore &events)
            {
                if (true) // RICHEEK: maybe add a check to determine if your pt2 is initialized
                {
                    if (m_timestamp_offset < 0 && events.size() > 0)
                    {
                        // THIS IS CRITICAL: large nsec timestamps break static_cast<float> conversion later
                        m_timestamp_offset = events[0].timestamp();
                        RCLCPP_INFO(m_node->get_logger(), "Timestamp offset set to %ld", m_timestamp_offset);
                    }

                    eventStoreToTensor(events, m_params.events_kept);
                    m_events_tensor_ptr->to(torch::kCUDA);

                    /*** RICHEEK: here you should have access to all data ***/
                    for (auto i = 0; i < 10; ++i)
                    {
                        RCLCPP_INFO(m_node->get_logger(),
                            "Event %zu:\tx=%f,\ty=%f,\tt=%f,\tp=%f", i,
                            m_events_tensor_ptr->index({i, 0}).item<float>(),
                            m_events_tensor_ptr->index({i, 1}).item<float>(),
                            m_events_tensor_ptr->index({i, 2}).item<float>(),
                            m_events_tensor_ptr->index({i, 3}).item<float>()
                        );
                    }
                    // you should now be able to process it with your pt2 model

                } else {
                    RCLCPP_WARN(m_node->get_logger(), "Inference engine is not initialized.");
                    return;
                }
            });
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    TorchConverter::~TorchConverter()
    {
        RCLCPP_INFO(m_node->get_logger(), "Destructor is activated.");
        stop();
        m_events_tensor_ptr.reset();
        rclcpp::shutdown();
    }

    inline void TorchConverter::parameterInitilization() const
    {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        rcl_interfaces::msg::IntegerRange int_range;
        rcl_interfaces::msg::FloatingPointRange float_range;

        m_node->declare_parameter("input_topic", m_params.input_topic);
        m_node->declare_parameter("output_topic", m_params.output_topic);
        m_node->declare_parameter("pt2_path", m_params.pt2_path);

        float_range.set__from_value(10.0).set__to_value(1000.0);
        descriptor.floating_point_range = {float_range};
        m_node->declare_parameter("frame_rate", m_params.frame_rate, descriptor);
        int_range.set__from_value(10'000).set__to_value(200'000).set__step(1);
        descriptor.integer_range = {int_range};
        m_node->declare_parameter("events_kept", m_params.events_kept, descriptor);
    }

    inline void TorchConverter::parameterPrinter() const
    {
        RCLCPP_INFO(m_node->get_logger(), "-------- Parameters --------");
        RCLCPP_INFO(m_node->get_logger(), "input_topic: %s", m_params.input_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "output_topic: %s", m_params.output_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "pt2_path: %s", m_params.pt2_path.c_str());
        RCLCPP_INFO(m_node->get_logger(), "frame_rate: %f", m_params.frame_rate);
        RCLCPP_INFO(m_node->get_logger(), "events_kept: %d", m_params.events_kept);
    }

    inline bool TorchConverter::readParameters()
    {
        if (!m_node->get_parameter("input_topic", m_params.input_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter input_topic.");
            return false;
        }
        if (!m_node->get_parameter("output_topic", m_params.output_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter output_topic.");
            return false;
        }
        if (!m_node->get_parameter("pt2_path", m_params.pt2_path))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter pt2_path.");
            return false;
        }
        if (!m_node->get_parameter("frame_rate", m_params.frame_rate))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter frame_rate.");
            return false;
        }
        if (!m_node->get_parameter("events_kept", m_params.events_kept))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter events_kept.");
            return false;
        }
        return true;
    }

    rcl_interfaces::msg::SetParametersResult TorchConverter::paramsCallback(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        
        for (const auto &param : parameters)
        {
            if (param.get_name() == "input_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.input_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "input_topic must be a string";
                }
            }
            else if (param.get_name() == "output_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.output_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "output_topic must be a string";
                }
            }
            else if (param.get_name() == "pt2_path")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.pt2_path = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "pt2_path must be a string";
                }
            }
            else if (param.get_name() == "frame_rate")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
                {
                    m_params.frame_rate = param.as_double();
                }
                else
                {
                    result.successful = false;
                    result.reason = "frame_rate must be a double";
                }
            }
            else if (param.get_name() == "events_kept")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.events_kept = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "events_kept must be an integer";
                }
            }
            else
            {
                result.successful = false;
                result.reason = "unknown parameter";
            }
        }
        updateConfiguration();
        return result;
    }


} // namespace torch_converter


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(torch_converter::TorchConverter)