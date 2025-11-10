#include "inference_node.hpp"

namespace ros2_cpp_evfn_infer
{
    InferenceNode::InferenceNode(const rclcpp::NodeOptions &options)
        : Node("inference_node", options), m_node{this}
    {
        RCLCPP_INFO(m_node->get_logger(), "[InferenceNode::InferenceNode] Initializing...");
        auto node_name = m_node->get_name();
        this->inference_node_ctor(node_name);
        RCLCPP_INFO(m_node->get_logger(), "[InferenceNode::InferenceNode] Initialization complete!");
        RCLCPP_INFO(m_node->get_logger(), "[InferenceNode::InferenceNode] Beginning spin...");
        this->start();
    }

    void InferenceNode::inference_node_ctor(const std::string &t_node_name)
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

        m_events_subscriber = m_node->
            create_subscription<dv_ros2_msgs::msg::EventArray>(
                m_params.input_topic,
                10,
                std::bind(
                    &InferenceNode::eventCallback,
                    this,
                    std::placeholders::_1
                )
            );

        m_flow_rgb_publisher = m_node->
            create_publisher<sensor_msgs::msg::Image>(m_params.flow_rgb_topic, 10);
        m_flow_raw_publisher = m_node->
            create_publisher<std_msgs::msg::Float32MultiArray>(m_params.flow_raw_topic, 10);

        RCLCPP_INFO(m_node->get_logger(), "Sucessfully launched.");
    }

    void InferenceNode::start()
    {
        // prepare inference engine
        m_inferencer = std::make_unique<TrtEngine>(m_params.engine_path);
        
        RCLCPP_INFO(m_node->get_logger(), "Loading TensorRT engine from: %s", m_params.engine_path.c_str());
        m_inferencer->load();
        
        RCLCPP_INFO(m_node->get_logger(), "Allocating device buffers...");
        m_inferencer->allocateBuffers();

        RCLCPP_INFO(m_node->get_logger(), "TensorRT engine is ready - starting slicer and inference thread.");
        // start slicer
        startSlicer();

        // run thread
        m_spin_thread = true;
        m_inference_thread = std::thread(&InferenceNode::execute_inference, this);
        RCLCPP_INFO(m_node->get_logger(), "Inference thread is started.");
    }

    void InferenceNode::stop()
    {
        RCLCPP_INFO(m_node->get_logger(), "Stopping the inference thread...");
        
        // stop the thread first
        if (m_spin_thread)
        {
            m_spin_thread = false;
            m_inference_thread.join();
        }
        
        // unique_ptr's reset should call destructor of trt engine
        m_inferencer.reset();
    }

    bool InferenceNode::isRunning() const
    {
        return m_spin_thread.load(std::memory_order_relaxed);
    }

    void InferenceNode::eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events)
    {

        // Grabbing this at runtime!
        this->ev_width = events->width;
        this->ev_height = events->height;
        if (m_volume_ptr == nullptr)
        {
            m_volume_ptr = std::make_unique<Eigen::Tensor<float, 3, Eigen::RowMajor>>(
                this->grid_channels,
                this->ev_height,
                this->ev_width
            );
        }

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

    void InferenceNode::slicerCallback(const dv::EventStore &events)
    {
        m_event_queue.push(events);
    }

    void InferenceNode::updateConfiguration()
    {
        if (m_job_id.has_value())
        {
            m_slicer->removeJob(m_job_id.value());
        }
        if (m_inferencer != nullptr)
        {
            stop();
            start();
        }
        startSlicer();
    }

    void InferenceNode::startSlicer()
    {
        // convert frame_rate to ms (delta time)
        int32_t delta_time = static_cast<int>(1000 / m_params.frame_rate);
        m_job_id = m_slicer->doEveryTimeInterval(
            dv::Duration(delta_time * 1000LL),
            std::bind(
                &InferenceNode::slicerCallback,
                this,
                std::placeholders::_1
            )
        );
    }

        
    // void saveVariantToHDF5(H5::H5File& file, const std::string& key, const H5DataVariant& value) {
    //     std::visit([&](auto&& data) {
    //         using T = std::decay_t<decltype(data)>;

    //         if constexpr (std::is_same_v<T, std::string>) {
    //             // Store string as a scalar dataset
    //             H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    //             H5::DataSpace scalar_space(H5S_SCALAR);
    //             auto dataset = file.createDataSet(key, str_type, scalar_space);
    //             dataset.write(data, str_type);
    //         }
    //         else if constexpr (std::is_arithmetic_v<T>) {
    //             // Single numeric scalar
    //             H5::DataSpace scalar_space(H5S_SCALAR);
    //             H5::PredType type;
    //             if constexpr (std::is_same_v<T, int>) type = H5::PredType::NATIVE_INT;
    //             else if constexpr (std::is_same_v<T, float>) type = H5::PredType::NATIVE_FLOAT;
    //             else if constexpr (std::is_same_v<T, double>) type = H5::PredType::NATIVE_DOUBLE;
    //             else throw std::runtime_error("Unsupported scalar type");
    //             auto dataset = file.createDataSet(key, type, scalar_space);
    //             dataset.write(&data, type);
    //         }
    //         else if constexpr (std::is_same_v<T, std::vector<int>> ||
    //                         std::is_same_v<T, std::vector<float>> ||
    //                         std::is_same_v<T, std::vector<double>>) {
    //             // 1D numeric array
    //             const hsize_t dims[1] = { data.size() };
    //             H5::DataSpace space(1, dims);
    //             H5::PredType type;
    //             if constexpr (std::is_same_v<typename T::value_type, int>) type = H5::PredType::NATIVE_INT;
    //             else if constexpr (std::is_same_v<typename T::value_type, float>) type = H5::PredType::NATIVE_FLOAT;
    //             else if constexpr (std::is_same_v<typename T::value_type, double>) type = H5::PredType::NATIVE_DOUBLE;
    //             else throw std::runtime_error("Unsupported vector type");
    //             auto dataset = file.createDataSet(key, type, space);
    //             dataset.write(data.data(), type);
    //         }
    //         else {
    //             throw std::runtime_error("Unsupported data type in variant");
    //         }
    //     }, value);
    // }


    // void saveHeterogeneousData(const std::map<std::string, H5DataVariant>& items, const std::string& filepath) {
    //     H5::H5File file(filepath, H5F_ACC_TRUNC);
    //     for (const auto& [key, value] : items)
    //         saveVariantToHDF5(file, key, value);
    // }


    template <typename EventStorage>
    void InferenceNode::eventStoreToVoxelgrid(const EventStorage &events, size_t usable_events)
    {
        ev_preproc::dv_gen_discretized_event_volume<dv::Event, EventStorage>(
            events, usable_events, m_prealloc_volume, m_flow_mask
        );
        size_t n_elements = m_prealloc_volume.size();
        std::copy_n(m_prealloc_volume.data(), n_elements, m_voxel_grid_flat.data());
        // RCLCPP_INFO(m_node->get_logger(), "Voxel grid updated with %zu elements.", n_elements);
    }

    void InferenceNode::flowToHSV(const cv::Mat &flow2ch)
    {
        // expects a 2-channel float (or convertible) image where channels are (u, v)
        if (flow2ch.empty() || flow2ch.channels() != 2)
        {
            RCLCPP_WARN(m_node->get_logger(), "flowToHSV: input must be a non-empty 2-channel image.");
            return;
        }

        cv::Mat flowf;
        if (flow2ch.depth() != CV_32F)
            flow2ch.convertTo(flowf, CV_32F);
        else
            flowf = flow2ch;

        const int rows = flowf.rows;
        const int cols = flowf.cols;

        // find maximum magnitude for normalization
        std::priority_queue<float> mag_pqueue;
        for (int r = 0; r < rows; ++r)
        {
            const cv::Vec2f* ptr = flowf.ptr<cv::Vec2f>(r);
            for (int c = 0; c < cols; ++c)
            {
                const cv::Vec2f &v = ptr[c];
                const float mag = std::hypot(v[0], v[1]);
                mag_pqueue.push(mag);
            }
        }
        float max_mag = mag_pqueue.empty() ? 1.0f : mag_pqueue.top();

        // create HSV image
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {

                if (!m_flow_mask[r * cols + c])
                {
                    m_hsv_flow.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
                    continue;
                }

                const cv::Vec2f* in_ptr = flowf.ptr<cv::Vec2f>(r);
                cv::Vec3b* out_ptr = m_hsv_flow.ptr<cv::Vec3b>(r);
                const cv::Vec2f &v = in_ptr[c];
                const float ux = v[0];
                const float uy = v[1];
                const float mag = std::hypot(ux, uy);

                // Hue: encode direction (angle). Map [-pi, pi] -> [0,179]
                float angle = std::atan2(uy, ux); // [-pi, pi]
                float hue_f = (angle + static_cast<float>(M_PI)) / (2.0f * static_cast<float>(M_PI)); // [0,1]
                int hue = static_cast<int>(std::round(hue_f * 179.0f));
                hue = std::clamp(hue, 0, 179);

                // Saturation/Value: encode magnitude normalized by max_mag
                uint8_t sat = 0;
                uint8_t val = 0;
                if (max_mag > std::numeric_limits<float>::epsilon())
                {
                    float norm = std::min(1.0f, mag / max_mag);
                    sat = static_cast<uint8_t>(std::round(1 * 255.0f));
                    val = static_cast<uint8_t>(std::round(norm * 255.0f));
                }

                out_ptr[c] = cv::Vec3b(static_cast<uint8_t>(hue), sat, val);
            }
        }
        // result stored in m_hsv_flow
    }

    void InferenceNode::publish_flow_vectors(const std::vector<float> &flow)
    {
        const size_t expected_size =
            static_cast<size_t>(2) * this->flow_height * this->flow_width;
        if (flow.size() != expected_size)
        {
            RCLCPP_WARN_STREAM(m_node->get_logger(),
                "Unexpected flow size: " << flow.size()
                    << " (expected " << expected_size << ")");
            return;
        }

        // prepare and publish custom message here
        std_msgs::msg::Float32MultiArray flow_msg;
        flow_msg.layout.dim.resize(3);
        flow_msg.layout.dim[0].label = "channels";
        flow_msg.layout.dim[0].size = 2;
        flow_msg.layout.dim[0].stride = 2 * this->flow_height * this->flow_width;
        flow_msg.layout.dim[1].label = "height";
        flow_msg.layout.dim[1].size = this->flow_height;
        flow_msg.layout.dim[1].stride = this->flow_height * this->flow_width;
        flow_msg.layout.dim[2].label = "width";
        flow_msg.layout.dim[2].size = this->flow_width;
        flow_msg.layout.dim[2].stride = this->flow_width;
        flow_msg.data.resize(expected_size);

        // RCLCPP_INFO(m_node->get_logger(), "Publishing flow vectors of size %zu.", flow.size());

        std::copy_n(flow.data(), flow.size(), flow_msg.data.data());

        // RCLCPP_INFO(m_node->get_logger(), "Publishing flow vectors...");
        m_flow_raw_publisher->publish(flow_msg);
    }

    void InferenceNode::publish_flow(const std::vector<float> &flow)
    {
        const size_t expected_size =
            static_cast<size_t>(2) * this->flow_height * this->flow_width;
        if (flow.size() != expected_size)
        {
            RCLCPP_WARN_STREAM(m_node->get_logger(),
                "Unexpected flow size: " << flow.size()
                    << " (expected " << expected_size << ")");
            return;
        }

        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> flow_tensor(
            const_cast<float*>(flow.data()),
            2, this->flow_height, this->flow_width
        );

        m_flow2ch.setTo(cv::Scalar(0, 0)); // reset
        for (int h = 0; h < this->flow_height; ++h)
        {
            for (int w = 0; w < this->flow_width; ++w)
            {
                // only show flow for pixels with events
                if (m_flow_mask[h * this->flow_width + w])
                {
                    m_flow2ch.at<cv::Vec2f>(h, w)[0] = flow_tensor(0, h, w);
                    m_flow2ch.at<cv::Vec2f>(h, w)[1] = flow_tensor(1, h, w);
                }
            }
        }

        flowToHSV(m_flow2ch);
        m_flow_rgb.setTo(cv::Scalar(0, 0, 0)); // reset
        cv::cvtColor(m_hsv_flow, m_flow_rgb, cv::COLOR_HSV2BGR);


        // initialize image message
        sensor_msgs::msg::Image img;
        img.header.stamp = m_node->get_clock()->now();
        img.header.frame_id = "flow";
        img.height = static_cast<uint32_t>(this->flow_height);
        img.width = static_cast<uint32_t>(this->flow_width);
        
        size_t n_channels = 3;
        size_t n_bytes = static_cast<size_t>(img.height) * static_cast<size_t>(img.width) * n_channels;
        img.data.resize(n_bytes); // allocate space for all channels
        img.encoding = "rgb8"; // 8-bit
        img.is_bigendian = false;
        img.step = static_cast<uint32_t>(img.width * n_channels * sizeof(uint8_t));

        // RCLCPP_INFO(m_node->get_logger(), "Preparing to publish flow image of size %zu bytes.", img.data.size());

        // // copy data over, filling the third channel potentially with zeros
        // for (auto h = 0; h < this->flow_height; ++h)
        // {
        //     for (auto w = 0; w < this->flow_width; ++w)
        //     {
        //         // mask-out non-event pixels
        //         if (m_flow_mask[h * this->flow_width + w] == false)
        //         {
        //             color_mat.at<cv::Vec3b>(h, w)[0] = 0;
        //             color_mat.at<cv::Vec3b>(h, w)[1] = 0;
        //             color_mat.at<cv::Vec3b>(h, w)[2] = 0;
        //             continue;
        //         }

        //         auto flow_x = flow_tensor(0, h, w);
        //         auto flow_y = flow_tensor(1, h, w);

        //         // map flow to color
        //         // u -> R, v -> G, B = 0
        //         color_mat.at<cv::Vec3b>(h, w)[0] = static_cast<uint8_t>(std::min(std::abs(flow_x) * this->flow_height * 255.0f, 255.0f));
        //         color_mat.at<cv::Vec3b>(h, w)[1] = static_cast<uint8_t>(std::min(std::abs(flow_y) * this->flow_width * 255.0f, 255.0f));
        //         color_mat.at<cv::Vec3b>(h, w)[2] = 0;
        //     }
        // }

        std::copy_n(m_flow_rgb.data, img.data.size(), img.data.data());

        // actually publish
        m_flow_rgb_publisher->publish(img);
        
        // // store to disk for debugging
        // std::ostringstream ss;
        // ss << "/tmp/flow_store/flow_frame_"
        //     << std::setw(8) << std::setfill('0') << m_frame_id++
        //     << "_mono.csv";
        // std::string path = ss.str();
        // cv::Mat img_from_msg = cv::Mat(img.height, img.width, (n_channels == 1) ? CV_8UC1 : CV_8UC3, img.data.data()).clone();
        // ev_preproc::store_csv(img_from_msg, path);
    }

    template <typename Iterable>
    void InferenceNode::debug_inspect_iterable(const Iterable& iterable, const size_t min_show)
    {
        const size_t n = iterable.size();
        if (n == 0)
        {
            RCLCPP_WARN(m_node->get_logger(), "iterable is empty.");
        }
        else
        {
            const size_t show = std::min<size_t>(min_show, n);

            std::ostringstream oss_first;
            oss_first << "iterable (size=" << n << ") first " << show << ": [";
            for (size_t i = 0; i < show; ++i)
            {
                if (i) oss_first << ", ";
                oss_first << iterable[i];
            }
            oss_first << "]";
            RCLCPP_INFO_STREAM(m_node->get_logger(), oss_first.str());

            std::ostringstream oss_last;
            oss_last << "iterable (size=" << n << ") last " << show << ": [";
            size_t start = (n > show) ? (n - show) : 0;
            for (size_t i = start; i < n; ++i)
            {
                if (i != start) oss_last << ", ";
                oss_last << iterable[i];
            }
            oss_last << "]";
            RCLCPP_INFO_STREAM(m_node->get_logger(), oss_last.str());
        }

        auto cnt = 0;
        for (const auto& val : iterable)
        {
            if (val != 0.0f)
            {
                cnt++;
            }
        }
        RCLCPP_INFO_STREAM(m_node->get_logger(), "iterable has " << cnt << " non-zero elements.");
    }

    void InferenceNode::inference_core(const dv::EventStore &events)
    {
        if (m_inferencer != nullptr)
        {
            if (events.size() > 0)
            {
                // THIS IS CRITICAL: large usec timestamps break static_cast<float> conversion later
                m_timestamp_offset = events[0].timestamp();
            }

            std::vector<dv::Event> ev_vec_roi(m_params.events_kept);


            /*** WARNING: This code is brittle, only tested for VGA-evfn default resolutions. ***/
            int spatial_subsample_factor = std::round(static_cast<float>(ev_width) / static_cast<float>(flow_width));
            int pad_x = (-ev_width / spatial_subsample_factor + flow_width) / 2;
            int pad_y = (-ev_height / spatial_subsample_factor + flow_height) / 2;
            
            // downsample with stride
            // this function also fixes:
            // - timestamp offset
            // - removes superfluous events beyond m_params.events_kept
            auto n_valid = ev_preproc::dv_event_stride<dv::Event, dv::EventStore>(
                events,
                spatial_subsample_factor,
                pad_x,
                pad_y,
                m_timestamp_offset,
                ev_vec_roi
            );


            RCLCPP_INFO(m_node->get_logger(), "Running inference on %zu events, down from %zu.",
                n_valid, events.size()
            );

            eventStoreToVoxelgrid<std::vector<dv::Event>>(ev_vec_roi, n_valid);

            m_inferencer->setInput(m_voxel_grid_flat, 0);
            m_inferencer->infer();
            m_inferencer->loadSingleOutput(m_infer_storage, 3); // total binding ordering

            publish_flow_vectors(m_infer_storage);
            publish_flow(m_infer_storage);
        } else {
            RCLCPP_WARN(m_node->get_logger(), "Inference engine is not initialized.");
            return;
        }
    }

    void InferenceNode::execute_inference()
    {
        RCLCPP_INFO(m_node->get_logger(), "Starting inference.");
        if (m_spin_thread == false)
        {
            RCLCPP_WARN(m_node->get_logger(), "Inference thread started while spin_thread is false.");
            return;
        }
        while (m_spin_thread)
        {
            m_event_queue.consume_all(
                std::bind(&InferenceNode::inference_core, this, std::placeholders::_1)
            );
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    InferenceNode::~InferenceNode()
    {
        RCLCPP_INFO(m_node->get_logger(), "Destructor is activated.");
        stop();
        rclcpp::shutdown();
    }

    inline void InferenceNode::parameterInitilization() const
    {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        rcl_interfaces::msg::IntegerRange int_range;
        rcl_interfaces::msg::FloatingPointRange float_range;

        m_node->declare_parameter("input_topic", m_params.input_topic);
        m_node->declare_parameter("flow_rgb_topic", m_params.flow_rgb_topic);
        m_node->declare_parameter("flow_raw_topic", m_params.flow_raw_topic);
        m_node->declare_parameter("engine_path", m_params.engine_path);

        float_range.set__from_value(10.0).set__to_value(1000.0);
        descriptor.floating_point_range = {float_range};
        m_node->declare_parameter("frame_rate", m_params.frame_rate, descriptor);
        int_range.set__from_value(10'000).set__to_value(200'000).set__step(1);
        descriptor.integer_range = {int_range};
        m_node->declare_parameter("events_kept", m_params.events_kept, descriptor);
    }

    inline void InferenceNode::parameterPrinter() const
    {
        RCLCPP_INFO(m_node->get_logger(), "-------- Parameters --------");
        RCLCPP_INFO(m_node->get_logger(), "input_topic: %s", m_params.input_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "flow_rgb_topic: %s", m_params.flow_rgb_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "flow_raw_topic: %s", m_params.flow_raw_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "engine_path: %s", m_params.engine_path.c_str());
        RCLCPP_INFO(m_node->get_logger(), "frame_rate: %f", m_params.frame_rate);
        RCLCPP_INFO(m_node->get_logger(), "events_kept: %d", m_params.events_kept);
    }

    inline bool InferenceNode::readParameters()
    {
        if (!m_node->get_parameter("input_topic", m_params.input_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter input_topic.");
            return false;
        }
        if (!m_node->get_parameter("flow_rgb_topic", m_params.flow_rgb_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter flow_rgb_topic.");
            return false;
        }
        if (!m_node->get_parameter("flow_raw_topic", m_params.flow_raw_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter flow_raw_topic.");
            return false;
        }
        if (!m_node->get_parameter("engine_path", m_params.engine_path))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter engine_path.");
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

    rcl_interfaces::msg::SetParametersResult InferenceNode::paramsCallback(const std::vector<rclcpp::Parameter> &parameters)
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
            else if (param.get_name() == "flow_rgb_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.flow_rgb_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "flow_rgb_topic must be a string";
                }
            }
            else if (param.get_name() == "flow_raw_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.flow_raw_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "flow_raw_topic must be a string";
                }
            }
            else if (param.get_name() == "engine_path")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.engine_path = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "engine_path must be a string";
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


} // namespace ros2_cpp_evfn_infer


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_cpp_evfn_infer::InferenceNode)