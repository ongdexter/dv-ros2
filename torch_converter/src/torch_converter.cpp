#include "torch_converter.hpp"

namespace torch_converter
{
    TorchConverter::TorchConverter(const rclcpp::NodeOptions &options)
        : Node("torch_converter", options), m_last_known_height(0), m_node{this}
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
        RCLCPP_INFO(m_node->get_logger(), "Constructor is initialized for node: %s", t_node_name.c_str());
        parameterInitilization();
        if(!readParameters())
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read parameters.");
            rclcpp::shutdown();
            std::exit(EXIT_FAILURE);
        }
        parameterPrinter();

        m_slicer = std::make_unique<dv::EventStreamSlicer>();

        m_events_tensor_cuda = torch::zeros({m_params.events_kept, 4}, torch::kFloat32).to(torch::kCUDA);
        m_events_tensor_cpu = torch::zeros({m_params.events_kept, 4}, torch::kFloat32).to(torch::kCPU).contiguous();

        std::cout << "Event Tensor shape: " << m_events_tensor_cpu.sizes() << std::endl;

        // Initialize event frame size from parameters
        ev_width = static_cast<float>(m_params.ev_width);
        ev_height = static_cast<float>(m_params.ev_height);
        ev_time = static_cast<float>(m_params.ev_time) * 1000.0f; // convert ms to us

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
        
        m_odom_subscriber = m_node->
            create_subscription<nav_msgs::msg::Odometry>(
                "/neurofly1/control_odom",
                10,
                std::bind(
                    &TorchConverter::odomCallback,
                    this,
                    std::placeholders::_1
                )
            );

        m_depth_publisher = m_node->
            create_publisher<sensor_msgs::msg::Image>(m_params.output_topic, 10);

        // m_ptcloud_publisher = m_node->
        //     create_publisher<sensor_msgs::msg::PointCloud2>(m_params.output_topic, 10);

        RCLCPP_INFO(m_node->get_logger(), "Sucessfully launched.");
    }

    void TorchConverter::start()
    {
        c10::InferenceMode guard;

        // start prepare inference engine
        try
        {
            RCLCPP_INFO(m_node->get_logger(), "Loading F3 model from: %s", m_params.f3_pt2_path.c_str());
            
            auto start_load = std::chrono::high_resolution_clock::now();
            f3_loader.emplace(m_params.f3_pt2_path);
            auto end_load = std::chrono::high_resolution_clock::now();

            auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
            RCLCPP_INFO(m_node->get_logger(), "F3 model loading time: %ld ms", load_duration.count());
        }
        catch (const c10::Error &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "Error loading F3 model: %s", e.what());
            rclcpp::shutdown();
            std::exit(EXIT_FAILURE);
        }

        try
        {
            RCLCPP_INFO(m_node->get_logger(), "Loading DAV2 model from: %s", m_params.dav2_pt2_path.c_str());

            auto start_load = std::chrono::high_resolution_clock::now();
            dav2_loader.emplace(m_params.dav2_pt2_path);
            auto end_load = std::chrono::high_resolution_clock::now();

            auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
            RCLCPP_INFO(m_node->get_logger(), "DAV2 model loading time: %ld ms", load_duration.count());
        }
        catch (const c10::Error &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "Error loading DAV2 model: %s", e.what());
            rclcpp::shutdown();
            std::exit(EXIT_FAILURE);
        }
        // end prepare inference engine

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
    }

    bool TorchConverter::isRunning() const
    {
        return m_spin_thread.load(std::memory_order_relaxed);
    }

    void TorchConverter::eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events)
    {
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
        m_event_value.write(events);
    }

    void TorchConverter::updateConfiguration()
    {
        if (m_job_id.has_value())
        {
            m_slicer->removeJob(m_job_id.value());
        }
        startSlicer();
    }

    void TorchConverter::startSlicer()
    {
        // convert frame_rate to ms (delta time)
        int32_t delta_time = static_cast<int>(1000 / m_params.frame_rate);
        m_job_id = m_slicer->
            doEveryTimeInterval(
                dv::Duration(delta_time * 1000LL),
                std::bind(&TorchConverter::slicerCallback, this, std::placeholders::_1)
            );
    }

    int32_t TorchConverter::eventStoreToTensor(const dv::EventStore &events)
    {
        float step = 1.0f;
        int64_t const N = events.size();
        int const usable_events = m_params.events_kept;

        // subsample events to fit into max_events
        if (N > usable_events)
            step = static_cast<float>(N) / static_cast<float>(usable_events);

        m_events_tensor_cpu.zero_();

        int64_t const m_timestamp_offset = events[0].timestamp();

        float64_t cnt = 0.0f; // original event index
        int32_t events_filled = 0; // tensor row index

        try
        {
        auto accessor = m_events_tensor_cpu.accessor<float, 2>();
        while (events_filled < usable_events)
        {
            int64_t const cnt_int = static_cast<int64_t>(cnt);
            if (cnt_int >= N)
                break;

            float const normalized_time = (events[cnt_int].timestamp() - m_timestamp_offset) / ev_time;

            if (normalized_time >= 1.0f)
                break;

            // direct memory access for speed using accessor
            accessor[events_filled][0] = static_cast<float>(events[cnt_int].x()) / ev_width;
            accessor[events_filled][1] = static_cast<float>(events[cnt_int].y()) / ev_height;
            accessor[events_filled][2] = normalized_time;
            accessor[events_filled][3] = static_cast<float>(events[cnt_int].polarity() ? 1.0f : 0.0f);

            // next event
            cnt += step;
            events_filled++;
        }
        }
        catch (const c10::Error &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "LibTorch error during event transfer to tensor: %s", e.what());
            return -1;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "Error during event transfer to tensor: %s", e.what());
            return -1;
        }

        return events_filled;
    }

    void TorchConverter::execute_inference()
    {
        c10::InferenceMode guard;

        RCLCPP_INFO(m_node->get_logger(), "Starting inference.");
        if (m_spin_thread == false)
        {
            RCLCPP_WARN(m_node->get_logger(), "Inference thread started while spin_thread is false.");
            return;
        }
        while (m_spin_thread)
        {
            m_event_value.consume([&](const dv::EventStore &events)
            {
                int const N = events.size();
                if (N == 0)
                {
                    RCLCPP_WARN(m_node->get_logger(), "No events to process in execute_inference.");
                    return;
                }

                // if (N <= 30000)
                // {
                //     RCLCPP_WARN(m_node->get_logger(), "Too few events (%d) received for inference, skipping.", N);
                //     return;
                // }

                RCLCPP_INFO(m_node->get_logger(), "Processing %d events in execute_inference.", N);
                int32_t const events_filled = eventStoreToTensor(events);
                RCLCPP_INFO(m_node->get_logger(), "Filled %d events into tensor.", events_filled);

                try
                {
                    m_events_tensor_cuda.copy_(m_events_tensor_cpu);
                }
                catch (const c10::Error &e)
                {
                    RCLCPP_ERROR(m_node->get_logger(), "Failed to move tensor to CUDA: %s", e.what());
                    return;
                }

                // prepare inputs
                torch::Tensor sliced_events_tensor_cuda = m_events_tensor_cuda.index({
                    torch::indexing::Slice(torch::indexing::None, events_filled)
                });
                std::vector<torch::Tensor> inputs = {sliced_events_tensor_cuda};

                RCLCPP_INFO(m_node->get_logger(),
                    "Input events tensor shape: [%ld, %ld]",
                    inputs[0].size(0),
                    inputs[0].size(1)
                );

                // Print the min and max x,y,t values
                RCLCPP_INFO(m_node->get_logger(),
                    "Input events tensor x min: %f, x max: %f, y min: %f, y max: %f, t min: %f, t max: %f",
                    inputs[0].index({torch::indexing::Slice(), 0}).min().item<float>(),
                    inputs[0].index({torch::indexing::Slice(), 0}).max().item<float>(),
                    inputs[0].index({torch::indexing::Slice(), 1}).min().item<float>(),
                    inputs[0].index({torch::indexing::Slice(), 1}).max().item<float>(),
                    inputs[0].index({torch::indexing::Slice(), 2}).min().item<float>(),
                    inputs[0].index({torch::indexing::Slice(), 2}).max().item<float>()
                );

                // run F3 model
                std::vector<torch::Tensor> f3_outputs;
                try
                {
                    f3_outputs = f3_loader.value().run(inputs);
                    RCLCPP_INFO(m_node->get_logger(),
                        "F3 model output tensor shape: [%ld, %ld, %ld, %ld]",
                        f3_outputs[0].size(0),
                        f3_outputs[0].size(1),
                        f3_outputs[0].size(2),
                        f3_outputs[0].size(3)
                    );
                }
                catch (const std::exception& e) {
                    RCLCPP_ERROR(m_node->get_logger(), "Error during F3 model inference: %s", e.what());
                } catch (const std::string& s) {
                    RCLCPP_ERROR(m_node->get_logger(), "String exception during F3 model inference: %s", s.c_str());
                } catch (...) {
                    RCLCPP_ERROR(m_node->get_logger(), "Unknown error during F3 model inference.");
                }

                // run DAV2 model
                std::vector<torch::Tensor> dav2_outputs;
                try
                {
                    torch::Tensor f3_feat = f3_outputs[0].permute({0, 3, 2, 1});

                    torch::Tensor f3_feat_ds = torch::nn::functional::interpolate(
                        f3_feat,
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{m_params.dav2_height, m_params.dav2_width})
                            .mode(torch::kBilinear)
                            .align_corners(true));

                    std::vector<torch::Tensor> dav2_inputs = {f3_feat_ds.to(torch::kFloat32)};

                    dav2_outputs = dav2_loader.value().run(dav2_inputs);

                    RCLCPP_INFO(m_node->get_logger(),
                        "DAV2 model output tensor shape: [%ld, %ld, %ld]",
                        dav2_outputs[0].size(0),
                        dav2_outputs[0].size(1),
                        dav2_outputs[0].size(2)
                    );
                }
                catch (const std::exception& e) {
                    RCLCPP_ERROR(m_node->get_logger(), "Error during DAV2 model inference: %s", e.what());
                } catch (const std::string& s) {
                    RCLCPP_ERROR(m_node->get_logger(), "String exception during DAV2 model inference: %s", s.c_str());
                } catch (...) {
                    RCLCPP_ERROR(m_node->get_logger(), "Unknown error during DAV2 model inference.");
                }

                // Generate point cloud from disparity predictions
                torch::Tensor depth;
                try
                {
                    // Get disparity output: (1, H, W)
                    torch::Tensor disparity = dav2_outputs[0].squeeze().to(torch::kCUDA).to(torch::kFloat32); // (H, W)

                    int64_t height = disparity.size(0);
                    int64_t width = disparity.size(1);

                    // Choose a scale factor to convert disparity to depth
                    // For relative depth, we use: depth = scale / (disparity + epsilon)
                    float scale = 10.0f;  // Can be tuned based on your scene
                    float epsilon = 1e-6f;  // Small value to avoid division by zero
                    // float known_flying_height = m_params.known_flying_height; // in meters
                    float known_flying_height = static_cast<float>(m_last_known_height); // in meters
                    RCLCPP_INFO(m_node->get_logger(),
                        "Using known flying height: %f meters for depth scaling.",
                        known_flying_height);

                    // Convert disparity to depth
                    depth = scale / (disparity + epsilon);

                    RCLCPP_INFO(m_node->get_logger(),
                        "Mean depth value: %f, Std depth value: %f, Min depth value: %f, Max depth value: %f",
                        depth.mean().item<float>(),
                        depth.std().item<float>(),
                        depth.min().item<float>(),
                        depth.max().item<float>()
                    );

                    // Camera intrinsics (assuming pinhole camera model)
                    float fx = 689.09169887014286f * static_cast<float>(width) / ev_width; // Focal length x
                    float fy = 689.42642441965040f * static_cast<float>(height) / ev_height; // Focal length y
                    float cx = 300.97918405659993f * static_cast<float>(width) / ev_width;  // Principal point x
                    float cy = 206.21218355427942f * static_cast<float>(height) / ev_height;  // Principal point y

                    RCLCPP_INFO(m_node->get_logger(),
                        "Updated Camera intrinsics: fx=%f, fy=%f, cx=%f, cy=%f",
                        fx, fy, cx, cy
                    );
                    
                    // Create pixel coordinate grids
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
                    torch::Tensor u = torch::arange(width, options).view({1, width}).expand({height, width});  // (H, W)
                    torch::Tensor v = torch::arange(height, options).view({height, 1}).expand({height, width}); // (H, W)

                    // Back-project to 3D coordinates
                    // torch::Tensor x = (u - cx) * depth / fx;  // (H, W)
                    torch::Tensor y = (v - cy) * depth / fy;  // (H, W)
                    // torch::Tensor z = depth;                   // (H, W)

                    // Get the pixel coordinates of the floor, this can be a point with minimum y
                    torch::Tensor flat_y = y.index({torch::indexing::Slice(298, torch::indexing::None)}).flatten();
                    int64_t max_idx = flat_y.argmax(/*dim=*/0).item<int64_t>();
                    int64_t floor_v = max_idx / width + 298;
                    int64_t floor_u = max_idx % width;

                    float32_t max_y_value = y.index({floor_v, floor_u}).item<float>();

                    RCLCPP_INFO(m_node->get_logger(),
                        "Floor found at pixel [u=%ld, v=%ld] with y=%f",
                        floor_u, floor_v, max_y_value);

                    float scale_factor = known_flying_height / max_y_value;
                    RCLCPP_INFO(m_node->get_logger(),
                        "Scaling factor for depth adjustment: %f",
                        scale_factor);

                    // Adjust depth using scale factor
                    // lets say max depth is 50 meters. We clamp depth before scaling to prevent overflow
                    depth = torch::clamp(depth, 0.0f, 20.0f / scale_factor);
                    depth = depth * scale_factor;

                    RCLCPP_INFO(m_node->get_logger(),
                        "After scaling - Mean depth value: %f, Std depth value: %f, Min depth value: %f, Max depth value: %f",
                        depth.mean().item<float>(),
                        depth.std().item<float>(),
                        depth.min().item<float>(),
                        depth.max().item<float>()
                    );
                    // y = y * scale_factor;  // (H, W)
                    // torch::Tensor x = (u - cx) * depth / fx;  // (H, W)
                    // torch::Tensor z = depth;                   // (H,

                    // Stack coordinates and reshape to point cloud: (H*W, 3)
                    // torch::Tensor points_3d = torch::stack({x, y, z}, /*dim=*/2).view({-1, 3});

                    // Move to CPU for publishing
                    // torch::Tensor point_cloud_cpu = points_3d.to(torch::kCPU).contiguous();

                    // RCLCPP_INFO(m_node->get_logger(),
                        // "Generated point cloud with %ld points from disparity map of size [%ld, %ld]",
                        // point_cloud_cpu.size(0), height, width);

                    // Publish the point cloud
                    // publish_point_cloud(point_cloud_cpu);
                }
                catch (const std::exception& e) {
                    RCLCPP_ERROR(m_node->get_logger(), "Error during point cloud generation: %s", e.what());
                } catch (const std::string& s) {
                    RCLCPP_ERROR(m_node->get_logger(), "String exception during point cloud generation: %s", s.c_str());
                } catch (...) {
                    RCLCPP_ERROR(m_node->get_logger(), "Unknown error during point cloud generation.");
                }

                try
                {
                    // torch::Tensor depth_tensor = dav2_outputs[0].squeeze().to(torch::kCPU); // (H, W)
                    depth = depth.to(torch::kCPU).contiguous() * 1000.0f; // convert to mm and move to CPU
                    depth = depth.to(torch::kUInt16);
                    publish_depth_raw(depth);
                    // publish_depth_image(depth);
                }
                catch (const std::exception& e) {
                    RCLCPP_ERROR(m_node->get_logger(), "Error during publishing depth image: %s", e.what());
                } catch (const std::string& s) {
                    RCLCPP_ERROR(m_node->get_logger(), "String exception during publishing depth image: %s", s.c_str());
                } catch (...) {
                    RCLCPP_ERROR(m_node->get_logger(), "Unknown error during publishing depth image.");
                }


            });
            std::this_thread::sleep_for(std::chrono::microseconds(100000));
        }
    }

    void TorchConverter::publish_point_cloud(torch::Tensor &point_cloud_tensor)
    {
        // point_cloud_tensor: (N, 3)
        sensor_msgs::msg::PointCloud2 pc_msg;

        pc_msg.header.stamp = m_node->get_clock()->now();
        pc_msg.header.frame_id = "map";
        pc_msg.height = 1;
        pc_msg.width = static_cast<uint32_t>(point_cloud_tensor.size(0));
        pc_msg.is_dense = false;
        pc_msg.is_bigendian = false;
        pc_msg.point_step = sizeof(float) * 3; // x, y, z
        pc_msg.row_step = pc_msg.point_step * pc_msg.width;

        // Define the fields (x, y, z)
        sensor_msgs::msg::PointField field_x;
        field_x.name = "x";
        field_x.offset = 0;
        field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;

        sensor_msgs::msg::PointField field_y;
        field_y.name = "y";
        field_y.offset = sizeof(float);
        field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;

        sensor_msgs::msg::PointField field_z;
        field_z.name = "z";
        field_z.offset = sizeof(float) * 2;
        field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;

        pc_msg.fields = {field_x, field_y, field_z};
        size_t n_bytes = static_cast<size_t>(pc_msg.width) * pc_msg.point_step;
        pc_msg.data.resize(n_bytes); // allocate space
        std::memcpy(pc_msg.data.data(), point_cloud_tensor.data_ptr(), n_bytes);

        m_ptcloud_publisher->publish(pc_msg);
    }

    void TorchConverter::publish_depth_raw(torch::Tensor &depth_tensor)
    {
        // depth_tensor: (H, W)
        // publish depth map
        sensor_msgs::msg::Image depth_msg;
        depth_msg.header.stamp = m_node->get_clock()->now();
        depth_msg.header.frame_id = "depth";
        depth_msg.height = static_cast<uint32_t>(m_params.dav2_height);
        depth_msg.width = static_cast<uint32_t>(m_params.dav2_width);

        size_t n_channels = 1;
        size_t n_bytes = static_cast<size_t>(depth_msg.height) * static_cast<size_t>(depth_msg.width) * n_channels * sizeof(uint16_t);
        depth_msg.data.resize(n_bytes); // allocate space
        depth_msg.encoding = "16UC1";
        depth_msg.is_bigendian = false;
        depth_msg.step = static_cast<uint32_t>(depth_msg.width * n_channels * sizeof(uint16_t));

        std::memcpy(depth_msg.data.data(), depth_tensor.data_ptr(), n_bytes);

        m_depth_publisher->publish(depth_msg);
    }

    void TorchConverter::publish_depth_image(torch::Tensor &depth_tensor)
    {
        // depth_tensor: (H, W)

        float32_t min_depth = depth_tensor.min().item<float32_t>();
        float32_t max_depth = depth_tensor.max().item<float32_t>();
        depth_tensor = (depth_tensor - min_depth) / (max_depth - min_depth) * 255.0f;
        depth_tensor = depth_tensor.to(torch::kUInt8);

        // publish depth map
        sensor_msgs::msg::Image depth_msg;
        depth_msg.header.stamp = m_node->get_clock()->now();
        depth_msg.header.frame_id = "depth";
        depth_msg.height = static_cast<uint32_t>(m_params.dav2_height);
        depth_msg.width = static_cast<uint32_t>(m_params.dav2_width);

        size_t n_channels = 1;
        size_t n_bytes = static_cast<size_t>(depth_msg.height) * static_cast<size_t>(depth_msg.width) * n_channels;
        depth_msg.data.resize(n_bytes); // allocate space
        depth_msg.encoding = "mono8";
        depth_msg.is_bigendian = false;
        depth_msg.step = static_cast<uint32_t>(depth_msg.width * n_channels);

        std::memcpy(depth_msg.data.data(), depth_tensor.data_ptr(), n_bytes);

        m_depth_publisher->publish(depth_msg);
    }

    void TorchConverter::odomCallback(nav_msgs::msg::Odometry::SharedPtr odom_msg)
    {
        m_last_known_height = odom_msg->pose.pose.position.z;
    }

    TorchConverter::~TorchConverter()
    {
        RCLCPP_INFO(m_node->get_logger(), "Destructor is activated.");
        stop();
        rclcpp::shutdown();
    }

    inline void TorchConverter::parameterInitilization() const
    {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        rcl_interfaces::msg::IntegerRange int_range;
        rcl_interfaces::msg::FloatingPointRange float_range;

        m_node->declare_parameter("input_topic", m_params.input_topic);
        m_node->declare_parameter("output_topic", m_params.output_topic);
        m_node->declare_parameter("f3_pt2_path", m_params.f3_pt2_path);
        m_node->declare_parameter("dav2_pt2_path", m_params.dav2_pt2_path);

        float_range.set__from_value(10.0).set__to_value(1000.0);
        descriptor.floating_point_range = {float_range};
        m_node->declare_parameter("frame_rate", m_params.frame_rate, descriptor);
        int_range.set__from_value(10'000).set__to_value(200'000).set__step(1);
        descriptor.integer_range = {int_range};
        m_node->declare_parameter("events_kept", m_params.events_kept, descriptor);
        m_node->declare_parameter("dav2_height", m_params.dav2_height);
        m_node->declare_parameter("dav2_width", m_params.dav2_width);
        m_node->declare_parameter("known_flying_height", m_params.known_flying_height);
    }

    inline void TorchConverter::parameterPrinter() const
    {
        RCLCPP_INFO(m_node->get_logger(), "-------- Parameters --------");
        RCLCPP_INFO(m_node->get_logger(), "input_topic: %s", m_params.input_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "output_topic: %s", m_params.output_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "f3_pt2_path: %s", m_params.f3_pt2_path.c_str());
        RCLCPP_INFO(m_node->get_logger(), "dav2_pt2_path: %s", m_params.dav2_pt2_path.c_str());
        RCLCPP_INFO(m_node->get_logger(), "frame_rate: %f", m_params.frame_rate);
        RCLCPP_INFO(m_node->get_logger(), "events_kept: %d", m_params.events_kept);
        RCLCPP_INFO(m_node->get_logger(), "dav2_height: %d", m_params.dav2_height);
        RCLCPP_INFO(m_node->get_logger(), "dav2_width: %d", m_params.dav2_width);
        RCLCPP_INFO(m_node->get_logger(), "known_flying_height: %f", m_params.known_flying_height);
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
        if (!m_node->get_parameter("f3_pt2_path", m_params.f3_pt2_path))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter f3_pt2_path.");
            return false;
        }
        if (!m_node->get_parameter("dav2_pt2_path", m_params.dav2_pt2_path))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter dav2_pt2_path.");
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
        if (!m_node->get_parameter("dav2_height", m_params.dav2_height))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter dav2_height.");
            return false;
        }
        if (!m_node->get_parameter("dav2_width", m_params.dav2_width))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter dav2_width.");
            return false;
        }
        if (!m_node->get_parameter("known_flying_height", m_params.known_flying_height))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter known_flying_height.");
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
            else if (param.get_name() == "f3_pt2_path")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.f3_pt2_path = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "f3_pt2_path must be a string";
                }
            }
            else if (param.get_name() == "dav2_pt2_path")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.dav2_pt2_path = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "dav2_pt2_path must be a string";
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
            else if (param.get_name() == "dav2_height")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.dav2_height = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "dav2_height must be an integer";
                }
            }
            else if (param.get_name() == "dav2_width")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.dav2_width = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "dav2_width must be an integer";
                }
            }
            else if (param.get_name() == "known_flying_height")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
                {
                    m_params.known_flying_height = param.as_double();
                }
                else
                {
                    result.successful = false;
                    result.reason = "known_flying_height must be a double";
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
