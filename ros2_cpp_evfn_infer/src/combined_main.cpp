#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <random>
#include <memory>

#include "event_preprocessing.hpp"
#include "trt_engine.hpp"

/**
    Example showcasing how to:
        - preprocess events
        - run TensorRT inference

    To compile and run, firstly set the path to your trt engine in main() below.
    Then run the following:
        $ source env_cfg.sh
        $ g++ -O3 combined_main.cpp trt_engine.cpp \
            -I/root/dv-ros2/ros2_cpp_evfn_infer/include/ros2_cpp_evfn_infer/ \
            -I/usr/include/eigen3 \
            -I/usr/local/cuda-12.2/targets/aarch64-linux/include/ \
            -lnvinfer -lcudart -fopenmp \
            -o combo.out
        $ ./combo.out
*/

using namespace ev_preproc;

float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

int main()
{
    std::cout << "Starting event preprocessing and inference example..." << std::endl;

    // voxelgrid parameters
    int height = 256;
    int width = 320;
    int n_bins = 18;

    // simulate data
    size_t num_events = 1'000'000;
    int t_max = 10;
    std::vector<Event> events_raw;
    events_raw.reserve(num_events);
    for (auto i = 0; i < num_events; ++i)
    {
        float polarity = (random_float(0.0f, 1.0f) > 0.5f) ? 1.0f : -1.0f;
        events_raw.emplace_back(
            random_float(0.0f, static_cast<float>(height)),
            random_float(0.0f, static_cast<float>(width)),
            random_float(0.0f, static_cast<float>(t_max)),
            polarity
        );
    }

    // prepare to subsample to a manageable number of events
    size_t n_usable_events = 50'000;
    std::vector<Event> events;
    events.resize(n_usable_events);

    // define volume size
    VolumeSize vol_size{n_bins, height, width};

    // preallocate volume
    Eigen::Tensor<float, 3> prealloc_volume(vol_size.T, vol_size.X, vol_size.Y);

    // time event processing
    auto time_start = std::chrono::high_resolution_clock::now();
    subsample_events<Event>(events_raw, n_usable_events, events);
    auto time_1 = std::chrono::high_resolution_clock::now();
    gen_discretized_event_volume<Event>(events, vol_size, prealloc_volume);
    auto time_2 = std::chrono::high_resolution_clock::now();

    // to test output direct loading
    std::vector<float> output_storage_flat(height * width * 2);

    uint64_t proc_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_start).count();
    std::cout << "Event preprocessing completed in " << proc_time << " ms." << std::endl;
    std::cout << "Subsampling time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_start).count() << " ms." << std::endl;
    std::cout << "Voxel grid generation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count() << " ms." << std::endl;

    // Initialize TensorRT engine
    // **** TODO: set this path to your own TensorRT engine ****
    std::string engine_path =
        "/home/shared_external/ml_deployments/trt_engines/evfn_python_converted_fp_full.trt";
    TrtEngine trt_engine(engine_path);

    try {
        // one-time only: load engine and allocate buffers
        trt_engine.load();
        trt_engine.allocateBuffers();
        std::cout << "TensorRT engine loaded and buffers allocated." << std::endl;
        
        // first iteration will be warm-up and slow, subsequent iterations should be fast
        for (auto rep = 0; rep < 5; ++rep) {
            std::cout << "Running inference iteration " << rep + 1 << "..." << std::endl;        
            
            // benchmark input setting, inference, and output retrieval
            auto inf_start = std::chrono::high_resolution_clock::now();
            size_t n_elements = prealloc_volume.size();
            std::vector<float> input_flat(n_elements);
            // hopefully efficient flattening
            std::memcpy(input_flat.data(), prealloc_volume.data(), n_elements * sizeof(float));
            trt_engine.setInput(input_flat, 0);
            auto setup_end = std::chrono::high_resolution_clock::now();
            trt_engine.infer();
            auto inf_end = std::chrono::high_resolution_clock::now();
            auto outputs = trt_engine.getOutputs();
            auto output_end = std::chrono::high_resolution_clock::now();

            // check if this matches complete retrieval
            trt_engine.loadSingleOutput(output_storage_flat, 3);

            // print timing info
            std::cout << "Input setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - inf_start).count() << " ms." << std::endl;
            std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(inf_end - setup_end).count() << " ms." << std::endl;
            std::cout << "Output retrieval time: " << std::chrono::duration_cast<std::chrono::milliseconds>(output_end - inf_end).count() << " ms." << std::endl;
            std::cout << "Total TensorRT time: " << std::chrono::duration_cast<std::chrono::milliseconds>(output_end - inf_start).count() << " ms." << std::endl;

            // ensure output contains something
            std::cout << "Inference completed. Number of output bindings: " << outputs.size() << std::endl;
            for (size_t i = 0; i < outputs.size(); ++i) {
                std::cout << "Output " << i << " has " << outputs[i].size() << " elements." << std::endl;
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
                std::cout << "First 5 elements of output " << i << ": ";
                for (size_t j = 0; j < std::min(outputs[i].size(), size_t(5)); ++j) {
                    std::cout << outputs[i][j] << " ";
                }
                std::cout << std::endl;
                std::cout << "Last 5 elements of output " << i << ": ";
                for (size_t j = (outputs[i].size() >= 5 ? outputs[i].size() - 5 : 0); j < outputs[i].size(); ++j) {
                    std::cout << outputs[i][j] << " ";
                }
                std::cout << std::endl;

                if (i == 2) {
                    // compare to direct loading
                    std::cout << "Comparing output " << i << " to direct loading..." << std::endl;
                    bool match = true;
                    for (size_t j = 0; j < outputs[i].size(); ++j) {
                        if (outputs[i][j] != output_storage_flat[j]) {
                            match = false;
                            std::cout << "Mismatch at element " << j << ": output=" << outputs[i][j]
                                      << " direct_load=" << output_storage_flat[j] << std::endl;
                            break;
                        }
                    }
                    if (match) {
                        std::cout << "Output " << i << " matches direct loading." << std::endl;
                    } else {
                        std::cout << "Output " << i << " does NOT match direct loading." << std::endl;
                    }
                }
            }
            std::cout << "----------------------------------------" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during TensorRT operations: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}