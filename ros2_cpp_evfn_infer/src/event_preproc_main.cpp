#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <vector>
#include "event_preprocessing.cpp"
#include <random>
#include <Eigen/Dense>

using namespace ev_preproc;

float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

uint64_t process_random_batch()
{
    // voxelgrid parameters
    int height = 480;
    int width = 640;
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
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_diff = time_end - time_start;
    std::chrono::duration<double, std::milli> time_diff_subsample = time_1 - time_start;
    std::chrono::duration<double, std::milli> time_diff_voxel = time_end - time_1;
    std::cout <<
        "Processed " << events.size() << " events in " <<
        time_diff.count() << " ms (" <<
        time_diff_subsample.count() << " ms subsampling, " <<
        time_diff_voxel.count() << " ms voxelization)" << std::endl;

    // print some stats
    auto cnt = 0;
    for (auto t = 0; t < vol_size.T; ++t) {
        for (auto y = 0; y < vol_size.Y; ++y) {
            for (auto x = 0; x < vol_size.X; ++x) {
                cnt += (prealloc_volume(t, x, y) != 0) ? 1 : 0;
            }
        }
    }
    std::cout << "Number of non-zero elements in the volume: " << cnt << std::endl;

    return static_cast<uint64_t>(time_diff.count());
}

int main() {
    size_t num_batches = 10;
    std::vector<uint64_t> durations_ms;
    for (auto i = 0; i < num_batches; ++i)
    {
        auto dur = process_random_batch();
        durations_ms.push_back(dur);
    }

    double avg = Eigen::Map<Eigen::Array<uint64_t, Eigen::Dynamic, 1>>(
        durations_ms.data(), durations_ms.size()
    ).cast<double>().mean();
    double std = std::sqrt(
        (Eigen::Map<Eigen::Array<uint64_t, Eigen::Dynamic, 1>>(
                durations_ms.data(), durations_ms.size()
            ).cast<double>() - avg
        ).square().mean()
    );
    std::cout << "Average duration: " << avg << " ms, Std: " << std << " ms" << std::endl;

    return 0;
}
