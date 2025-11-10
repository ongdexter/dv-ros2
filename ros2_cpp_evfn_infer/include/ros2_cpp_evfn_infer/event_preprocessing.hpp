#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <unsupported/Eigen/CXX11/Tensor>
#include <tuple>
#include <omp.h>

// uncomment to disable assert() -> saves a few ms
#define NDEBUG
#include <cassert>

namespace ev_preproc
{

// tunable thread count
#ifndef OMP_NTHREADS
#define OMP_NTHREADS 8
#endif


struct Event {
    float x;
    float y;
    float t;
    float p;

    Event(float x_, float y_, float t_, float p_)
        : x(x_), y(y_), t(t_), p(p_) {}
    
    Event() : x(-1), y(-1), t(-1), p(0) {}
};

struct VolumeSize {
    int T;
    int H;
    int W;

    VolumeSize(int t, int h, int w) : T(t), H(h), W(w) {}
};

inline void store_csv(const cv::Mat& img, const std::string& filename="output.csv")
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            for (int c = 0; c < img.channels(); ++c) {
                file << img.at<cv::Vec3f>(i, j)[c];
                if (c < img.channels() - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }
    }

    file.close();
}

inline std::tuple<
    Eigen::VectorXi,
    Eigen::VectorXf,
    Eigen::VectorXi,
    Eigen::VectorXf
> calc_floor_ceil_delta(const Eigen::VectorXf& x) {
    Eigen::VectorXf x_floor = (x.array() + 1e-8).floor();
    Eigen::VectorXf x_ceil = (x.array() - 1e-8).ceil();
    Eigen::VectorXf x_ceil_fake = x.array().floor() + 1;

    Eigen::VectorXf delta_ce = (x.array() - x_floor.array());
    Eigen::VectorXf delta_fl = (x_ceil_fake.array() - x.array());
    return {
        x_floor.cast<int>(), delta_fl,
        x_ceil.cast<int>(), delta_ce
    };
}

inline std::tuple<
    Eigen::VectorXi,
    Eigen::VectorXf
> create_update(
    const Eigen::VectorXi& x,
    const Eigen::VectorXi& y,
    const Eigen::VectorXi& t,
    const Eigen::VectorXf& dt,
    const Eigen::VectorXi& p,
    const VolumeSize& vol_size
) {
    using namespace Eigen;

    assert((
        void("[create_update]: x values must be valid."),
        (x.array() >= 0).all() && (x.array() < vol_size.W).all()
    ));
    assert((
        void("[create_update]: y values must be valid."),
        (y.array() >= 0).all() && (y.array() < vol_size.H).all()
    ));
    int half_T = vol_size.T / 2 + 1;
    assert((
        void("[create_update]: t values must be valid."),
        (t.array() >= 0).all() && (t.array() < half_T).all()
    ));

    int N = p.size();
    Eigen::VectorXi inds(N);
    inds.setZero();

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (auto i = 0; i < N; ++i) {
        // negative polarity events go to the last half of the volume
        int vm = (p(i) < 0) ? half_T - 1 : 0;
        auto ind = (t(i) + vm) * vol_size.H * vol_size.W +
            y(i) * vol_size.W +
            x(i);
        inds(i) = ind;
    }

    return {inds, dt};
}

template <typename EventT, typename EventStorage = std::vector<EventT>>
inline void dv_gen_discretized_event_volume(
    const EventStorage& events,
    const size_t usable_events,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& prealloc_volume, // output tensor
    std::vector<bool>& flow_mask // output vector
) {
    using namespace Eigen;

    auto N = usable_events;
    VolumeSize vol_size{
        prealloc_volume.dimension(0l), // time bins
        prealloc_volume.dimension(1l), // height
        prealloc_volume.dimension(2l)  // width
    };
    prealloc_volume.setZero();

    std::vector<std::atomic<bool>> afm(flow_mask.size());
    for (auto& flag : afm) {
        flag.store(false);
    }

    VectorXi x(N), y(N), p(N);
    VectorXf t(N);

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (auto i = 0; i < N; ++i)
    {
        x(i) = static_cast<int>(events[i].x());
        y(i) = static_cast<int>(events[i].y());
        p(i) = events[i].polarity() ? 1 : -1;
        t(i) = static_cast<float>(events[i].timestamp());
    }

    Eigen::Index w_min, w_max;
    float t_min = t.minCoeff(&w_min);
    float t_max = t.maxCoeff(&w_max);
    float denom = std::max(1e-6f, t_max - t_min);
    float scale = (float(vol_size.T / 2 - 1)) / denom;
    VectorXf t_scaled = (t.array() - t_min) * scale;

    auto [t_fl, delta_fl, t_ce, delta_ce] = calc_floor_ceil_delta(t_scaled);

    auto [inds_fl, vals_fl] = create_update(x, y, t_fl, delta_fl, p, vol_size);
    auto [inds_ce, vals_ce] = create_update(x, y, t_ce, delta_ce, p, vol_size);

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (int i = 0; i < inds_fl.size(); ++i)
    {
        if (vals_fl(i) != 0) {
            int idx = inds_fl(i);
            int ti = idx / (vol_size.W * vol_size.H);
            int rem = idx % (vol_size.W * vol_size.H);
            int row_i = rem / vol_size.W;
            int col_i = rem % vol_size.W;
            prealloc_volume(ti, row_i, col_i) += vals_fl(i);
            afm[rem].store(true);
        }
        if (vals_ce(i) != 0) {
            int idx = inds_ce(i);
            int ti = idx / (vol_size.W * vol_size.H);
            int rem = idx % (vol_size.W * vol_size.H);
            int row_i = rem / vol_size.W;
            int col_i = rem % vol_size.W;
            prealloc_volume(ti, row_i, col_i) += vals_ce(i);
            afm[rem].store(true);
        }
    }

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (size_t i = 0; i < flow_mask.size(); ++i) {
        flow_mask[i] = afm[i].load();
    }
}


template <typename EventT, typename EventStorage = std::vector<EventT>>
inline void gen_discretized_event_volume(
    const EventStorage& events,
    const VolumeSize& vol_size,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& prealloc_volume // output tensor, shaped as vol_size
) {
    using namespace Eigen;

    assert((
        void("[gen_discretized_event_volume]: prealloc_volume must have vol_size dimensions."),
        prealloc_volume.dimension(0l) == vol_size.T &&
        prealloc_volume.dimension(1l) == vol_size.X &&
        prealloc_volume.dimension(2l) == vol_size.Y
    ));

    int N = events.size();
    prealloc_volume.setZero();

    VectorXi x(N), y(N), p(N);
    VectorXf t(N);

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (auto i = 0; i < N; ++i)
    {
        x(i) = static_cast<int>(events[i].x);
        y(i) = static_cast<int>(events[i].y);
        t(i) = events[i].t;
        p(i) = static_cast<int>(events[i].p);
    }

    float t_min = t.minCoeff();
    float t_max = t.maxCoeff();
    float denom = std::max(1e-6f, t_max - t_min);
    float scale = (float(vol_size.T / 2)) / denom;
    VectorXf t_scaled = (t.array() - t_min) * scale;

    auto [t_fl, delta_fl, t_ce, delta_ce] = calc_floor_ceil_delta(t_scaled);

    auto [inds_fl, vals_fl] = create_update(x, y, t_fl, delta_fl, p, vol_size);
    auto [inds_ce, vals_ce] = create_update(x, y, t_ce, delta_ce, p, vol_size);

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (int i = 0; i < inds_fl.size(); ++i)
    {
        if (vals_fl(i) != 0.0f) {
            int idx = inds_fl(i);
            int ti = idx / (vol_size.W * vol_size.H);
            int rem = idx % (vol_size.W * vol_size.H);
            int row_i = rem / vol_size.W;
            int col_i = rem % vol_size.W;
            prealloc_volume(ti, row_i, col_i) += static_cast<float>(vals_fl(i));
        }
        if (vals_ce(i) != 0.0f) {
            int idx = inds_ce(i);
            int ti = idx / (vol_size.W * vol_size.H);
            int rem = idx % (vol_size.W * vol_size.H);
            int row_i = rem / vol_size.W;
            int col_i = rem % vol_size.W;
            prealloc_volume(ti, row_i, col_i) += static_cast<float>(vals_ce(i));
        }
    }
}


template <typename EventT, typename EventStorage = std::vector<EventT>>
inline void subsample_events(
    const EventStorage& events,
    int max_events,
    std::vector<EventT>& out_events_prealloc
) {
    int N = events.size();
    float step = static_cast<float>(N) / static_cast<float>(max_events);
    if (N <= max_events) {
        step = 1.0f;
    }

    // instead of randomness, simplify to even-spaced sampling
    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (auto i = 0; i < max_events; ++i)
    {
        int idx = static_cast<int>(std::floor(i * step));
        idx = std::min(idx, N - 1);
        out_events_prealloc[i] = events[idx];
    }
}

template <typename EventT>
inline size_t dv_event_roi(
    const std::vector<EventT>& events,
    int x_min, int x_max,
    int y_min, int y_max,
    std::vector<EventT>& out_events_prealloc
) {
    auto cnt = 0;
    for (const auto& event : events)
    {
        if (event.x() >= x_min && event.x() < x_max &&
            event.y() >= y_min && event.y() < y_max)
        {
            out_events_prealloc[cnt] = event;
            cnt++;
        }
    }
    return cnt;
}


template <typename EventT, typename EventStorage = std::vector<EventT>>
inline size_t dv_event_stride(
    const EventStorage& events,
    int stride,
    int pad_x,
    int pad_y,
    int64_t timestamp_offset,
    std::vector<EventT>& out_events_prealloc
) {
    auto cnt = 0;
    bool neg_ts_flag = false;
    int64_t min_ts = std::numeric_limits<int64_t>::max();
    int64_t max_ts = std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < events.size(); i += stride)
    {
        if (events[i].x() % stride == 0) continue;
        if (events[i].y() % stride == 0) continue;
        out_events_prealloc[cnt] = dv::Event(
            events[i].timestamp() - timestamp_offset,
            events[i].x() / stride + pad_x,
            events[i].y() / stride + pad_y,
            events[i].polarity()
        );
        cnt++;

        // IMPORTANT: drop events if we exceed preallocated size
        // TODO: maybe somehow randomize dropping?
        if (cnt >= out_events_prealloc.size()) {
            break;
        }
        if (out_events_prealloc[cnt - 1].timestamp() < min_ts) {
            min_ts = out_events_prealloc[cnt - 1].timestamp();
        }
        if (out_events_prealloc[cnt - 1].timestamp() > max_ts) {
            max_ts = out_events_prealloc[cnt - 1].timestamp();
        }

        neg_ts_flag = neg_ts_flag || out_events_prealloc[cnt].timestamp() < 0;
    }

    // if (neg_ts_flag) {
    //     std::cout << "[dv_event_stride]: Warning, negative timestamps detected after offset adjustment." << std::endl;
    // }

    // std::cout << "[T I M E] min ts: " << min_ts << ", max ts: " << max_ts << std::endl;

    return cnt;
}

inline void dv_adjust_event_timestamps(
    std::vector<dv::Event>& events, // in-place modification
    int64_t timestamp_offset,
    size_t usable_events
) {
    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (size_t i = 0; i < usable_events; ++i)
    {
        events[i] = dv::Event(
            events[i].timestamp() - timestamp_offset,
            events[i].x(),
            events[i].y(),
            events[i].polarity()
        );
    }
}

inline void dv_event_interpolate(
    const Eigen::Tensor<float, 3>& input_volume,
    Eigen::Tensor<float, 3>& output_volume
) {
    // Simple nearest-neighbor interpolation
    assert((
        void("[dv_event_interpolate]: input and output volumes must have same number of channels."),
        input_volume.dimension(0) == output_volume.dimension(0)
    ));
    int channels = input_volume.dimension(0);
    int in_x = input_volume.dimension(1);
    int in_y = input_volume.dimension(2);
    int out_x = output_volume.dimension(1);
    int out_y = output_volume.dimension(2);

    float scale_x = static_cast<float>(in_x) / static_cast<float>(out_x);
    float scale_y = static_cast<float>(in_y) / static_cast<float>(out_y);

    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (int c = 0; c < channels; ++c) {
        for (int x = 0; x < out_x; ++x) {
            for (int y = 0; y < out_y; ++y) {
                int src_x = std::min(
                    static_cast<int>(std::round(static_cast<float>(x) * scale_x)),
                    in_x - 1
                );
                int src_y = std::min(
                    static_cast<int>(std::round(static_cast<float>(y) * scale_y)),
                    in_y - 1
                );
                output_volume(c, x, y) = input_volume(c, src_x, src_y);
            }
        }
    }
}

} // namespace ev_preproc