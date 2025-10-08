#!/usr/bin/env python3
import sys
import time
import argparse
import numpy as np
import h5py
import cv2
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def draw_events_frame(xs, ys, ps, height, width):
    # create color image (uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(xs) > 0:
        xs = np.asarray(xs, dtype=np.int32)
        ys = np.asarray(ys, dtype=np.int32)
        ps = np.asarray(ps, dtype=np.int8)
        xs = np.clip(xs, 0, width-1)
        ys = np.clip(ys, 0, height-1)
        idxs = (ys, xs)
        # positive polarity: red (BGR: [0,0,255]), negative: green (BGR: [0,255,0])
        pos_mask = ps > 0
        neg_mask = ps <= 0
        img[ys[pos_mask], xs[pos_mask]] = np.array([0, 0, 255], dtype=np.uint8)
        img[ys[neg_mask], xs[neg_mask]] = np.array([0, 255, 0], dtype=np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file')
    parser.add_argument('--dt-ms', type=float, default=20.0, help='Frame duration in milliseconds')
    args = parser.parse_args()

    with h5py.File(args.h5file, 'r') as f:
        ex = f['events/x'][:]
        ey = f['events/y'][:]
        et = f['events/t'][:]
        ep = f['events/p'][:]

        # get image size if stored (optional)
        height = int(f['events'].attrs.get('height', 480))
        width = int(f['events'].attrs.get('width', 640))

        # optional IMU data (angular velocities and timestamps)
        if 'imu' in f:
            try:
                imu_tx = f['imu/t'][:]
                imu_wx = f['imu/wx'][:]
                imu_wy = f['imu/wy'][:]
                imu_wz = f['imu/wz'][:]
                has_imu = True
            except Exception:
                # missing expected imu datasets
                has_imu = False
                imu_tx = imu_wx = imu_wy = imu_wz = np.array([])
        else:
            has_imu = False
            imu_tx = imu_wx = imu_wy = imu_wz = np.array([])

    print(f"Loaded {len(ex)} events from {args.h5file}; image size {width}x{height}")
    print(f"Event timestamps from {et[0]*1e-6:.3f}s to {et[-1]*1e-6:.3f}s")
    print(f"IMU timestamps from {imu_tx[0]*1e-6:.3f}s to {imu_tx[-1]*1e-6:.3f}s" if has_imu and imu_tx.size > 0 else "No IMU data")

    dt = args.dt_ms * 1000.0
    t0 = et[0] if len(et) > 0 else 0.0
    t_end = et[-1] if len(et) > 0 else 0.0
    cur_t = t0
    i = 0
    paused = False

    cv2.namedWindow('events', cv2.WINDOW_NORMAL)

    # If IMU exists, set up a Matplotlib plot showing angular velocity and a moving time marker
    if has_imu and imu_tx.size > 0:
        plt.ion()
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
        ax[0].plot(imu_tx, imu_wx, color='r')
        ax[0].set_ylabel('wx')
        ax[1].plot(imu_tx, imu_wy, color='g')
        ax[1].set_ylabel('wy')
        ax[2].plot(imu_tx, imu_wz, color='b')
        ax[2].set_ylabel('wz')
        ax[2].set_xlabel('time (us)')
        # vertical marker line (initial at start)
        vlines = [ax[i].axvline(t0, color='k', lw=1) for i in range(3)]
        fig.canvas.draw()
        plt.show(block=False)

    start_t = t0

    while cur_t < t_end:
        if not paused:
            t1 = cur_t + dt
            # find events in [cur_t, t1)
            start_idx = i
            # advance i while events are before t1
            while i < len(et) and et[i] < t1:
                i += 1
            end_idx = i

            xs = ex[start_idx:end_idx]
            ys = ey[start_idx:end_idx]
            ps = ep[start_idx:end_idx]

            frame = draw_events_frame(xs, ys, ps, height, width)
            cur_t = t1

            # update IMU time marker if plot exists
            if has_imu and imu_tx.size > 0:
                for vl in vlines:
                    vl.set_xdata(cur_t)
                try:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                except Exception:
                    # interactive backend may not be available
                    pass

        cv2.imshow('events', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused = not paused
        elif key == ord('q') or key == 27:
            break
    end_t = cur_t
    print(f"Displayed events from t={start_t*1e-6:.3f}s to t={end_t*1e-6:.3f}s")
    print("Duration: {:.3f} seconds".format((end_t - start_t) * 1e-6))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

