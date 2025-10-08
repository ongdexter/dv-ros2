#!/usr/bin/env python3
import sys
import time
import argparse
import numpy as np
import h5py
import cv2


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

    print(f"Loaded {len(ex)} events from {args.h5file}; image size {width}x{height}")

    dt = args.dt_ms * 1000.0
    t0 = et[0] if len(et) > 0 else 0.0
    t_end = et[-1] if len(et) > 0 else 0.0
    cur_t = t0
    i = 0
    paused = False

    cv2.namedWindow('events', cv2.WINDOW_NORMAL)

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

