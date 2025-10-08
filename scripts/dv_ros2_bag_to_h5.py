#!/usr/bin/env python3
import sys
import os
import struct
from collections import defaultdict
import math
import concurrent.futures

import h5py

import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader
import rclpy
from rclpy.serialization import deserialize_message
import importlib
import argparse

def parse_chunk(messages, topic_type_items):
    # Rebuild topic_type dict since data must be serializable for ProcessPoolExecutor
    topic_type = dict(topic_type_items)

    msg_type_cache = {}
    def get_msg_type(type_name):
        if type_name in msg_type_cache:
            return msg_type_cache[type_name]
        try:
            pkg, _, msg = type_name.partition('/msg/')
            if not pkg or not msg:
                parts = type_name.split('/')
                pkg = parts[0]
                msg = parts[-1]
            mod = importlib.import_module(f"{pkg}.msg")
            py_type = getattr(mod, msg)
        except Exception as e:
            raise ImportError(f"Failed to import message type {type_name}: {e}")
        msg_type_cache[type_name] = py_type
        return py_type

    events_x = []
    events_y = []
    events_t = []
    events_p = []

    imu_t = []
    imu_ax = []
    imu_ay = []
    imu_az = []
    imu_wx = []
    imu_wy = []
    imu_wz = []

    for topic, serialized_msg, t in messages:
        if topic not in topic_type:
            continue
        type_name = topic_type[topic]
        try:
            msg_type = get_msg_type(type_name)
            msg = deserialize_message(serialized_msg, msg_type)
        except Exception as e:
            # fallback: skip message
            print(f"Failed to deserialize message on {topic}: {e}")
            continue

        if topic.endswith('/events') or topic == '/events':
            for ev in msg.events:
                events_x.append(int(ev.x))
                events_y.append(int(ev.y))
                ev_time = ev.ts.sec + ev.ts.nanosec * 1e-9
                ev_us = int(ev_time * 1e6)
                events_t.append(ev_us)
                events_p.append(1 if ev.polarity else 0)

        elif topic.endswith('/imu') or topic == '/imu':
            header = msg.header
            tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            tstamp_us = int(tstamp * 1e6)
            imu_t.append(tstamp)
            imu_ax.append(msg.linear_acceleration.x)
            imu_ay.append(msg.linear_acceleration.y)
            imu_az.append(msg.linear_acceleration.z)
            imu_wx.append(msg.angular_velocity.x)
            imu_wy.append(msg.angular_velocity.y)
            imu_wz.append(msg.angular_velocity.z)

    print(f"Parsed chunk with {len(events_x)} events and {len(imu_t)} IMU readings")
    return (events_x, events_y, events_t, events_p,
            imu_t, imu_ax, imu_ay, imu_az, imu_wx, imu_wy, imu_wz)


def parallel_parse_bag(input_path, output_path, chunk_size=100, max_cores=10):
    storage_options = StorageOptions(uri=input_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_type = {t.name: t.type for t in topics}

    # Read all messages into memory
    all_messages = []
    while reader.has_next():
        all_messages.append(reader.read_next())

    num_chunks = math.ceil(len(all_messages) / chunk_size)

    print(f"Total messages: {len(all_messages)}, processing in {num_chunks} chunks of up to {chunk_size} messages each")

    # For ProcessPoolExecutor, send topic_type as list of items (serializable)
    topic_type_items = list(topic_type.items())

    # Use ProcessPoolExecutor for multi-core processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = []
        for i in range(num_chunks):
            chunk = all_messages[i * chunk_size:(i + 1) * chunk_size]
            futures.append(executor.submit(parse_chunk, chunk, topic_type_items))

        results = [future.result() for future in futures]

    # Merge results maintaining order
    events_x, events_y, events_t, events_p = [], [], [], []
    imu_t, imu_ax, imu_ay, imu_az, imu_wx, imu_wy, imu_wz = [], [], [], [], [], [], []

    for res in results:
        (res_events_x, res_events_y, res_events_t, res_events_p,
         res_imu_t, res_imu_ax, res_imu_ay, res_imu_az,
         res_imu_wx, res_imu_wy, res_imu_wz) = res

        events_x.extend(res_events_x)
        events_y.extend(res_events_y)
        events_t.extend(res_events_t)
        events_p.extend(res_events_p)

        imu_t.extend(res_imu_t)
        imu_ax.extend(res_imu_ax)
        imu_ay.extend(res_imu_ay)
        imu_az.extend(res_imu_az)
        imu_wx.extend(res_imu_wx)
        imu_wy.extend(res_imu_wy)
        imu_wz.extend(res_imu_wz)

    # Write to HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        events_grp = h5f.create_group('events')
        imu_grp = h5f.create_group('imu')

        events_grp.create_dataset('x', data=events_x, dtype='u4')
        events_grp.create_dataset('y', data=events_y, dtype='u4')
        events_grp.create_dataset('t', data=events_t, dtype='f8')
        events_grp.create_dataset('p', data=events_p, dtype='u1')

        imu_grp.create_dataset('t', data=imu_t, dtype='f8')
        imu_grp.create_dataset('ax', data=imu_ax, dtype='f8')
        imu_grp.create_dataset('ay', data=imu_ay, dtype='f8')
        imu_grp.create_dataset('az', data=imu_az, dtype='f8')
        imu_grp.create_dataset('wx', data=imu_wx, dtype='f8')
        imu_grp.create_dataset('wy', data=imu_wy, dtype='f8')
        imu_grp.create_dataset('wz', data=imu_wz, dtype='f8')

    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ROS2 MCAP bag to HDF5 with parallel processing.")
    parser.add_argument("--input", help="Path to input ROS2 MCAP bag")
    parser.add_argument("--output", default='output.h5', help="Path to output HDF5 file")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of messages per chunk (default: 100)")
    parser.add_argument("--max-cores", type=int, default=os.cpu_count()-2, help="Maximum number of CPU cores to use")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    chunk_size = args.chunk_size
    max_cores = args.max_cores

    print(f"Using up to {max_cores} CPU cores for processing")

    parallel_parse_bag(input_path, output_path, chunk_size, max_cores)
