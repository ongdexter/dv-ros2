#!/usr/bin/env python3
import sys
import os
import struct
from collections import defaultdict

import h5py

import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader
import rclpy
from rclpy.serialization import deserialize_message
import importlib

def parse_bag(input_path, output_path):

    # Init reader
    storage_options = StorageOptions(uri=input_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Build topic->type map from metadata
    topics = reader.get_all_topics_and_types()
    topic_type = {t.name: t.type for t in topics}

    # Prepare message type objects for deserialization
    msg_type_cache = {}
    def get_msg_type(type_name):
        # cache
        if type_name in msg_type_cache:
            return msg_type_cache[type_name]

        # fallback: dynamic import of the generated Python message package
        try:
            pkg, _, msg = type_name.partition('/msg/')
            if not pkg or not msg:
                # if the string uses a different separator
                parts = type_name.split('/')
                pkg = parts[0]
                msg = parts[-1]
            # import the msg module and get the class
            mod = importlib.import_module(f"{pkg}.msg")
            py_type = getattr(mod, msg)
        except Exception as e:
            raise ImportError(f"Failed to import message type {type_name}: {e}")

        msg_type_cache[type_name] = py_type
        return py_type

    # Prepare HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        # create groups
        events_grp = h5f.create_group('events')
        imu_grp = h5f.create_group('imu')

        # use lists to accumulate before writing datasets
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

        # iterate messages
        while reader.has_next():
            (topic, serialized_msg, t) = reader.read_next()
            t_sec = t
            # topic may be e.g. '/events' or '/imu'
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
                # dv_ros2_msgs/msg/EventArray
                # message has fields: header, height, width, events[]
                for ev in msg.events:
                    events_x.append(int(ev.x))
                    events_y.append(int(ev.y))
                    # convert builtin_interfaces/Time to float seconds
                    ev_time = ev.ts.sec + ev.ts.nanosec * 1e-9
                    # convert to microseconds
                    ev_us = int(ev_time * 1e6)
                    events_t.append(ev_us)
                    events_p.append(1 if ev.polarity else 0)

            elif topic.endswith('/imu') or topic == '/imu':
                # sensor_msgs/msg/Imu
                header = msg.header
                tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
                imu_t.append(tstamp)
                imu_ax.append(msg.linear_acceleration.x)
                imu_ay.append(msg.linear_acceleration.y)
                imu_az.append(msg.linear_acceleration.z)
                imu_wx.append(msg.angular_velocity.x)
                imu_wy.append(msg.angular_velocity.y)
                imu_wz.append(msg.angular_velocity.z)

        # Write datasets
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

    print(f"Saved events and IMU to {output_path}")


if __name__ == '__main__':

    input_path = "bags/ev_2025-10-08-18-20-54"
    output_path = "output.h5"

    parse_bag(input_path, output_path)

