#!/usr/bin/env python3
import sys
import os
import struct
from collections import defaultdict
import math
import concurrent.futures

import h5py

import numpy as np

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
        pkg, _, msg = type_name.partition('/msg/')
        if not pkg or not msg:
            parts = type_name.split('/')
            pkg = parts[0]
            msg = parts[-1]
        mod = importlib.import_module(f"{pkg}.msg")
        py_type = getattr(mod, msg)
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

    # Images for neurofly zed rgb
    images_neuro_t = []
    images_neuro_h = []
    images_neuro_w = []
    images_neuro_encoding = []
    images_neuro_data = []

    # Images for flow rgb
    images_flow_t = []
    images_flow_h = []
    images_flow_w = []
    images_flow_encoding = []
    images_flow_data = []

    # Flow raw (Float32MultiArray) from /flow/raw
    flow_raw_t = []
    flow_raw_shapes = []
    flow_raw_data = []

    # Odometry for vicon (/neurofly1/control_odom) and zed (/neurofly1/zed_node/odom)
    vicon_t = []
    vicon_px = []
    vicon_py = []
    vicon_pz = []
    vicon_qx = []
    vicon_qy = []
    vicon_qz = []
    vicon_qw = []
    vicon_vx = []
    vicon_vy = []
    vicon_vz = []
    vicon_avx = []
    vicon_avy = []
    vicon_avz = []

    zed_t = []
    zed_px = []
    zed_py = []
    zed_pz = []
    zed_qx = []
    zed_qy = []
    zed_qz = []
    zed_qw = []
    zed_vx = []
    zed_vy = []
    zed_vz = []
    zed_avx = []
    zed_avy = []
    zed_avz = []

    for topic, serialized_msg, t in messages:
        # if topic not in topic_type:
        #     continue
        type_name = topic_type[topic]
        msg_type = get_msg_type(type_name)
        msg = deserialize_message(serialized_msg, msg_type)

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
            imu_t.append(tstamp_us)
            imu_ax.append(msg.linear_acceleration.x)
            imu_ay.append(msg.linear_acceleration.y)
            imu_az.append(msg.linear_acceleration.z)
            imu_wx.append(msg.angular_velocity.x)
            imu_wy.append(msg.angular_velocity.y)
            imu_wz.append(msg.angular_velocity.z)

        # Vicon odometry
        elif topic.endswith('/control_odom'):
            header = msg.header
            tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            tstamp_us = int(tstamp * 1e6)
            vicon_t.append(tstamp_us)

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            twist = msg.twist.twist

            vicon_px.append(pos.x)
            vicon_py.append(pos.y)
            vicon_pz.append(pos.z)
            vicon_qx.append(ori.x)
            vicon_qy.append(ori.y)
            vicon_qz.append(ori.z)
            vicon_qw.append(ori.w)

            vicon_vx.append(twist.linear.x)
            vicon_vy.append(twist.linear.y)
            vicon_vz.append(twist.linear.z)
            vicon_avx.append(twist.angular.x)
            vicon_avy.append(twist.angular.y)
            vicon_avz.append(twist.angular.z)

        # ZED odometry
        elif topic.endswith('/zed_node/odom'):
            header = msg.header
            tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            tstamp_us = int(tstamp * 1e6)
            zed_t.append(tstamp_us)

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            twist = msg.twist.twist

            zed_px.append(pos.x)
            zed_py.append(pos.y)
            zed_pz.append(pos.z)
            zed_qx.append(ori.x)
            zed_qy.append(ori.y)
            zed_qz.append(ori.z)
            zed_qw.append(ori.w)

            zed_vx.append(twist.linear.x)
            zed_vy.append(twist.linear.y)
            zed_vz.append(twist.linear.z)
            zed_avx.append(twist.angular.x)
            zed_avy.append(twist.angular.y)
            zed_avz.append(twist.angular.z)

        elif topic.endswith('/neurofly1/zed_node/left/image_rect_color'):
            # sensor_msgs/msg/Image
            header = msg.header
            tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            tstamp_us = int(tstamp * 1e6)
            h = int(msg.height)
            w = int(msg.width)
            encoding = getattr(msg, 'encoding', '')
            # msg.data may be bytes or list of ints
            data_bytes = bytes(msg.data)
            images_neuro_t.append(tstamp_us)
            images_neuro_h.append(h)
            images_neuro_w.append(w)
            images_neuro_encoding.append(encoding)
            images_neuro_data.append(data_bytes)

        elif topic.endswith('/flow/rgb'):
            # sensor_msgs/msg/Image
            header = msg.header
            tstamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            tstamp_us = int(tstamp * 1e6)
            h = int(msg.height)
            w = int(msg.width)
            encoding = getattr(msg, 'encoding', '')
            data_bytes = bytes(msg.data)
            images_flow_t.append(tstamp_us)
            images_flow_h.append(h)
            images_flow_w.append(w)
            images_flow_encoding.append(encoding)
            images_flow_data.append(data_bytes)

        elif topic.endswith('/flow/raw'):
            # std_msgs/Float32MultiArray
            header = getattr(msg, 'layout', None)
            # timestamp from msg if available (some msgs may not have header)
            header_ts = getattr(msg, 'header', None)
            tstamp_us = None
            if header_ts is not None:
                tstamp = header_ts.stamp.sec + header_ts.stamp.nanosec * 1e-9
                tstamp_us = int(tstamp * 1e6)

            # parse layout dims
            dims = [int(d.size) for d in msg.layout.dim]

            # fallback timestamp: use reader-provided t
            tstamp_us = int(t * 1e-3)

            # data is a sequence of floats
            arr = np.array(msg.data, dtype=np.float32)
            data_bytes = arr.tobytes()

            flow_raw_t.append(tstamp_us)
            flow_raw_shapes.append(dims)
            flow_raw_data.append(data_bytes)

    print(f"Parsed chunk with {len(events_x)} events, {len(imu_t)} IMU readings, {len(images_neuro_t)} neuro images, {len(images_flow_t)} flow images, {len(vicon_t)} vicon, {len(zed_t)} zed")
    out = {}
    if events_x:
        out['events'] = (events_x, events_y, events_t, events_p)
    if imu_t:
        out['imu'] = (imu_t, imu_ax, imu_ay, imu_az, imu_wx, imu_wy, imu_wz)
    if images_neuro_t:
        out['neuro_images'] = (images_neuro_t, images_neuro_h, images_neuro_w, images_neuro_encoding, images_neuro_data)
    if images_flow_t:
        out['flow_images'] = (images_flow_t, images_flow_h, images_flow_w, images_flow_encoding, images_flow_data)
    if flow_raw_t:
        out['flow_raw'] = (flow_raw_t, flow_raw_shapes, flow_raw_data)
    if vicon_t:
        out['vicon_odom'] = (vicon_t, vicon_px, vicon_py, vicon_pz, vicon_qx, vicon_qy, vicon_qz, vicon_qw,
                             vicon_vx, vicon_vy, vicon_vz, vicon_avx, vicon_avy, vicon_avz)
    if zed_t:
        out['zed_odom'] = (zed_t, zed_px, zed_py, zed_pz, zed_qx, zed_qy, zed_qz, zed_qw,
                           zed_vx, zed_vy, zed_vz, zed_avx, zed_avy, zed_avz)
    return out


def parallel_parse_bag(input_path, output_path, chunk_size=100, max_cores=10):
    storage_options = StorageOptions(uri=input_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_type = {t.name: t.type for t in topics}

    # We'll stream messages in chunks from the reader and submit parse jobs
    # in batches to avoid holding all messages/results in memory.
    topic_type_items = list(topic_type.items())
    chunk_index = 0
    print(f"Streaming messages and processing in chunks of up to {chunk_size} messages")

    # Create HDF5 file and datasets upfront with resizable dimensions
    # Use optimizations: chunking, compression, and larger write buffers
    with h5py.File(output_path, 'w', libver='latest') as h5f:
        events_grp = h5f.create_group('events')
        imu_grp = h5f.create_group('imu')

        # Create resizable datasets with chunking for better I/O performance
        # Chunk size of 100000 balances memory and I/O efficiency
        chunk_size_events = 100000
        chunk_size_imu = 10000
        
        events_grp.create_dataset('x', shape=(0,), maxshape=(None,), dtype='u4', 
                                   chunks=(chunk_size_events,), compression='gzip', compression_opts=1)
        events_grp.create_dataset('y', shape=(0,), maxshape=(None,), dtype='u4', 
                                   chunks=(chunk_size_events,), compression='gzip', compression_opts=1)
        events_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', 
                                   chunks=(chunk_size_events,), compression='gzip', compression_opts=1)
        events_grp.create_dataset('p', shape=(0,), maxshape=(None,), dtype='u1', 
                                   chunks=(chunk_size_events,), compression='gzip', compression_opts=1)

        imu_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('ax', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('ay', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('az', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('wx', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('wy', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)
        imu_grp.create_dataset('wz', shape=(0,), maxshape=(None,), dtype='f8', 
                                chunks=(chunk_size_imu,), compression='gzip', compression_opts=1)

        # Odometry groups for vicon and zed
        vicon_grp = h5f.create_group('vicon_odom')
        zed_grp = h5f.create_group('zed_odom')

        # create resizable odometry datasets (timestamps in us, positions, orientation, linear & angular velocities)
        odom_chunks = (chunk_size_imu,)
        for grp in (vicon_grp, zed_grp):
            grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('px', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('py', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('pz', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('qx', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('qy', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('qz', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('qw', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('vx', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('vy', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('vz', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('avx', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('avy', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)
            grp.create_dataset('avz', shape=(0,), maxshape=(None,), dtype='f8', chunks=odom_chunks, compression='gzip', compression_opts=1)

        # Image datasets
        images_grp = h5f.create_group('images')
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        str_dt = h5py.string_dtype(encoding='utf-8')

        neuro_grp = images_grp.create_group('neurofly_rgb')
        # datasets for neuro images will be created lazily on first append:
        # neuro_grp['images'] -> uint8 dataset with shape (N, H, W, C)
        # neuro_grp['t'] -> timestamps

        flow_grp = images_grp.create_group('flow_rgb')
        # datasets for flow images will be created lazily on first append
        flow_raw_grp = images_grp.create_group('flow_raw')
        vlen_float = h5py.vlen_dtype(np.float32)
        vlen_int = h5py.vlen_dtype(np.int32)

        # Use ProcessPoolExecutor for multi-core processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            futures = []
            flush_interval = 10  # Flush every N chunks instead of every chunk

            # Stream reader -> submit -> consume in-order when batch full
            while True:
                # build one chunk from the reader
                chunk = []
                for _ in range(chunk_size):
                    if not reader.has_next():
                        break
                    chunk.append(reader.read_next())

                if not chunk:
                    break

                # submit the chunk for parsing
                futures.append(executor.submit(parse_chunk, chunk, topic_type_items))

                # when we have max_cores futures queued, process them in order
                if len(futures) >= max_cores:
                    for future in futures:
                        res = future.result()

                        # events
                        res_events_x = res_events_y = res_events_t = res_events_p = None
                        if 'events' in res:
                            (res_events_x, res_events_y, res_events_t, res_events_p) = res['events']

                        # imu
                        res_imu_t = None
                        res_imu_ax = res_imu_ay = res_imu_az = res_imu_wx = res_imu_wy = res_imu_wz = None
                        if 'imu' in res:
                            (res_imu_t, res_imu_ax, res_imu_ay, res_imu_az, res_imu_wx, res_imu_wy, res_imu_wz) = res['imu']

                        # images neuro
                        res_images_neuro_t = res_images_neuro_h = res_images_neuro_w = res_images_neuro_encoding = res_images_neuro_data = None
                        if 'neuro_images' in res:
                            (res_images_neuro_t, res_images_neuro_h, res_images_neuro_w, res_images_neuro_encoding, res_images_neuro_data) = res['neuro_images']

                        # images flow
                        res_images_flow_t = res_images_flow_h = res_images_flow_w = res_images_flow_encoding = res_images_flow_data = None
                        if 'flow_images' in res:
                            (res_images_flow_t, res_images_flow_h, res_images_flow_w, res_images_flow_encoding, res_images_flow_data) = res['flow_images']

                        # odometry
                        res_vicon = None
                        res_zed = None
                        if 'vicon_odom' in res:
                            res_vicon = res['vicon_odom']
                        if 'zed_odom' in res:
                            res_zed = res['zed_odom']

                        # Append events data incrementally
                        if res_events_x:
                            current_events_size = events_grp['x'].shape[0]
                            new_events_size = current_events_size + len(res_events_x)
                            
                            events_grp['x'].resize((new_events_size,))
                            events_grp['y'].resize((new_events_size,))
                            events_grp['t'].resize((new_events_size,))
                            events_grp['p'].resize((new_events_size,))
                            
                            events_grp['x'][current_events_size:new_events_size] = res_events_x
                            events_grp['y'][current_events_size:new_events_size] = res_events_y
                            events_grp['t'][current_events_size:new_events_size] = res_events_t
                            events_grp['p'][current_events_size:new_events_size] = res_events_p

                        # Append IMU data incrementally
                        if res_imu_t:
                            current_imu_size = imu_grp['t'].shape[0]
                            new_imu_size = current_imu_size + len(res_imu_t)
                            
                            imu_grp['t'].resize((new_imu_size,))
                            imu_grp['ax'].resize((new_imu_size,))
                            imu_grp['ay'].resize((new_imu_size,))
                            imu_grp['az'].resize((new_imu_size,))
                            imu_grp['wx'].resize((new_imu_size,))
                            imu_grp['wy'].resize((new_imu_size,))
                            imu_grp['wz'].resize((new_imu_size,))
                            
                            imu_grp['t'][current_imu_size:new_imu_size] = res_imu_t
                            imu_grp['ax'][current_imu_size:new_imu_size] = res_imu_ax
                            imu_grp['ay'][current_imu_size:new_imu_size] = res_imu_ay
                            imu_grp['az'][current_imu_size:new_imu_size] = res_imu_az
                            imu_grp['wx'][current_imu_size:new_imu_size] = res_imu_wx
                            imu_grp['wy'][current_imu_size:new_imu_size] = res_imu_wy
                            imu_grp['wz'][current_imu_size:new_imu_size] = res_imu_wz

                        # Append vicon odometry incrementally
                        if res_vicon:
                            (res_vicon_t, res_vicon_px, res_vicon_py, res_vicon_pz, res_vicon_qx, res_vicon_qy, res_vicon_qz, res_vicon_qw,
                             res_vicon_vx, res_vicon_vy, res_vicon_vz, res_vicon_avx, res_vicon_avy, res_vicon_avz) = res_vicon

                            current_vicon_size = vicon_grp['t'].shape[0]
                            new_vicon_size = current_vicon_size + len(res_vicon_t)

                            vicon_grp['t'].resize((new_vicon_size,))
                            vicon_grp['px'].resize((new_vicon_size,))
                            vicon_grp['py'].resize((new_vicon_size,))
                            vicon_grp['pz'].resize((new_vicon_size,))
                            vicon_grp['qx'].resize((new_vicon_size,))
                            vicon_grp['qy'].resize((new_vicon_size,))
                            vicon_grp['qz'].resize((new_vicon_size,))
                            vicon_grp['qw'].resize((new_vicon_size,))
                            vicon_grp['vx'].resize((new_vicon_size,))
                            vicon_grp['vy'].resize((new_vicon_size,))
                            vicon_grp['vz'].resize((new_vicon_size,))
                            vicon_grp['avx'].resize((new_vicon_size,))
                            vicon_grp['avy'].resize((new_vicon_size,))
                            vicon_grp['avz'].resize((new_vicon_size,))

                            vicon_grp['t'][current_vicon_size:new_vicon_size] = res_vicon_t
                            vicon_grp['px'][current_vicon_size:new_vicon_size] = res_vicon_px
                            vicon_grp['py'][current_vicon_size:new_vicon_size] = res_vicon_py
                            vicon_grp['pz'][current_vicon_size:new_vicon_size] = res_vicon_pz
                            vicon_grp['qx'][current_vicon_size:new_vicon_size] = res_vicon_qx
                            vicon_grp['qy'][current_vicon_size:new_vicon_size] = res_vicon_qy
                            vicon_grp['qz'][current_vicon_size:new_vicon_size] = res_vicon_qz
                            vicon_grp['qw'][current_vicon_size:new_vicon_size] = res_vicon_qw
                            vicon_grp['vx'][current_vicon_size:new_vicon_size] = res_vicon_vx
                            vicon_grp['vy'][current_vicon_size:new_vicon_size] = res_vicon_vy
                            vicon_grp['vz'][current_vicon_size:new_vicon_size] = res_vicon_vz
                            vicon_grp['avx'][current_vicon_size:new_vicon_size] = res_vicon_avx
                            vicon_grp['avy'][current_vicon_size:new_vicon_size] = res_vicon_avy
                            vicon_grp['avz'][current_vicon_size:new_vicon_size] = res_vicon_avz

                        # Append zed odometry incrementally
                        if res_zed:
                            (res_zed_t, res_zed_px, res_zed_py, res_zed_pz, res_zed_qx, res_zed_qy, res_zed_qz, res_zed_qw,
                             res_zed_vx, res_zed_vy, res_zed_vz, res_zed_avx, res_zed_avy, res_zed_avz) = res_zed

                            current_zed_size = zed_grp['t'].shape[0]
                            new_zed_size = current_zed_size + len(res_zed_t)

                            zed_grp['t'].resize((new_zed_size,))
                            zed_grp['px'].resize((new_zed_size,))
                            zed_grp['py'].resize((new_zed_size,))
                            zed_grp['pz'].resize((new_zed_size,))
                            zed_grp['qx'].resize((new_zed_size,))
                            zed_grp['qy'].resize((new_zed_size,))
                            zed_grp['qz'].resize((new_zed_size,))
                            zed_grp['qw'].resize((new_zed_size,))
                            zed_grp['vx'].resize((new_zed_size,))
                            zed_grp['vy'].resize((new_zed_size,))
                            zed_grp['vz'].resize((new_zed_size,))
                            zed_grp['avx'].resize((new_zed_size,))
                            zed_grp['avy'].resize((new_zed_size,))
                            zed_grp['avz'].resize((new_zed_size,))

                            zed_grp['t'][current_zed_size:new_zed_size] = res_zed_t
                            zed_grp['px'][current_zed_size:new_zed_size] = res_zed_px
                            zed_grp['py'][current_zed_size:new_zed_size] = res_zed_py
                            zed_grp['pz'][current_zed_size:new_zed_size] = res_zed_pz
                            zed_grp['qx'][current_zed_size:new_zed_size] = res_zed_qx
                            zed_grp['qy'][current_zed_size:new_zed_size] = res_zed_qy
                            zed_grp['qz'][current_zed_size:new_zed_size] = res_zed_qz
                            zed_grp['qw'][current_zed_size:new_zed_size] = res_zed_qw
                            zed_grp['vx'][current_zed_size:new_zed_size] = res_zed_vx
                            zed_grp['vy'][current_zed_size:new_zed_size] = res_zed_vy
                            zed_grp['vz'][current_zed_size:new_zed_size] = res_zed_vz
                            zed_grp['avx'][current_zed_size:new_zed_size] = res_zed_avx
                            zed_grp['avy'][current_zed_size:new_zed_size] = res_zed_avy
                            zed_grp['avz'][current_zed_size:new_zed_size] = res_zed_avz

                        # Flush to disk periodically instead of every chunk
                        chunk_index += 1
                        if (chunk_index) % flush_interval == 0:
                            h5f.flush()
                            print(f"Written {chunk_index} chunks to disk (flushed)")

                    # free the futures list and continue streaming
                    futures = []
                    continue

            # process any remaining futures after stream end
            for future in futures:
                res = future.result()

                # Append neuro images: decode bytes -> numpy arrays -> store in dataset images (N,H,W,C) and timestamps
                def _channels_from_encoding(enc, h, w, data_len):
                    enc = (enc or '').lower()
                    if 'rgb' in enc and 'rgba' not in enc:
                        return 3
                    if 'bgr' in enc and 'bgra' not in enc:
                        return 3
                    if 'rgba' in enc or 'bgra' in enc:
                        return 4
                    if 'mono' in enc or 'gray' in enc:
                        return 1
                    # fallback: infer
                    if data_len == h * w:
                        return 1
                    if data_len == h * w * 3:
                        return 3
                    if data_len == h * w * 4:
                        return 4
                    return 3

                def _resize_nearest(img, th, tw):
                    h, w = img.shape[0], img.shape[1]
                    if h == th and w == tw:
                        return img
                    row_idx = np.linspace(0, h - 1, th).astype(np.intp)
                    col_idx = np.linspace(0, w - 1, tw).astype(np.intp)
                    return img[row_idx[:, None], col_idx]

                if res_images_neuro_t:
                    # decode images into list of numpy arrays
                    neuro_imgs = []
                    for h, w, enc, data in zip(res_images_neuro_h, res_images_neuro_w, res_images_neuro_encoding, res_images_neuro_data):
                        arr = np.frombuffer(data, dtype=np.uint8)
                        ch = _channels_from_encoding(enc, h, w, arr.size)
                        arr = arr.reshape((h, w, ch)) if ch > 1 else arr.reshape((h, w))
                        if ch == 1:
                            img = np.expand_dims(arr, axis=2)
                        else:
                            img = arr
                        # convert BGR/BGRA -> RGB/RGBA if needed (handle alpha properly)
                        enc_l = (enc or '').lower()
                        if 'bgra' in enc_l:
                            # BGRA -> RGBA
                            img = img[..., [2, 1, 0, 3]]
                        elif 'bgr' in enc_l:
                            # BGR -> RGB
                            img = img[..., ::-1]
                        neuro_imgs.append(img.astype(np.uint8))

                    # create dataset lazily based on first image shape (store native camera resolution)
                    if neuro_imgs:
                        first = neuro_imgs[0]
                        th, tw, tc = first.shape
                        if 'images' not in neuro_grp:
                            neuro_grp.create_dataset('images', shape=(0, th, tw, tc), maxshape=(None, th, tw, tc), dtype='u1', chunks=(1, th, tw, tc), compression='gzip', compression_opts=1)
                            neuro_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', chunks=(chunk_size_events,), compression='gzip', compression_opts=1)

                        cur = neuro_grp['images'].shape[0]
                        new = cur + len(neuro_imgs)
                        neuro_grp['images'].resize((new, th, tw, tc))
                        neuro_grp['t'].resize((new,))

                        for idx, img in enumerate(neuro_imgs):
                            if img.shape != (th, tw, tc):
                                img = _resize_nearest(img, th, tw)
                            neuro_grp['images'][cur + idx, ...] = img
                        neuro_grp['t'][cur:new] = res_images_neuro_t

                # Append flow images similarly
                if res_images_flow_t:
                    flow_imgs = []
                    for h, w, enc, data in zip(res_images_flow_h, res_images_flow_w, res_images_flow_encoding, res_images_flow_data):
                        arr = np.frombuffer(data, dtype=np.uint8)
                        ch = _channels_from_encoding(enc, h, w, arr.size)
                        arr = arr.reshape((h, w, ch)) if ch > 1 else arr.reshape((h, w))
                        if ch == 1:
                            img = np.expand_dims(arr, axis=2)
                        else:
                            img = arr
                        # convert BGR/BGRA -> RGB/RGBA if needed (handle alpha properly)
                        enc_l = (enc or '').lower()
                        if 'bgra' in enc_l:
                            img = img[..., [2, 1, 0, 3]]
                        elif 'bgr' in enc_l:
                            img = img[..., ::-1]
                        flow_imgs.append(img.astype(np.uint8))

                    if flow_imgs:
                        first = flow_imgs[0]
                        th, tw, tc = first.shape
                        # store at native camera resolution
                        if 'images' not in flow_grp:
                            flow_grp.create_dataset('images', shape=(0, th, tw, tc), maxshape=(None, th, tw, tc), dtype='u1', chunks=(1, th, tw, tc), compression='gzip', compression_opts=1)
                            flow_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', chunks=(chunk_size_events,), compression='gzip', compression_opts=1)

                        cur = flow_grp['images'].shape[0]
                        new = cur + len(flow_imgs)
                        flow_grp['images'].resize((new, th, tw, tc))
                        flow_grp['t'].resize((new,))

                        for idx, img in enumerate(flow_imgs):
                            if img.shape != (th, tw, tc):
                                img = _resize_nearest(img, th, tw)
                            flow_grp['images'][cur + idx, ...] = img
                        flow_grp['t'][cur:new] = res_images_flow_t

                # Append flow raw Float32MultiArray entries
                res_flow_raw_t = res_flow_raw_shapes = res_flow_raw_data = None
                if 'flow_raw' in res:
                    (res_flow_raw_t, res_flow_raw_shapes, res_flow_raw_data) = res['flow_raw']

                if res_flow_raw_t:
                    # create datasets lazily
                    if 'data' not in flow_raw_grp:
                        flow_raw_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='f8', chunks=(chunk_size_events,), compression='gzip', compression_opts=1)
                        flow_raw_grp.create_dataset('data', shape=(0,), maxshape=(None,), dtype=vlen_float)
                        flow_raw_grp.create_dataset('shape', shape=(0,), maxshape=(None,), dtype=vlen_int)

                    cur = flow_raw_grp['t'].shape[0]
                    new = cur + len(res_flow_raw_t)
                    flow_raw_grp['t'].resize((new,))
                    flow_raw_grp['data'].resize((new,))
                    flow_raw_grp['shape'].resize((new,))

                    # convert bytes back to numpy arrays for vlen storage
                    data_list = [np.frombuffer(b, dtype=np.float32) for b in res_flow_raw_data]
                    shape_list = [np.array(s, dtype=np.int32) for s in res_flow_raw_shapes]

                    flow_raw_grp['data'][cur:new] = data_list
                    flow_raw_grp['shape'][cur:new] = shape_list
                    flow_raw_grp['t'][cur:new] = res_flow_raw_t

                # increment processed chunk counter and optionally flush
                chunk_index += 1
                if (chunk_index) % flush_interval == 0:
                    h5f.flush()
                    print(f"Written {chunk_index} chunks to disk (flushed)")

    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ROS2 MCAP bag to HDF5 with parallel processing.")
    parser.add_argument("--input", help="Path to input ROS2 MCAP bag")
    parser.add_argument("--output", default='output.h5', help="Path to output HDF5 file")
    parser.add_argument("--chunk-size", type=int, default=200, help="Number of messages per chunk (default: 100)")
    parser.add_argument("--max-cores", type=int, default=os.cpu_count()-2, help="Maximum number of CPU cores to use")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    chunk_size = args.chunk_size
    max_cores = args.max_cores

    print(f"Using up to {max_cores} CPU cores for processing")

    parallel_parse_bag(input_path, output_path, chunk_size, max_cores)
