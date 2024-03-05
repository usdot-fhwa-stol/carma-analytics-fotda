import rosbag
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# calculate the offset between ros ckock and the sim clock
def get_time_offset(ros_bag_file):
    time_offset = 0
    with rosbag.Bag(ros_bag_file, 'r') as bag:
        _, msg, t = next(bag.read_messages(topics=['/sim_clock']))
        time_offset = (t.secs * 1000 + t.nsecs // 1000000) - (msg.clock.secs * 1000 + msg.clock.nsecs // 1000000)
    return time_offset



def get_detected_objects(ros_bag_file, outputfile, time_offset):
    detected_obj_topic_name = '/environment/fused_external_objects'
    print(f'time_offset {time_offset}')
    messages = []
    with rosbag.Bag(ros_bag_file, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=[detected_obj_topic_name]):
            messages.extend(msg.objects)

    if outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwriting file.')
    with open(outputfile, 'w', newline='') as file:
        # check for duplicate
        writer = csv.writer(file)
        writer.writerow(['Timestamp (ms)', 'ObjID', 'Positionx',  'Positiony']) 
        for message in messages:
            timestamp_ms = message.header.stamp.secs * 1000 + message.header.stamp.nsecs // 1000000 - time_offset
            writer.writerow([timestamp_ms, message.id, message.pose.pose.position.x, message.pose.pose.position.y]) 

    print(f"Sorted data from topic '{detected_obj_topic_name}' saved to '{outputfile}'")
    print(f"Calculated sim time offset: '{time_offset}'")

def main():
    parser = argparse.ArgumentParser(description='Script to parse ROS Bags into CSV data')
    parser.add_argument('--ros-bag-file', help='ROS Bags File.', type=str, required=True) 
    # parser.add_argument('--csv-dir', help='Directory to write csv files to.', type=str, required=True)
    parser.add_argument('--csv-dir', help='Directory to write csv file to.', type=Path, required=True)  
    args = parser.parse_args()
    
    time_offset = get_time_offset(args.ros_bag_file)
    # csv_dir_path = Path(args.csv_dir)
    get_detected_objects(args.ros_bag_file, args.csv_dir/'vehicle_detected_objects.csv', time_offset)


if __name__ == '__main__':
    main()