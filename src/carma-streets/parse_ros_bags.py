import argparse
import csv
import math
from pathlib import Path

import rosbag


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
        writer = csv.writer(file)
        writer.writerow(['Timestamp (ms)', 'ObjID', 'Positionx',  'Positiony'])
        for message in messages:
            timestamp_ms = message.header.stamp.secs * 1000 + message.header.stamp.nsecs // 1000000 - time_offset
            writer.writerow([timestamp_ms, message.id%1000, message.pose.pose.position.x, message.pose.pose.position.y])

    print(f"Sorted data from topic '{detected_obj_topic_name}' saved to '{outputfile}'")
    print(f"Calculated sim time offset: '{time_offset}'")


def get_object_odometry(actor_id, ros_bag_file, output_file, time_offset):
    def get_object_with_id(id_, msg):
        for object_ in msg.objects:
            if object_.id == id_:
                return object_

    with rosbag.Bag(ros_bag_file, "r") as bag:
        messages = [msg for (_, msg, _) in bag.read_messages(topics=["/carla/objects"])]

    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "cdasim_time_ms",
                "map_position_x_m",
                "map_position_y_m",
                "map_velocity_x_mps",
                "map_velocity_y_mps",
            ]
        )

        for message in messages:
            pedestrian_odometry = get_object_with_id(actor_id, message)

            if not pedestrian_odometry or math.isclose(
                pedestrian_odometry.twist.linear.x, 0.0
            ):
                continue

            cdasim_time_ms = (message.header.stamp - time_offset).to_sec() * 1_000
            writer.writerow(
                [
                    cdasim_time_ms,
                    pedestrian_odometry.pose.position.x,
                    pedestrian_odometry.pose.position.y,
                    pedestrian_odometry.twist.linear.x,
                    pedestrian_odometry.twist.linear.y,
                ]
            )


def main():
    parser = argparse.ArgumentParser(description='Script to parse ROS Bags into CSV data')
    parser.add_argument('--ros-bag-file', help='ROS Bags File.', type=str, required=True)
    parser.add_argument('--csv-dir', help='Directory to write csv file to.', type=Path, required=True)
    args = parser.parse_args()

    time_offset = get_time_offset(args.ros_bag_file)
    get_detected_objects(args.ros_bag_file, args.csv_dir/'vehicle_detected_objects.csv', time_offset)

    get_object_odometry(221,
        args.ros_bag_file, args.csv_dir / "pedestrian_odometry.csv", time_offset
    )


if __name__ == '__main__':
    main()
