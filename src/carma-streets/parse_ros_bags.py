import argparse
import csv
from pathlib import Path

import more_itertools
import rosbag


def find_msg_closest_to_timestamp(bag_msgs, timestamp):
    for current, next_ in more_itertools.pairwise(bag_msgs):
        if next_[2] > timestamp:
            if abs(timestamp - current[2]) < abs(next_[2] - timestamp):
                return current

            return next_


# calculate the offset between ros clock and the sim clock
def get_time_offset(ros_bag_file):
    with rosbag.Bag(ros_bag_file, "r") as bag:
        cdasim_clock_msgs = bag.read_messages(topics=["/sim_clock"])  # CDASim clock
        carla_clock_msgs = bag.read_messages(topics=["/clock"])  # CARLA clock

        carla_clock_at_cdasim_start = find_msg_closest_to_timestamp(
            carla_clock_msgs, next(cdasim_clock_msgs)[2]
        )

        return carla_clock_at_cdasim_start[1].clock - next(cdasim_clock_msgs)[1].clock


def get_detected_objects(ros_bag_file, output_file, time_offset):
    detected_obj_topic_name = "/environment/fused_external_objects"
    messages = []
    with rosbag.Bag(ros_bag_file, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[detected_obj_topic_name]):
            messages.extend(msg.objects)

    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["cdasim_time_ms", "object_id", "map_position_x_m", "map_position_y_m"]
        )
        for message in messages:
            cdasim_time_ms = (message.header.stamp - time_offset).to_sec() * 1_000
            writer.writerow(
                [
                    cdasim_time_ms,
                    message.id % 1_000,
                    message.pose.pose.position.x,
                    message.pose.pose.position.y,
                ]
            )

    print(
        f"Sorted data from topic '{detected_obj_topic_name}' saved to '{output_file}'"
    )
    print(f"Calculated sim time offset: '{time_offset}'")


def main():
    parser = argparse.ArgumentParser(
        description="Script to parse ROS Bags into CSV data"
    )
    parser.add_argument(
        "--ros-bag-file", help="ROS Bags File.", type=str, required=True
    )
    parser.add_argument(
        "--csv-dir", help="Directory to write csv file to.", type=Path, required=True
    )
    args = parser.parse_args()

    time_offset = get_time_offset(args.ros_bag_file)
    get_detected_objects(
        args.ros_bag_file, args.csv_dir / "vehicle_detected_objects.csv", time_offset
    )


if __name__ == "__main__":
    main()
