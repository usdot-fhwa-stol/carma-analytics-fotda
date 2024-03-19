import argparse
import csv
import math
from pathlib import Path

import more_itertools
import rosbag


def find_msg_closest_to_timestamp(bag_msgs, timestamp):
    for current, next_ in more_itertools.pairwise(bag_msgs):
        # Each element in `bag_msgs` is a triple:
        # <topic_name, msg, system_time_at_receive>
        _, _, current_log_timestamp = current
        _, _, next_log_timestamp = next_

        # We only need to iterate until we find two messages who's logged
        # timestamps straddle the timestamp of interest. All messages
        # afterwards will have increasing differences.
        if next_log_timestamp > timestamp:
            if abs(current_log_timestamp - timestamp) < abs(
                timestamp - next_log_timestamp
            ):
                return current

            return next_


# calculate the offset between ros clock and the sim clock
def get_time_offset(ros_bag_file):
    with rosbag.Bag(ros_bag_file, "r") as bag:
        cdasim_clock_msgs = bag.read_messages(topics=["/sim_clock"])  # CDASim clock
        carla_clock_msgs = bag.read_messages(topics=["/clock"])  # CARLA clock

        _, cdasim_clock_msg, cdasim_clock_log_time = next(cdasim_clock_msgs)

        carla_clock_at_cdasim_start = find_msg_closest_to_timestamp(
            carla_clock_msgs, cdasim_clock_log_time
        )

        return carla_clock_at_cdasim_start[1].clock - cdasim_clock_msg.clock


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
            ["Message Time (ms)", "Object ID", "Map Position X (m)", "Map Position Y (m)"]
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


def get_carla_object_odometry(actor_id, ros_bag_file, output_file, time_offset):
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
                "Message Time (ms)",
                "Map Position X (m)",
                "Map Position Y (m)",
                "Map Velocity X (mps)",
                "Map Velocity Y (mps)",
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


def get_vehicle_odometry(ros_bag_file, output_file, time_offset):
    with rosbag.Bag(ros_bag_file, "r") as bag:
        messages = [
            msg for (_, msg, _) in bag.read_messages(topics=["/carla/carma_1/odometry"])
        ]

    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Message Time (ms)",
                "Map Position X (m)",
                "Map Position Y (m)",
                "Body Twist Longitudinal (mps)",
            ]
        )

        for message in messages:
            cdasim_time_ms = (message.header.stamp - time_offset).to_sec() * 1_000
            writer.writerow(
                [
                    cdasim_time_ms,
                    message.pose.pose.position.x,
                    message.pose.pose.position.y,
                    message.twist.twist.linear.x,
                ]
            )


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

    get_carla_object_odometry(
        221, args.ros_bag_file, args.csv_dir / "pedestrian_odometry.csv", time_offset
    )

    get_vehicle_odometry(
        args.ros_bag_file, args.csv_dir / "vehicle_odometry.csv", time_offset
    )


if __name__ == "__main__":
    main()
