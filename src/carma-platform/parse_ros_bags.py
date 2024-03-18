import argparse
import csv
import math
from pathlib import Path

import more_itertools
import rosbag


def find_msg_closest_to_timestamp(bag_msgs, timestamp):
    for current, next_ in more_itertools.pairwise(bag_msgs):
        # Each element in `bag_msgs` is a triple:
        # <topic_name, msg, rosbag_t_received>
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

def get_object_msgs_with_its_system_times(ros_bag_file, topic_name):

    messages = []
    with rosbag.Bag(ros_bag_file, "r") as bag:
        for _, msg, rosbag_t_received in bag.read_messages(topics=[topic_name]):
            messages.append((msg.objects, rosbag_t_received.to_sec())) #nano to sec
    return messages

def get_objects_from_incoming_sdsm(ros_bag_file, output_file, time_offset):
    incoming_msgs_topic_name = "/message/incoming_sdsm"

    sdsm_msgs = get_object_msgs_with_its_system_times(ros_bag_file, incoming_msgs_topic_name)

    print(f"Detected sdsm_msgs size {len(sdsm_msgs)}" )
    #print(sdsm_msgs[0])
    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    sim_times = []
    with rosbag.Bag(ros_bag_file, "r") as bag:
        for _, msg, t in bag.read_messages(topics=["/sim_clock"]):
            sim_times.append((t, msg.clock)) # Assuming 'clock' attribute holds simulation time in seconds

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["cdasim_time_ms", "rosbag_t_received_s", "object_id"]
        )

        last_idx = 0
        for message, rosbag_t_received_s in sdsm_msgs:

            for t, sim_time in sim_times[last_idx:]:
                if (t.to_sec() > rosbag_t_received_s):
                    break
                last_idx += 1

            if (last_idx >= len(sim_times)):
                break

            cdasim_time_ms = math.floor(sim_times[last_idx][1].to_sec() * 1000)
            for object_data in message.detected_object_data:

                writer.writerow(
                    [
                        cdasim_time_ms,
                        rosbag_t_received_s,
                        object_data.detected_object_common_data.detected_id.object_id,
                    ]
                )

    print(
        f"Sorted data from topic '{incoming_msgs_topic_name}' saved to '{output_file}'"
    )
    print(f"Calculated sim time offset: '{time_offset}'")

def get_detected_objects(ros_bag_file, output_file, time_offset):

    detected_obj_topic_name = "/environment/fused_external_objects"

    messages = get_object_msgs_with_its_system_times(ros_bag_file, detected_obj_topic_name)

    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["cdasim_time_ms", "rosbag_t_received_s", "object_id", "map_position_x_m", "map_position_y_m"]
        )
        for message, rosbag_t_received_s in messages:
            for obj in message:
                cdasim_time_ms = math.floor((obj.header.stamp - time_offset).to_sec() * 1_000)
                writer.writerow(
                    [
                        cdasim_time_ms,
                        rosbag_t_received_s,
                        obj.id % 1_000,
                        obj.pose.pose.position.x,
                        obj.pose.pose.position.y,
                    ]
                )

    print(
        f"Sorted data from topic '{detected_obj_topic_name}' saved to '{output_file}'"
    )
    print(f"Calculated sim time offset: '{time_offset}'")

def get_detected_objects_with_sim_received_time(ros_bag_file, output_file, time_offset):

    detected_obj_topic_name = "/environment/fused_external_objects"

    messages = get_object_msgs_with_its_system_times(ros_bag_file, detected_obj_topic_name)

    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Output file {output_file} already exists. Overwriting file.")

    sim_times = []
    with rosbag.Bag(ros_bag_file, "r") as bag:
        for _, msg, t in bag.read_messages(topics=["/sim_clock"]):
            sim_times.append((t, msg.clock)) # Assuming 'clock' attribute holds simulation time in seconds

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["cdasim_time_ms", "rosbag_t_received_s", "object_id", "map_position_x_m", "map_position_y_m"]
        )

        last_idx = 0
        for message, rosbag_t_received_s in messages:
            for t, sim_time in sim_times[last_idx:]:
                if (t.to_sec() > rosbag_t_received_s):
                    break
                last_idx += 1

            if (last_idx >= len(sim_times)):
                break

            cdasim_time_ms = math.floor(sim_times[last_idx][1].to_sec() * 1000)
            for obj in message:
                writer.writerow(
                    [
                        cdasim_time_ms,
                        rosbag_t_received_s,
                        obj.id % 1_000,
                        obj.pose.pose.position.x,
                        obj.pose.pose.position.y,
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

            cdasim_time_ms = math.floor((message.header.stamp - time_offset).to_sec() * 1_000)
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
            cdasim_time_ms = math.floor((message.header.stamp - time_offset).to_sec() * 1_000)
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
    get_detected_objects_with_sim_received_time(
        args.ros_bag_file, args.csv_dir / "vehicle_detected_objects.csv", time_offset
    )

    get_objects_from_incoming_sdsm(
        args.ros_bag_file, args.csv_dir / "objects_from_incoming_sdsm.csv", time_offset
    )

#    get_carla_object_odometry(
#        221, args.ros_bag_file, args.csv_dir / "pedestrian_odometry.csv", time_offset
#    )
#
#    get_vehicle_odometry(
#        args.ros_bag_file, args.csv_dir / "vehicle_odometry.csv", time_offset
#    )
#

if __name__ == "__main__":
    main()
