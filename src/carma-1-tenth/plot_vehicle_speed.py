# Plot the target and actual vehicle speeds


from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
import matplotlib.pyplot as plt
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import datetime
import argparse, argcomplete
import os


def plot_vehicle_speed(bag_dir, show_plots=True):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    target_topic = '/ackermann_cmd'
    vel_topic = '/odom'
    # Open bag
    reader, type_map = open_bagfile(bag_dir, topics=[target_topic, vel_topic], storage_id=storage_id)
    # Gather number of messages on each topic
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    # Variables to store data from bag
    vel_count = 0
    current_target_velocity = 0.0
    velocities = np.zeros((topic_count_dict[vel_topic],))
    target_velocities = np.zeros((topic_count_dict[vel_topic],))
    velocity_times = np.zeros((topic_count_dict[vel_topic],))
    # Iterate through bag
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[target_topic] + topic_count_dict[vel_topic])):
        if(reader.has_next()):
            (topic, data, timestamp) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == target_topic:
                # Store current target velocity
                current_target_velocity = msg.drive.speed
            elif topic == vel_topic:
                # Store measured velocity and target velocity at that time
                velocities[vel_count] = msg.twist.twist.linear.x
                target_velocities[vel_count] = current_target_velocity
                velocity_times[vel_count] = timestamp
                vel_count += 1
    # Convert timestamps to seconds from start of the bag
    velocity_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in velocity_times]
    velocity_time_seconds = [(date - velocity_datetimes[0]).total_seconds() for date in velocity_datetimes]
    plt.plot(velocity_time_seconds, target_velocities, '-', label="Target Velocity")
    plt.plot(velocity_time_seconds, velocities, '--', label="Measured Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    if show_plots:
        plt.show()
    return velocities, target_velocities


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot the target and actual vehicle speeds")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    velocity_deviations, min_target_velocity = plot_vehicle_speed(os.path.normpath(os.path.abspath(argdict["bag_in"])))
