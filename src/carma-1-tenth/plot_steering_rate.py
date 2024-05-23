# Plot the steering rate velocity over time, with the ability to compare between two runs

from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import datetime as dt
import os


# from ackermann_to_vesc_node parameters in c1t_bringup/params/params.yaml
STEERING_TO_SERVO_OFFSET = 0.425
STEERING_TO_SERVO_GAIN = -0.55


def plot_steering_rate(bag_dir, label, start_offset=0.0):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    servo_topic = '/commands/servo/position'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    reader, type_map = open_bagfile(bag_dir, topics=[servo_topic], storage_id=storage_id)
    timestamps = np.zeros(topic_count_dict[servo_topic],)
    steering_angles = np.zeros(topic_count_dict[servo_topic],)
    for idx in tqdm.tqdm(iterable=range(topic_count_dict[servo_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            timestamps[idx] = t_
            steering_angles[idx] = (msg.data - STEERING_TO_SERVO_OFFSET) / STEERING_TO_SERVO_GAIN
    dates = np.array([dt.datetime.fromtimestamp(ts * 1e-9) for ts in timestamps])
    start_time = dates[0]
    times = np.array([(date - start_time).total_seconds() - start_offset for date in dates])
    steering_rates = np.gradient(steering_angles, times)
    plt.plot(times, steering_rates, label=label)
    print("Standard Deviation:", np.std(steering_rates[:-5]))


if __name__=="__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Plot steering rate of C1T trucks")
    parser.add_argument("filtered_bag", type=str, help="Directory of bag with filtered steering")
    parser.add_argument("--filtered_bag_offset", type=float, default=0.0, help="Time offset for start of bag with filtered steering")
    parser.add_argument("--unfiltered_bag", type=str, help="Directory of bag with unfiltered steering")
    parser.add_argument("--unfiltered_bag_offset", type=float, default=0.0, help="Time offset for start of bag with unfiltered steering")
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_steering_rate(os.path.normpath(os.path.abspath(argdict["filtered_bag"])), "Filtered Steering", argdict["filtered_bag_offset"])
    if argdict["unfiltered_bag"]:
        plot_steering_rate(os.path.normpath(os.path.abspath(argdict["unfiltered_bag"])), "Unfiltered Steering", argdict["unfiltered_bag_offset"])
    plt.xlim([0,16])
    plt.ylim([-10, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (rad/s)")
    plt.title("Steering Rate vs. Time")
    plt.legend()
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
