import sys
import rosbag2_py
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import datetime as dt
import os

STEERING_TO_SERVO_OFFSET = 0.425
STEERING_TO_SERVO_GAIN = -0.55

def get_rosbag_options(path, serialization_format="cdr", storage_id="sqlite3"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

def open_bagfile(filepath: str, serialization_format="cdr", storage_id="sqlite3"):
    storage_options, converter_options = get_rosbag_options(filepath, serialization_format=serialization_format, storage_id=storage_id)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    # Create maps for quicker lookup
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    topic_metadata_map = {topic_types[i].name: topic_types[i] for i in range(len(topic_types))}
    return topic_types, type_map, topic_metadata_map, reader

def open_bagfile_writer(filepath: str, serialization_format="cdr", storage_id="sqlite3"):
    storage_options, converter_options = get_rosbag_options(filepath, serialization_format=serialization_format, storage_id=storage_id)

    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)
    return writer


def plot_steering_rate(bag_dir, label, start_offset=0.0):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    topic_types, type_map, topic_metadata_map, reader = open_bagfile(bag_dir, storage_id=storage_id)
    servo_topic = '/commands/servo/position'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    topic_counts = np.array( list(topic_count_dict.values()) ) 
    total_msgs = np.sum( topic_counts )
    msg_dict = {key : [] for key in topic_count_dict.keys()}
    filt = rosbag2_py.StorageFilter([servo_topic])
    reader.set_filter(filt)
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
    parser.add_argument("bag_in", type=str, help="Bag to load")
    parser.add_argument("--bag_in_2", type=str, help="Second bag to load")
    parser.add_argument("--png_out", type=str, help="Output file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_steering_rate(os.path.normpath(os.path.abspath(argdict["bag_in"])), "Unfiltered Steering")
    if argdict["bag_in_2"]:
        plot_steering_rate(os.path.normpath(os.path.abspath(argdict["bag_in_2"])), "Filtered Steering", 6.7)
    plt.xlim([0,16])
    plt.ylim([-10, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (rad/s)")
    plt.legend()
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()

