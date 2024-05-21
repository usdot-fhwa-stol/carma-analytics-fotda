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


def plot_route_driven(bag_dir):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    _, type_map, _, reader = open_bagfile(bag_dir, storage_id=storage_id)
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    filt = rosbag2_py.StorageFilter([odom_topic, route_topic])
    reader.set_filter(filt)
    route_graph = None
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    odom_count = 0
    for idx in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if msg_type == "visualization_msgs/msg/MarkerArray":
                route_graph = msg
            else:
                odometry[odom_count] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
                odom_count += 1
    plt.plot(-odometry[:,1], odometry[:,0])
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            plt.plot(-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x, 'ro')
            x_min = np.min([-route_graph.markers[i].pose.position.y, x_min])
            y_min = np.min([route_graph.markers[i].pose.position.x, y_min])
            x_max = np.max([-route_graph.markers[i].pose.position.y, x_max])
            y_max = np.max([route_graph.markers[i].pose.position.x, y_max])
    plt.xlim([x_min - 1.0, x_max + 1.0])
    plt.ylim([y_min - 1.0, y_max + 1.0])


if __name__=="__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Plot steering rate of C1T trucks")
    parser.add_argument("bag_in", type=str, help="Bag to load")
    parser.add_argument("--png_out", type=str, help="Output file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_route_driven(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.xlabel("Horizontal Coordinate (m)")
    plt.ylabel("Vertical Coordinate (m)")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
