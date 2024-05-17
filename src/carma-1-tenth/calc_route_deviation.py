import sys
import rosbag2_py
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import datetime as dt
from scipy.interpolate import make_interp_spline
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

def find_closest_point(point_arr, point):
    difference_arr = np.linalg.norm(point_arr - point, axis=1)
    min_index = difference_arr.argmin()
    if min_index == 0 or min_index == len(difference_arr) - 1:   # don't want to count deviations if we have not yet reached the route or completed it
        return None
    return point_arr[min_index]

def calc_route_deviation(bag_dir, start_offset=0.0):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    topic_types, type_map, topic_metadata_map, reader = open_bagfile(bag_dir, storage_id=storage_id)
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    topic_counts = np.array( list(topic_count_dict.values()) ) 
    total_msgs = np.sum( topic_counts )
    msg_dict = {key : [] for key in topic_count_dict.keys()}
    filt = rosbag2_py.StorageFilter([odom_topic, route_topic])
    reader.set_filter(filt)
    route_graph = None
    odom_count = 0
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    for idx in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if msg_type == "visualization_msgs/msg/MarkerArray":
                route_graph = msg
            else:
                odometry[odom_count] = [-msg.pose.pose.position.y, msg.pose.pose.position.x]
                odom_count += 1
    route_coordinates = []
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            route_coordinates.append([-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x])
    route_coordinates_with_distance = np.zeros((len(route_coordinates), 3))
    route_coordinates_with_distance[:, 1:] = np.array(route_coordinates)
    running_distance = 0.0
    for i in range(len(route_coordinates)):
        if i > 0:
            running_distance += np.linalg.norm(route_coordinates_with_distance[i, 1:] - route_coordinates_with_distance[i-1, 1:])
        route_coordinates_with_distance[i,0] = running_distance
    x_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,1])
    y_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,2])
    samples = np.linspace(route_coordinates_with_distance[0,0], route_coordinates_with_distance[-1,0], 5000)
    route_x_points = x_spline(samples)
    route_y_points = y_spline(samples)
    route_deviations = []
    odom_movements = []
    for i in range(1, len(odometry)):
        closest_point = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            route_deviations.append(np.linalg.norm(odometry[i] - closest_point))
            odom_movements.append(np.linalg.norm(odometry[i]- odometry[i-1]))
            
    print("Average Deviation:", np.mean(route_deviations))
    print("Maximum Deviation:", np.max(route_deviations))
    plt.plot(range(len(route_deviations)), route_deviations)
    plt.plot(range(len(odom_movements)), odom_movements)

if __name__=="__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Plot steering rate of C1T trucks")
    parser.add_argument("bag_in", type=str, help="Bag to load")
    parser.add_argument("--png_out", type=str, help="Output file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    calc_route_deviation(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.xlabel("Index")
    plt.ylabel("Deviation from Route (m)")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
