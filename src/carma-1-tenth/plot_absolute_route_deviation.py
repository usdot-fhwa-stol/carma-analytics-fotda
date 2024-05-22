from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import datetime as dt
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os


def find_closest_point(point_arr, point):
    difference_arr = np.linalg.norm(point_arr - point, axis=1)
    min_index = difference_arr.argmin()
    # Don't want to include deviations if we have not yet reached the route or have completed it
    if min_index == 0 or min_index == len(difference_arr) - 1:
        return None
    return point_arr[min_index]


def plot_absolute_route_deviation(bag_dir, start_offset=0.0):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic], storage_id=storage_id)
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}

    route_graph = None
    odom_count = 0
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    odometry_times = np.zeros((topic_count_dict[odom_topic],))
    # Iterate through bag and store odometry + route_graph messages
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
                odometry_times[odom_count] = t_
                odom_count += 1
    route_coordinates = []
    # Rotate the route_graph coordinates 90 degrees to match C1T coordinates (x-forward, y-left)
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            route_coordinates.append([-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x])
    route_coordinates_with_distance = np.zeros((len(route_coordinates), 3))
    route_coordinates_with_distance[:, 1:] = np.array(route_coordinates)
    running_distance = 0.0
    # Assign a distance along the route for each x,y coordinate along the route
    for i in range(len(route_coordinates)):
        if i > 0:
            running_distance += np.linalg.norm(route_coordinates_with_distance[i, 1:] - route_coordinates_with_distance[i-1, 1:])
        route_coordinates_with_distance[i,0] = running_distance
    # Fit 2D splines that map the distance along the route to the x and y coordinates to upsample the points
    x_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,1])
    y_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,2])
    samples = np.linspace(route_coordinates_with_distance[0,0], route_coordinates_with_distance[-1,0], 5000)
    route_x_points = x_spline(samples)
    route_y_points = y_spline(samples)
    route_deviations = []
    route_times = []
    # For each odometry message, compute the deviation from the closest point along the route
    for i in range(1, len(odometry)):
        closest_point = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            route_deviations.append(np.linalg.norm(odometry[i] - closest_point))
            route_times.append(odometry_times[i])

    dates = np.array([dt.datetime.fromtimestamp(ts * 1e-9) for ts in route_times])
    start_time = dates[0]
    times = np.array([(date - start_time).total_seconds() - start_offset for date in dates])

    print("Average Deviation:", np.mean(route_deviations))
    print("Maximum Deviation:", np.max(route_deviations))
    plt.plot(times, route_deviations)


if __name__=="__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Plot steering rate of C1T trucks")
    parser.add_argument("bag_in", type=str, help="Bag to load")
    parser.add_argument("--png_out", type=str, help="Output file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_absolute_route_deviation(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.xlabel("Time (s)")
    plt.ylabel("Deviation from Route (m)")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
