# Plot the standard deviation of the vehicle's localization as a function of downtrack distance


from rosbag_utils import open_bagfile
from plot_crosstrack_error import find_closest_point
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import argparse, argcomplete
import os


def plot_localization_std_deviation(bag_dir, start_offset=0.0):
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
            if topic == route_topic:
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
    samples = np.linspace(route_coordinates_with_distance[0,0], route_coordinates_with_distance[-1,0], 10000)
    route_x_points = x_spline(samples)
    route_y_points = y_spline(samples)
    odom_std_deviations = []
    distances_along_route = []
    # For each odometry message, compute the deviation from the closest point along the route
    for i in range(5, len(odometry) - 5):
        closest_point, _ = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            odom_std_deviations.append(np.std(odometry[i-1:i+1], axis=0))
            if len(distances_along_route):
                distances_along_route.append(distances_along_route[-1] + np.linalg.norm(closest_point - previous_closest_point))
            else:
                distances_along_route.append(np.linalg.norm(closest_point - np.array([route_x_points[0], route_y_points[0]])))
        previous_closest_point = closest_point
    odom_std_deviations = np.array(odom_std_deviations).T


    print("Average Standard Deviation (x):", np.mean(odom_std_deviations[0]))
    print("Maximum Standard Deviation (x):", np.max(odom_std_deviations[0]))
    print("Average Standard Deviation (y):", np.mean(odom_std_deviations[1]))
    print("Maximum Standard Deviation (y):", np.max(odom_std_deviations[1]))

    plt.plot(distances_along_route, odom_std_deviations[0], label="x")
    plt.plot(distances_along_route, odom_std_deviations[1], label="y")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot standard deviation of the vehicle's localization as a function of downtrack distance")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_localization_std_deviation(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.xlabel("Downtrack (m)")
    plt.ylabel("Localization Standard Deviation (m)")
    plt.title("Localization Standard Deviation vs. Downtrack")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.legend()
    plt.show()
