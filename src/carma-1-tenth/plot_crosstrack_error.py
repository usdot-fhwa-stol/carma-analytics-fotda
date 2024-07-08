# Plot the crosstrack error as a function of downtrack (distance traveled along the route)


from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import argparse, argcomplete
import os


def find_closest_point(point_arr, point, trim_ends=True):
    difference_arr = np.linalg.norm(point_arr - point, axis=1)
    min_index = difference_arr.argmin()
    # Don't want to include deviations if we have not yet reached the route or have completed it
    if trim_ends and (min_index == 0 or min_index == len(difference_arr) - 1):
        return None, None
    return point_arr[min_index], min_index


def is_left(route_a, route_b, odometry):
    cross_product = np.cross(route_b - route_a, odometry - route_a)
    if cross_product > 0:
        return True
    else:
        return False


def plot_crosstrack_error(bag_dir, show_plots=True):
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
    route_coordinates = np.array(route_coordinates)
    route_coordinates_with_distance = np.zeros((len(route_coordinates), 3))
    route_coordinates_with_distance[:, 1:] = route_coordinates
    running_distance = 0.0
    # Assign a distance along the route for each x,y coordinate along the route
    for i in range(len(route_coordinates)):
        if i > 0:
            running_distance += np.linalg.norm(route_coordinates_with_distance[i, 1:] - route_coordinates_with_distance[i-1, 1:])
        route_coordinates_with_distance[i,0] = running_distance
    # Fit 2D splines that map the distance along the route to the x and y coordinates to upsample the points
    x_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,1], k=1)
    y_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,2], k=1)
    samples = np.linspace(route_coordinates_with_distance[0,0], route_coordinates_with_distance[-1,0], 10000)
    route_x_points = x_spline(samples)
    route_y_points = y_spline(samples)
    route_deviations = []
    distances_along_route = []
    # For each odometry message, compute the deviation from the closest point along the route
    for i in range(1, len(odometry)):
        closest_point, closest_index = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            if is_left(np.array([route_x_points[closest_index - 5], route_y_points[closest_index - 5]]), np.array([route_x_points[closest_index + 5], route_y_points[closest_index + 5]]), odometry[i]):
                route_deviations.append(np.linalg.norm(odometry[i] - closest_point))
            else:
                route_deviations.append(-np.linalg.norm(odometry[i] - closest_point))
            if len(distances_along_route):
                distances_along_route.append(distances_along_route[-1] + np.linalg.norm(closest_point - previous_closest_point))
            else:
                distances_along_route.append(np.linalg.norm(closest_point - np.array([route_x_points[0], route_y_points[0]])))
        previous_closest_point = closest_point


    print("Average Deviation:", np.mean(np.abs(route_deviations)))
    print("Maximum Deviation:", np.max(np.abs(route_deviations)))
    if show_plots:
        plt.plot(distances_along_route, route_deviations, label="Crosstrack Error")
        plt.plot(distances_along_route, np.zeros(len(route_deviations)), label="Route")
    return distances_along_route, route_deviations


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot deviation between C1T path driven and desired route")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_crosstrack_error(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.xlabel("Downtrack (m)")
    plt.ylabel("Crosstrack Error (m)")
    plt.title("Crosstrack Error vs. Downtrack")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.legend()
    plt.show()
