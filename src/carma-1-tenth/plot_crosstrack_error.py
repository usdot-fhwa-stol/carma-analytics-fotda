# Plot the crosstrack error as a function of downtrack (distance traveled along the route)


from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
from rosbag_utils import find_path_driven, find_closest_point
import argparse, argcomplete
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
import networkx as nx
import os


def is_left(route_a, route_b, odometry):
    cross_product = np.cross(route_b - route_a, odometry - route_a)
    if cross_product > 0:
        return True
    else:
        return False
    
def get_route_coordinates(route_messages, odometry):
    route_coordinates = []
    # Rotate the route_graph coordinates 90 degrees to match C1T coordinates (x-forward, y-left)
    if type(route_messages[0]) == MarkerArray:
        route_message = route_messages[0]
        nx_graph = nx.DiGraph()
        route_graph_coordinates = []
        # Create a networkx graph from the route graph marker array (nodes = 2, edges = 5)
        for i in range(len(route_message.markers)):
            if route_message.markers[i].type == 2:
                route_graph_coordinates.append([route_message.markers[i].id, -route_message.markers[i].pose.position.y, route_message.markers[i].pose.position.x])
                nx_graph.add_node(route_message.markers[i].id, pos=(-route_message.markers[i].pose.position.y, route_message.markers[i].pose.position.x))
        route_graph_coordinates = np.array(route_graph_coordinates)
        for i in range(len(route_message.markers)):
            if route_message.markers[i].type == 5:
                _, start_index = find_closest_point(route_graph_coordinates[:, 1:], [-route_message.markers[i].points[0].y, route_message.markers[i].points[0].x], trim_ends=False)
                _, end_index = find_closest_point(route_graph_coordinates[:, 1:], [-route_message.markers[i].points[1].y, route_message.markers[i].points[1].x], trim_ends=False)
                nx_graph.add_edge(route_graph_coordinates[start_index, 0], route_graph_coordinates[end_index, 0])
        # Find the list of nodes that were traversed
        route_coordinates_reached = find_path_driven(odometry, nx_graph)
        samples = np.linspace(0, 1, 100)
        for i in range(1, len(route_coordinates_reached)):
            points = (1 - samples)[:, np.newaxis] * route_coordinates_reached[i-1, 1:] + samples[:, np.newaxis] * route_coordinates_reached[i, 1:]
            route_coordinates += points.tolist()[1:-1]
    elif type(route_messages[0]) == Path:
        for route_message in route_messages:
            for i in range(len(route_message.poses)):
                route_coordinates.append([-route_message.poses[i].pose.position.y, route_message.poses[i].pose.position.x])
    return route_coordinates


def plot_crosstrack_error(bag_dir, route_topic, show_plots=True):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    # Odometry topic (map frame)
    odom_topic = '/amcl_pose'
    # Open bag
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic], storage_id=storage_id)
    # Gather number of messages on each topic
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}

    # Message received on route_topic
    route_messages = []
    # Count number of odom messages processed
    odom_count = 0
    # Stored odometry messages
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    # Odometry message times
    odometry_times = np.zeros((topic_count_dict[odom_topic],))
    # Iterate through bag and store odometry + route_graph messages
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic])):
        if(reader.has_next()):
            (topic, data, timestamp) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == route_topic:
                # Store route graph
                route_messages.append(msg)
            else:
                # Store odometry message and time
                odometry[odom_count] = [-msg.pose.pose.position.y, msg.pose.pose.position.x]
                odometry_times[odom_count] = timestamp
                odom_count += 1
    
    # Get the coordinates from route_graph
    route_coordinates = get_route_coordinates(route_messages, odometry)
    route_coordinates = np.array(route_coordinates)
    route_x_points, route_y_points = route_coordinates[:,0], route_coordinates[:,1]
    route_deviations = []
    running_distance = 0.0
    distances_along_route = []
    # For each odometry message, compute the deviation from the closest point along the route
    for i in range(1, len(odometry)):
        closest_point, closest_index = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            if is_left(np.array([route_x_points[closest_index - 1], route_y_points[closest_index - 1]]), np.array([route_x_points[closest_index + 1], route_y_points[closest_index + 1]]), odometry[i]):
                route_deviations.append(np.linalg.norm(odometry[i] - closest_point))
            else:
                route_deviations.append(-np.linalg.norm(odometry[i] - closest_point))
            if len(distances_along_route):
                distances_along_route.append(running_distance + np.linalg.norm(closest_point - previous_closest_point))
                running_distance += np.linalg.norm(closest_point - previous_closest_point)
            else:
                distances_along_route.append(0.0)
            previous_closest_point = closest_point

    if show_plots:
        plt.plot(distances_along_route, route_deviations, label="Crosstrack Error")
        plt.plot(distances_along_route, np.zeros(len(route_deviations)), label="Route")
        plt.xlabel("Downtrack (m)")
        plt.ylabel("Crosstrack Error (m)")
        plt.title("Crosstrack Error vs. Downtrack")
        plt.legend()
        plt.show()
        print("Average Deviation:", np.mean(np.abs(route_deviations)))
        print("Maximum Deviation:", np.max(np.abs(route_deviations)))
    return np.array(distances_along_route), np.array(route_deviations)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot deviation between C1T path driven and desired route")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("route_topic", type=str, help="Topic containing desired route to follow")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_crosstrack_error(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["route_topic"])