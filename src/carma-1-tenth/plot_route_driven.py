# Plot the path driven by the C1T truck and the route that it intended to follow


import argparse, argcomplete
from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from rosbag_utils import find_closest_point, find_path_driven
from plot_localization import plot_localization
import networkx as nx
import os

def get_slowdown_speeds(speeds, target_speeds):
    slowdown_speeds = []
    max_target_speed = np.max(target_speeds)
    min_slowdown_speed = max_target_speed
    for i in range(len(speeds)):
        if target_speeds[i] < max_target_speed:
            if speeds[i] < min_slowdown_speed:
                min_slowdown_speed = speeds[i]
                min_slowdown_idx = i
        else:
            if min_slowdown_speed != max_target_speed:
                slowdown_speeds.append([min_slowdown_idx, min_slowdown_speed])
                min_slowdown_speed = max_target_speed
    slowdown_speeds.append([min_slowdown_idx, min_slowdown_speed])
    return slowdown_speeds


def plot_colorline(x, y, c):
    col = cm.coolwarm_r((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=col[i], linewidth=2.0)
    im = ax.scatter(x, y, c=c, s=0, cmap=cm.coolwarm_r)
    cb = plt.colorbar(im)
    cb.ax.yaxis.set_label_coords(5.0, 0.0)
    cb.ax.set_ylabel('Speed (m/s)', rotation=270)
    return im


def plot_route_driven(bag_dir, show_localization=False, show_speed=False):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    vel_topic = '/odom'
    target_vel_topic = '/cmd_vel'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    # Open bag
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic, vel_topic, target_vel_topic], storage_id=storage_id)
    route_graph = None
    odometry = np.zeros((topic_count_dict[odom_topic], 4))
    current_velocity = 0.0
    current_target_velocity = 0.0
    odom_count = 0
    # Iterate through bag
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic] + topic_count_dict[vel_topic] + topic_count_dict[target_vel_topic])):
        if(reader.has_next()):
            (topic, data, _) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == route_topic:
                route_graph = msg
            elif topic == odom_topic:
                odometry[odom_count] = [-msg.pose.pose.position.y, msg.pose.pose.position.x, current_velocity, current_target_velocity]
                odom_count += 1
            elif topic == vel_topic:
                current_velocity = msg.twist.twist.linear.x
            elif topic == target_vel_topic:
                current_target_velocity = msg.linear.x
    # Find max and min x/y values in graph to scale plots
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    route_graph_coordinates = []
    nx_graph = nx.DiGraph()
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            # For each graph node, store its coordinates and adjust the min/max x/y
            route_graph_coordinates.append([route_graph.markers[i].id, -route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x])
            x_min = np.min([-route_graph.markers[i].pose.position.y, x_min])
            y_min = np.min([route_graph.markers[i].pose.position.x, y_min])
            x_max = np.max([-route_graph.markers[i].pose.position.y, x_max])
            y_max = np.max([route_graph.markers[i].pose.position.x, y_max])
            nx_graph.add_node(route_graph.markers[i].id, pos=(-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x))
    route_graph_coordinates = np.array(route_graph_coordinates)
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 5:
            plt.plot([-route_graph.markers[i].points[0].y, -route_graph.markers[i].points[1].y], [route_graph.markers[i].points[0].x, route_graph.markers[i].points[1].x], 'r')
            _, start_index = find_closest_point(route_graph_coordinates[:, 1:], [-route_graph.markers[i].points[0].y, route_graph.markers[i].points[0].x], trim_ends=False)
            _, end_index = find_closest_point(route_graph_coordinates[:, 1:], [-route_graph.markers[i].points[1].y, route_graph.markers[i].points[1].x], trim_ends=False)
            nx_graph.add_edge(route_graph_coordinates[start_index, 0], route_graph_coordinates[end_index, 0])
    route_coordinates_reached = find_path_driven(odometry[:,:2], nx_graph)
    map_coords_to_downtrack = dict()
    running_distance = 0.0
    previous_coordinate = None
    for coordinate in route_coordinates_reached:
        if previous_coordinate is None:
            map_coords_to_downtrack[tuple(np.round(coordinate[1:], 2))] = running_distance
        else:
            running_distance += np.linalg.norm(coordinate[1:] - previous_coordinate[1:])
            map_coords_to_downtrack[tuple(np.round(coordinate[1:], 2))] = running_distance
        previous_coordinate = coordinate            
    if show_speed:
        plot_colorline(odometry[:,0], odometry[:,1], odometry[:,2])
        slowdown_speeds = get_slowdown_speeds(odometry[:,2], odometry[:,3])
        legend_initialized = False
        for idx, speed in slowdown_speeds:
            if not legend_initialized:
                plt.scatter(odometry[idx,0], odometry[idx,1], s=20, c='red', label = "Slowdown Speed", zorder=10)
                legend_initialized = True
            else:
                plt.scatter(odometry[idx,0], odometry[idx,1], s=20, c='red', zorder=10)
            plt.text(odometry[idx,0] + 0.1, odometry[idx,1] + 0.1, "{:.2f}".format(speed))
    else:
        plt.plot(route_graph_coordinates[1:,1], route_graph_coordinates[1:,2], 'ro', label="Route")
        plt.plot(odometry[:,0], odometry[:,1], 'b', label="Estimated Path Driven")
        for coordinate, downtrack in zip(map_coords_to_downtrack.keys(), map_coords_to_downtrack.values()):
            plt.text(coordinate[0] + 0.1, coordinate[1] + 0.1, "{:.1f}".format(downtrack))
    plt.xlim([x_min - 1.0, x_max + 1.0])
    plt.ylim([y_min - 1.0, y_max + 1.0])
    # For each graph node, plot and ellipse using the vehicle's localization std. dev. to show uncertainty in position
    if show_localization:
        ax = plt.gca()
        particle_distances_along_route, particle_std_deviations = plot_localization(bag_dir, show_plots=False)
        for route_coordinate in route_coordinates_reached:
            closest_odom_to_route, _ = find_closest_point(odometry[:,:2], route_coordinate[1:], trim_ends=False)
            route_coordinate_downtrack_distance = map_coords_to_downtrack[tuple(np.round(route_coordinate[1:], 2))]
            std_deviation_at_route_coordinate = particle_std_deviations[np.abs(route_coordinate_downtrack_distance - particle_distances_along_route).argmin()]
            e2 = Ellipse(closest_odom_to_route, 2.0 * std_deviation_at_route_coordinate[0], 2.0 * std_deviation_at_route_coordinate[1], color='b', fill=False, ls="--")
            ax.add_patch(e2)
    plt.xlabel("Horizontal Coordinate (m)")
    plt.ylabel("Vertical Coordinate (m)")
    plt.legend()
    plt.title("Estimated Path Driven")
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot the intended route and path driven of C1T trucks")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("--show_localization", action="store_true", help="Show ellipses representing localization uncertainty")
    parser.add_argument("--show_speed", action="store_true", help="Color the route using the vehicle's speed")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_route_driven(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["show_localization"], argdict["show_speed"])
