# Plot the path driven by the C1T truck and the route that it intended to follow


import argparse, argcomplete
from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from plot_crosstrack_error import find_closest_point
from plot_localization import plot_localization
import os


def plot_route_driven(bag_dir, show_plots=True):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    # Open bag
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic], storage_id=storage_id)
    route_graph = None
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    odom_count = 0
    # Iterate through bag
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic])):
        if(reader.has_next()):
            (topic, data, _) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == route_topic:
                route_graph = msg
            else:
                odometry[odom_count] = [-msg.pose.pose.position.y, msg.pose.pose.position.x]
                odom_count += 1
    # Find max and min x/y values in graph to scale plots
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    route_graph_coordinates = []
    route_downtrack_distances = []
    map_coords_to_downtrack = dict()
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            # For each graph node, store its coordinates and adjust the min/max x/y
            route_graph_coordinates.append(np.array([-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x]))
            x_min = np.min([-route_graph.markers[i].pose.position.y, x_min])
            y_min = np.min([route_graph.markers[i].pose.position.x, y_min])
            x_max = np.max([-route_graph.markers[i].pose.position.y, x_max])
            y_max = np.max([route_graph.markers[i].pose.position.x, y_max])
            # Compute downtrack distance of graph node
            if len(route_downtrack_distances):
                route_downtrack_distances.append(route_downtrack_distances[-1] + np.linalg.norm(route_graph_coordinates[-1] - route_graph_coordinates[-2]))
            else:
                route_downtrack_distances.append(0.0)
            plt.text(route_graph_coordinates[-1][0] + 0.1, route_graph_coordinates[-1][1] + 0.1, "{:.1f}".format(route_downtrack_distances[-1]))
            map_coords_to_downtrack[tuple(route_graph_coordinates[-1])] = route_downtrack_distances[-1]
    route_graph_coordinates = np.array(route_graph_coordinates)
    plt.plot(route_graph_coordinates[:,0], route_graph_coordinates[:,1], 'ro-', label="Route")
    plt.plot(odometry[:,0], odometry[:,1], 'b', label="Estimated Path Driven")
    plt.xlim([x_min - 1.0, x_max + 1.0])
    plt.ylim([y_min - 1.0, y_max + 1.0])
    ax = plt.gca()
    particle_distances_along_route, particle_std_deviations = plot_localization(bag_dir, show_plots=False)
    # For each graph node, plot and ellipse using the vehicle's localization std. dev. to show uncertainty in position
    for route_coordinate in route_graph_coordinates:
        closest_odom_to_route, _ = find_closest_point(odometry, route_coordinate, trim_ends=False)
        route_coordinate_downtrack_distance = map_coords_to_downtrack[tuple(route_coordinate)]
        std_deviation_at_route_coordinate = particle_std_deviations[np.abs(route_coordinate_downtrack_distance - particle_distances_along_route).argmin()]
        e2 = Ellipse(closest_odom_to_route, 2.0 * std_deviation_at_route_coordinate[0], 2.0 * std_deviation_at_route_coordinate[1], color='b', fill=False, ls="--")
        ax.add_patch(e2)
    plt.xlabel("Horizontal Coordinate (m)")
    plt.ylabel("Vertical Coordinate (m)")
    plt.legend()
    plt.title("Estimated Path Driven Compared to Desired Route")
    if show_plots:
        plt.show()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot the intended route and path driven of C1T trucks")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_route_driven(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
