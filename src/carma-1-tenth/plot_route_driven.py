from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import os


def plot_route_driven(bag_dir):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic], storage_id=storage_id)
    route_graph = None
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    odom_count = 0
    for idx in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == route_topic:
                route_graph = msg
            else:
                odometry[odom_count] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
                odom_count += 1
    plt.plot(-odometry[:,1], odometry[:,0], label="Path Driven")
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            if i == 0:
                plt.plot(-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x, 'ro', label="Route")
            else:
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
    plt.legend()
    plt.title("Route Driven Compared to Desired Route")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
