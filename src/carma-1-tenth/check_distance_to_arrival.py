# Report the distance between the goal position (sent on /incoming_mobility_operation) and the position
# the C1T vehicle reports when acknowledging it has reached the goal (sent on /outgoing_mobility_operation)
# Assumes that there are an equal number of goal messages and acks in the ROS2 bag

from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import argparse, argcomplete
import os
import json


def check_distance_to_arrival(bag_dir):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    # Goal and Ack topics
    goal_topic = '/incoming_mobility_operation'
    ack_topic = '/outgoing_mobility_operation'
    # Open bag
    reader, type_map = open_bagfile(bag_dir, topics=[goal_topic, ack_topic], storage_id=storage_id)
    # Gather number of messages on each topic
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    # Number of goal messages must equal number of acks
    if topic_count_dict[goal_topic] != topic_count_dict[ack_topic]:
        print("Number of goal messages (%d) does not equal number of ack messages (%d)".format(topic_count_dict[goal_topic], topic_count_dict[ack_topic]))
        return np.array([np.inf])
    # Count number of goal/ack messages processed
    goal_position_count, ack_position_count = 0, 0
    # Store goal and ack positions
    goal_positions = np.zeros((topic_count_dict[goal_topic], 2))
    ack_positions = np.zeros((topic_count_dict[ack_topic], 2))
    # Iterate through bag
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[goal_topic] + topic_count_dict[ack_topic])):
        if(reader.has_next()):
            (topic, data, _) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == goal_topic:
                strategy_params = json.loads(msg.strategy_params)
                # Store goal position
                goal_positions[goal_position_count] = [strategy_params["destination"]["longitude"], strategy_params["destination"]["latitude"]]
                goal_position_count += 1
            elif topic == ack_topic:
                strategy_params = json.loads(msg.strategy_params)
                # Store ack position
                ack_positions[ack_position_count] = [strategy_params["location"]["longitude"], strategy_params["location"]["latitude"]]
                ack_position_count += 1
    # Return euclidean distance between each goal position and the reported ack position
    return np.linalg.norm(goal_positions - ack_positions, axis=1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute the differences between each goal destination and vehicle position on reported arrival")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    distances_from_goal = check_distance_to_arrival(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    print(distances_from_goal)
