# Verify the vehicle correctly acks the port drayage pickup, dropoff, and holding area messages by checking the cargo_id variable
# in the goal messages and the acknowledgement messages


from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import argparse, argcomplete
import os
import json


def check_port_drayage_ack(bag_dir, operation):
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
    # If number of goals does not equal number of acks, return False
    if topic_count_dict[goal_topic] != topic_count_dict[ack_topic]:
        print("Number of goal messages (%d) does not equal number of ack messages (%d)".format(topic_count_dict[goal_topic], topic_count_dict[ack_topic]))
        return False
    # Iterate through bag
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[goal_topic] + topic_count_dict[ack_topic])):
        if(reader.has_next()):
            (topic, data, _) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == goal_topic:
                strategy_params = json.loads(msg.strategy_params)
                # If goal message with desired operation is received, store what the cargo_id should be for the ack
                if strategy_params["operation"] == operation:
                    if operation == "PICKUP" or operation == "HOLDING_AREA":
                        goal_cargo_id = strategy_params["cargo_id"]
                    elif operation == "DROPOFF":
                        goal_cargo_id = ""
                    else:
                        raise ValueError("Unsupported operation %s, please use PICKUP, DROPOFF, or HOLDING_AREA")
            elif topic == ack_topic:
                strategy_params = json.loads(msg.strategy_params)
                # Check the ack message for the desired operation has the correct cargo_id
                if strategy_params["operation"] == operation:
                    if (strategy_params["operation"] == "PICKUP" or strategy_params["operation"] == "HOLDING_AREA") and strategy_params["cargo_id"] == goal_cargo_id:
                        return True
                    elif strategy_params["operation"] == "DROPOFF" and not strategy_params["cargo"]:
                        return True
    # Return false if desired operation is not in the bag
    return False

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Verify the vehicle correctly acks the port drayage pickup, dropoff, and holding area messages")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("operation", type=str, help="PICKUP, DROPOFF, or HOLDING_AREA")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    result = check_port_drayage_ack(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["operation"])
    print(result)
