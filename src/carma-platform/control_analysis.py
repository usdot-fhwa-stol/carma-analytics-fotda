from parse_ros2_bags import open_bagfile
import numpy as np
import argparse, argcomplete
import os
import yaml
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the differences between each goal destination and vehicle position on reported arrival"
    )
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)

    bag_dir = argdict["bag_in"]

    # Open metadata.yaml
    metadatafile: str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError(
            "Metadata file %s does not exist. Are you sure %s is a rosbag directory?"
            % (metadatafile, bag_dir)
        )
    with open(metadatafile, "r") as f:
        metadata_dict: dict = yaml.load(f, Loader=yaml.SafeLoader)[
            "rosbag2_bagfile_information"
        ]
    storage_id = metadata_dict["storage_identifier"]
    # Goal and Ack topics
    rviz_topic = "/goal_pose"
    goal_topic = "/incoming_mobility_operation"
    ack_topic = "/outgoing_mobility_operation"
    # Open bag
    reader, type_map = open_bagfile(
        bag_dir, topics=[goal_topic, ack_topic, rviz_topic], storage_id=storage_id
    )
    # Gather number of messages on each topic
    topic_count_dict = {
        entry["topic_metadata"]["name"]: entry["message_count"]
        for entry in metadata_dict["topics_with_message_count"]
    }
    topic_count_dict = {
        entry["topic_metadata"]["name"]: entry["message_count"]
        for entry in metadata_dict["topics_with_message_count"]
    }
    if rviz_topic not in topic_count_dict:
        topic_count_dict[rviz_topic] = 0
    elif goal_topic not in topic_count_dict:
        topic_count_dict[goal_topic] = 0
    # Count number of goal/ack messages processed
    goal_position_count, ack_position_count = 0, 0

    print(json.dumps(topic_count_dict, sort_keys=True, indent=4))
