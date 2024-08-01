# Check that there is at least one message on the specified topic in the provided bag

import yaml
import argparse, argcomplete
import os


def check_message_published(bag_dir, topic):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a rosbag directory?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    # Gather number of messages on each topic
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    if topic not in topic_count_dict:
        # If topic does not exist, there are no messages on it
        return False
    # Return true if there is at least one message on the specified topic
    return topic_count_dict[topic] > 0

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check that a message exists in the bag")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("topic", type=str, help="Topic to search for messages on")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    result = check_message_published(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["topic"])
    print(result)
