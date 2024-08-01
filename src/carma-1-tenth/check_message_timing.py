# Check the times between messages received on two topics

from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
import datetime
import argparse, argcomplete
import os


def check_message_timing(bag_dir, call_topic, response_topic):
    # Open metadata.yaml
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    # Open bag
    reader, _ = open_bagfile(bag_dir, topics=[call_topic, response_topic], storage_id=storage_id)
    # Gather number of messages on each topic
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    # If the call or response topics are not in the bag, return infinite time between messages
    if call_topic not in topic_count_dict or response_topic not in topic_count_dict:
        return np.array([np.inf])
    # Count number of call/response messages processed
    call_topic_count, response_topic_count = 0,0
    # Store times for the call and response messages
    call_topic_times, response_topic_times = np.zeros((topic_count_dict[call_topic],)), np.zeros((topic_count_dict[response_topic],))
    # Iterate through bag and store timestamps for call and response messages
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[call_topic] + topic_count_dict[response_topic])):
        if(reader.has_next()):
            (topic, _, timestamp) = reader.read_next()
            if topic == call_topic:
                call_topic_times[call_topic_count] = timestamp
                call_topic_count += 1
            elif topic == response_topic:
                # Since messages are read sequentially, we should never have more response messages than call messages
                if call_topic_count > response_topic_count:
                    response_topic_times[response_topic_count] = timestamp
                    response_topic_count += 1
    # Convert to python datetimes
    call_topic_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in call_topic_times]
    response_topic_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in response_topic_times]
    # Return the difference in seconds between the call and response messages
    return np.array([(response_time - call_time).total_seconds() for call_time, response_time in zip(call_topic_datetimes, response_topic_datetimes)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check the times between messages received on two topics")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("call_topic", type=str, help="Topic that initiates a message exchange")
    parser.add_argument("response_topic", type=str, help="Topic that responds to the call topic")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    time_differences = check_message_timing(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["call_topic"], argdict["response_topic"])
    print(time_differences)
