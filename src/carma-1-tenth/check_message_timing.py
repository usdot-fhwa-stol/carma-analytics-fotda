# Check the times between messages received on two topics

from rosbag_utils import open_bagfile
import numpy as np
import yaml
import tqdm
import datetime
import argparse, argcomplete
import os


def check_message_timing(bag_dir, call_topic, response_topic):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    reader, type_map = open_bagfile(bag_dir, topics=[call_topic, response_topic], storage_id=storage_id)
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    if call_topic not in topic_count_dict or response_topic not in topic_count_dict:
        return np.array([np.inf])
    call_topic_count, response_topic_count = 0,0
    call_topic_times, response_topic_times = np.zeros((topic_count_dict[call_topic],)), np.zeros((topic_count_dict[response_topic],))
    # Iterate through bag and store odometry + route_graph messages
    for _ in tqdm.tqdm(iterable=range(topic_count_dict[call_topic] + topic_count_dict[response_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            if topic == call_topic:
                call_topic_times[call_topic_count] = t_
                call_topic_count += 1
            elif topic == response_topic:
                if call_topic_count > response_topic_count:
                    response_topic_times[response_topic_count] = t_
                    response_topic_count += 1
    call_topic_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in call_topic_times]
    response_topic_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in response_topic_times]
    return np.array([(response_time - call_time).total_seconds() for call_time, response_time in zip(call_topic_datetimes, response_topic_datetimes)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot standard deviation of the vehicle's localization as a function of downtrack distance")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    parser.add_argument("call_topic", type=str, help="Topic that initiates a message exchange")
    parser.add_argument("response_topic", type=str, help="Topic that responds to the call topic")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    time_differences = check_message_timing(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["call_topic"], argdict["response_topic"])
    print(time_differences)
