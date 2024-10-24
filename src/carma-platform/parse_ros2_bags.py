import rosbag2_py
import numpy as np
import os
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def get_rosbag_options(path, serialization_format="cdr", storage_id="sqlite3"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options


def open_bagfile(path, topics=[]):
    """Configure and open MCAP file reader"""
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="mcap")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    if topics:
        filt = rosbag2_py.StorageFilter(topics)
        reader.set_filter(filt)

    return reader, type_map


def extract_mcap_data(mcap_path, topics, field_extractors=None):
    """
    Extract data from specified topics in an MCAP file.

    Args:
        mcap_path (str): Path to the MCAP file
        topics (list): List of topics to extract data from
        field_extractors (dict): Optional dictionary mapping topics to functions that extract
                                desired fields from the message. If None, returns entire message.
                                Example: {"/topic": lambda msg: msg.field_name}

    Returns:
        dict: Dictionary mapping topics to tuples of (timestamps, values)
              timestamps are in seconds from start of recording
              values are lists of extracted data for each message

    Raises:
        ValueError: If MCAP file doesn't exist or specified topics not found
    """
    if not os.path.exists(mcap_path):
        raise ValueError(f"MCAP file {mcap_path} does not exist")

    # Initialize default field extractors if none provided
    if field_extractors is None:
        field_extractors = {topic: lambda msg: msg for topic in topics}

    # Verify all topics have extractors
    missing_extractors = set(topics) - set(field_extractors.keys())
    if missing_extractors:
        raise ValueError(f"Missing field extractors for topics: {missing_extractors}")

    # Open bag
    reader, type_map = open_bagfile(str(mcap_path), topics=topics)

    # Verify all requested topics exist in the file
    missing_topics = set(topics) - set(type_map.keys())
    if missing_topics:
        raise ValueError(f"Topics not found in MCAP file: {missing_topics}")

    # Initialize data storage
    data = {topic: {"values": [], "timestamps": []} for topic in topics}

    # Read messages
    print("Reading messages...")
    while reader.has_next():
        topic, msg_data, timestamp = reader.read_next()
        if topic in topics:
            msg_type = type_map[topic]
            msg = deserialize_message(msg_data, get_message(msg_type))

            try:
                extracted_value = field_extractors[topic](msg)
                data[topic]["values"].append(extracted_value)
                data[topic]["timestamps"].append(timestamp)
            except Exception as e:
                print(
                    f"Warning: Failed to extract data from message on topic {topic}: {e}"
                )

    # Verify we got data for all topics
    empty_topics = [
        topic for topic, topic_data in data.items() if not topic_data["values"]
    ]
    if empty_topics:
        raise ValueError(f"No valid messages found for topics: {empty_topics}")

    # Convert to numpy arrays and normalize timestamps
    result = {}
    for topic, topic_data in data.items():
        timestamps = np.array(topic_data["timestamps"])
        values = np.array(topic_data["values"])

        # Convert timestamps to seconds from start
        start_time = min(data[t]["timestamps"][0] for t in topics)
        timestamps = (timestamps - start_time) / 1e9

        result[topic] = (timestamps, values)

    print("Finished extracting the required data for this analysis")
    return result
