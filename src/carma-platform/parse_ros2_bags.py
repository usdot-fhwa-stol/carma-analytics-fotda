import rosbag2_py
import numpy as np
import os
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def get_rosbag_options(path, serialization_format="cdr", storage_id="sqlite3"):
    """
    Get storage and converter options for reading a rosbag.

    Args:
        path (str): Path to the rosbag file.
        serialization_format (str): Serialization format for the bag (default: "cdr").
        storage_id (str): Storage ID for the bag (default: "sqlite3").

    Returns:
        tuple: A tuple containing storage options and converter options.
    """
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options


def open_bagfile(path, topics=[], serialization_format="cdr", storage_id="mcap"):
    """
    Configure and open an MCAP file reader.

    Args:
        path (str): Path to the MCAP file.
        topics (list): List of topics to filter (default: empty list).
        serialization_format (str): Serialization format for the bag (default: "cdr").
        storage_id (str): Storage ID for the bag (default: "mcap").

    Returns:
        tuple: A tuple containing the reader and a mapping of topic names to types.

    Raises:
        ValueError: If the bag file cannot be opened or if there are issues with the topics.
    """
    storage_options, converter_options = get_rosbag_options(
        path, serialization_format=serialization_format, storage_id=storage_id
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


def check_mcap_file_existence(mcap_path):
    """
    Check if the MCAP file exists.

    Args:
        mcap_path (str): Path to the MCAP file.

    Raises:
        ValueError: If the MCAP file does not exist.
    """
    if not os.path.exists(mcap_path):
        raise ValueError(f"MCAP file {mcap_path} does not exist")


def initialize_field_extractors(topics, field_extractors):
    """
    Initialize default field extractors if none provided.

    Args:
        topics (list): List of topics to initialize extractors for.
        field_extractors (dict): Optional dictionary mapping topics to extractor functions.

    Returns:
        dict: A dictionary of field extractors.

    Raises:
        ValueError: If there are missing field extractors for any of the specified topics.
    """
    if field_extractors is None:
        return {topic: lambda msg: msg for topic in topics}
    
    # Check for missing field extractors.
    missing_extractors = set(topics) - set(field_extractors.keys())
    if missing_extractors:
        raise ValueError(f"Missing field extractors for topics: {missing_extractors}")
    
    return field_extractors


def check_missing_topics(topics, type_map):
    """
    Check for missing topics in the MCAP file.

    Args:
        topics (list): List of topics to check.
        type_map (dict): Mapping of topic names to types.

    Raises:
        ValueError: If any of the specified topics are not found in the MCAP file.
    """
    missing_topics = set(topics) - set(type_map.keys())
    if missing_topics:
        raise ValueError(f"Topics not found in MCAP file: {missing_topics}")


def read_messages(reader, topics, type_map, field_extractors):
    """
    Read messages from the bag and extract data.

    Args:
        reader: The bag file reader.
        topics (list): List of topics to read messages from.
        type_map (dict): Mapping of topic names to types.
        field_extractors (dict): Dictionary of field extractors for each topic.

    Returns:
        dict: A dictionary containing extracted values and timestamps for each topic.

    Raises:
        Exception: If there is an error during message extraction.
    """
    data = {topic: {"values": [], "timestamps": []} for topic in topics}

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
                print(f"Warning: Failed to extract data from message on topic {topic}: {e}")

    return data


def filter_data_with_start_and_end_time(data, topics, start_time, end_time):
    """
    Filter data based on the specified time range.

    Args:
        data (dict): Dictionary containing extracted data.
        topics (list): List of topics to filter.
        start_time (float): Optional start time in seconds.
        end_time (float): Optional end time in seconds.

    Returns:
        dict: Filtered data for each topic.

    Raises:
        ValueError: If no data is found for a topic in the specified time range.
    """
    global_start_time = min(data[t]["timestamps"][0] for t in topics)
    result = {}

    for topic, topic_data in data.items():
        timestamps = np.array(topic_data["timestamps"])
        values = np.array(topic_data["values"])

        # Convert timestamps to seconds from start
        timestamps = (timestamps - global_start_time) / 1e9

        # Filter based on time range if specified
        if start_time is not None or end_time is not None:
            mask = np.ones_like(timestamps, dtype=bool)
            if start_time is not None:
                mask &= (timestamps >= start_time)
            if end_time is not None:
                mask &= (timestamps <= end_time)

            timestamps = timestamps[mask]
            values = values[mask]

            # Check if we have any data left after filtering
            if len(timestamps) == 0:
                raise ValueError(f"No data found for topic {topic} in specified time range")

        result[topic] = (timestamps, values)

    return result


def extract_mcap_data(mcap_path, topics, start_time=None, end_time=None, field_extractors=None):
    """
    Extract data from specified topics in an MCAP file within a given time range.

    Args:
        mcap_path (str): Path to the MCAP file.
        topics (list): List of topics to extract data from.
        start_time (float): Optional start time in seconds from beginning of recording.
        end_time (float): Optional end time in seconds from beginning of recording.
        field_extractors (dict): Optional dictionary mapping topics to functions that extract
                                desired fields from the message. If None, returns entire message.
                                Example: {"/topic": lambda msg: msg.field_name}

    Returns:
        dict: Dictionary mapping topics to tuples of (timestamps, values)
              timestamps are in seconds from start of recording
              values are lists of extracted data for each message.

    Raises:
        ValueError: If MCAP file doesn't exist, specified topics not found, or no valid messages found.
    """
    check_mcap_file_existence(mcap_path)
    field_extractors = initialize_field_extractors(topics, field_extractors)

    # Open bag
    reader, type_map = open_bagfile(str(mcap_path), topics=topics)
    check_missing_topics(topics, type_map)

    # Read messages
    data = read_messages(reader, topics, type_map, field_extractors)

    # Verify we got data for all topics
    empty_topics = [topic for topic, topic_data in data.items() if not topic_data["values"]]
    if empty_topics:
        raise ValueError(f"No valid messages found for topics: {empty_topics}")

    # Filter data based on time range
    result = filter_data_with_start_and_end_time(data, topics, start_time, end_time)

    print("Finished extracting the required data for this analysis")
    return result