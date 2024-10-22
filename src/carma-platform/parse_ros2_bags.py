import rosbag2_py
import numpy as np

def get_rosbag_options(path, serialization_format="cdr", storage_id="sqlite3"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def open_bagfile(path, topics=[], serialization_format="cdr", storage_id="sqlite3"):
    storage_options, converter_options = get_rosbag_options(path, serialization_format=serialization_format, storage_id=storage_id)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    # Create maps for quicker lookup
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    if topics:
        filt = rosbag2_py.StorageFilter(topics)
        reader.set_filter(filt)
    return reader, type_map
