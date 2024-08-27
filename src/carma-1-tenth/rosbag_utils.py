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

def find_closest_point(point_arr, point, trim_ends=True):
    difference_arr = np.linalg.norm(point_arr - point, axis=1)
    min_index = difference_arr.argmin()
    # Don't want to include deviations if we have not yet reached the route or have completed it
    if trim_ends and (min_index == 0 or min_index == len(difference_arr) - 1):
        return None, None
    return point_arr[min_index], min_index


def find_path_driven(odometry, nx_graph):
    route_coordinates_reached = []
    route_ids_reached = set()
    candidate_nodes = dict()
    previous_node_id = None
    for odom in odometry:
        min_distance = np.inf
        min_node = None
        for node in nx_graph.nodes(data=True):
            if np.linalg.norm(node[1]['pos'] - odom) < min_distance:
                min_distance = np.linalg.norm(node[1]['pos'] - odom)
                min_node = node
        # Add node if it is the first node or if it has not already been recorded and is a neighbor of the previous node
        if len(route_coordinates_reached) == 0:
            route_coordinates_reached.append([min_node[0], min_node[1]['pos'][0], min_node[1]['pos'][1]])
            route_ids_reached.add(min_node[0])
            previous_node_id = min_node[0]
            if len(list(nx_graph.neighbors(min_node[0]))) > 1:
                for neighbor in nx_graph.neighbors(min_node[0]):
                    candidate_nodes.add(neighbor)
        elif min_node[0] not in route_ids_reached:
            if len(candidate_nodes) == 0 and min_node[0] in nx_graph.neighbors(previous_node_id):
                route_coordinates_reached.append([min_node[0], min_node[1]['pos'][0], min_node[1]['pos'][1]])
                route_ids_reached.add(min_node[0])
                previous_node_id = min_node[0]
                if len(list(nx_graph.neighbors(min_node[0]))) > 1:
                    print("Root:", min_node)
                    for neighbor in nx_graph.neighbors(min_node[0]):
                        candidate_nodes[neighbor] = nx_graph.nodes[neighbor]
                        print("Child:", neighbor)
            elif min_node[0] not in candidate_nodes:
                for candidate_node in candidate_nodes.items():
                    if min_node[0] in nx_graph.neighbors(candidate_node[0]):
                        route_coordinates_reached.append([candidate_node[0], candidate_node[1]['pos'][0], candidate_node[1]['pos'][1]])
                        route_coordinates_reached.append([min_node[0], min_node[1]['pos'][0], min_node[1]['pos'][1]])
                        route_ids_reached.add(candidate_node[0])
                        route_ids_reached.add(min_node[0])
                        previous_node_id = min_node[0]
                        candidate_nodes.clear()
                        break

    return np.array(route_coordinates_reached)