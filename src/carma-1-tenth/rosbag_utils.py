import rosbag2_py
import numpy as np
import networkx as nx

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
    route_ids_reached = []
    blocked_nodes = set()
    previous_node_id = None
    branch_start = None
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
            route_ids_reached.append(min_node[0])
            previous_node_id = min_node[0]
            if len(list(nx_graph.neighbors(min_node[0]))) > 1:
                branch_start = min_node[0]
                for neighbor in nx_graph.neighbors(min_node[0]):
                    blocked_nodes.add(neighbor)
                    for next_neighbor in nx_graph.neighbors(neighbor):
                        blocked_nodes.add(next_neighbor)
        elif min_node[0] != route_ids_reached[-1]:
            if len(blocked_nodes) == 0 and min_node[0] in nx_graph.neighbors(previous_node_id):
                route_coordinates_reached.append([min_node[0], min_node[1]['pos'][0], min_node[1]['pos'][1]])
                route_ids_reached.append(min_node[0])
                previous_node_id = min_node[0]
                if len(list(nx_graph.neighbors(min_node[0]))) > 1:
                    branch_start = min_node[0]
                    for neighbor in nx_graph.neighbors(min_node[0]):
                        blocked_nodes.add(neighbor)
                        for next_neighbor in nx_graph.neighbors(neighbor):
                            blocked_nodes.add(next_neighbor)
            elif len(blocked_nodes) > 0 and min_node[0] not in blocked_nodes:
                shortest_path = nx.shortest_path(nx_graph, source=branch_start, target=min_node[0])
                if len(list(shortest_path)) > 4:
                    continue
                for nodeid in shortest_path:
                    if nodeid == branch_start:
                        continue
                    node = nx_graph.nodes[nodeid]
                    route_coordinates_reached.append([nodeid, node['pos'][0], node['pos'][1]])
                    route_ids_reached.append(nodeid)
                previous_node_id = min_node[0]
                blocked_nodes.clear()
                branch_start = None
    return np.array(route_coordinates_reached)