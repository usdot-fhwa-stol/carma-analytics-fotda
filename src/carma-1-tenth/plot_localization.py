# Plot the standard deviation of the vehicle's localization as a function of downtrack distance


from rosbag_utils import open_bagfile
from plot_crosstrack_error import find_closest_point
import numpy as np
import yaml
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import make_interp_spline, interp1d
import argparse, argcomplete
import os


def plot_localization(bag_dir, start_offset=0.0):
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadatafile):
        raise ValueError("Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    storage_id = metadata_dict['storage_identifier']
    odom_topic = '/amcl_pose'
    route_topic = '/route_graph'
    particles_topic = '/particle_cloud'
    reader, type_map = open_bagfile(bag_dir, topics=[odom_topic, route_topic, particles_topic], storage_id=storage_id)
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}

    route_graph = None
    odom_count = 0
    particles_count = 0
    odometry = np.zeros((topic_count_dict[odom_topic], 2))
    odometry_times = np.zeros((topic_count_dict[odom_topic],))
    particle_std_deviations = np.zeros((topic_count_dict[particles_topic],))
    particle_times = np.zeros((topic_count_dict[particles_topic],))
    # Iterate through bag and store odometry + route_graph messages
    for idx in tqdm.tqdm(iterable=range(topic_count_dict[odom_topic] + topic_count_dict[route_topic] + topic_count_dict[particles_topic])):
        if(reader.has_next()):
            (topic, data, t_) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == route_topic:
                route_graph = msg
            elif topic == odom_topic:
                odometry[odom_count] = [-msg.pose.pose.position.y, msg.pose.pose.position.x]
                odometry_times[odom_count] = t_
                odom_count += 1
            elif topic == particles_topic:
                particles = np.zeros((len(msg.particles), 2))
                weights = np.zeros((len(msg.particles),))
                for i in range(len(msg.particles)):
                    particles[i] = [msg.particles[i].pose.position.x, msg.particles[i].pose.position.y]
                    weights[i] = msg.particles[i].weight
                particle_std_deviations[particles_count] = np.mean(np.diagonal(np.sqrt(np.cov(particles.T, aweights=weights))))
                particle_times[particles_count] = t_
                particles_count += 1

    route_coordinates = []
    # Rotate the route_graph coordinates 90 degrees to match C1T coordinates (x-forward, y-left)
    for i in range(len(route_graph.markers)):
        if route_graph.markers[i].type == 2:
            route_coordinates.append([-route_graph.markers[i].pose.position.y, route_graph.markers[i].pose.position.x])
    route_coordinates_with_distance = np.zeros((len(route_coordinates), 3))
    route_coordinates_with_distance[:, 1:] = np.array(route_coordinates)
    running_distance = 0.0
    # Assign a distance along the route for each x,y coordinate along the route
    for i in range(len(route_coordinates)):
        if i > 0:
            running_distance += np.linalg.norm(route_coordinates_with_distance[i, 1:] - route_coordinates_with_distance[i-1, 1:])
        route_coordinates_with_distance[i,0] = running_distance
    # Fit 2D splines that map the distance along the route to the x and y coordinates to upsample the points
    x_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,1])
    y_spline = make_interp_spline(route_coordinates_with_distance[:,0], route_coordinates_with_distance[:,2])
    samples = np.linspace(route_coordinates_with_distance[0,0], route_coordinates_with_distance[-1,0], 10000)
    odometry_datetimes = [datetime.datetime.fromtimestamp(time * 1e-9) for time in odometry_times]
    odometry_times_seconds = [(date - odometry_datetimes[0]).total_seconds() for date in odometry_datetimes]
    route_x_points = x_spline(samples)
    route_y_points = y_spline(samples)
    odom_std_deviations = []
    odom_distances_along_route = []
    odometry_times_trimmed = []
    # For each odometry message, compute the standard deviation using the current, previous, and next message
    for i in range(2, len(odometry) - 2):
        closest_point, _ = find_closest_point(np.array([route_x_points, route_y_points]).T, odometry[i])
        if closest_point is not None and np.linalg.norm(odometry[i]- odometry[i-1]) >= 0.005:
            odom_std_deviations.append(np.mean(np.std(odometry[i-2:i+2], axis=0)))
            odometry_times_trimmed.append(odometry_times_seconds[i])
            if len(odom_distances_along_route):
                odom_distances_along_route.append(odom_distances_along_route[-1] + np.linalg.norm(closest_point - previous_closest_point))
            else:
                odom_distances_along_route.append(np.linalg.norm(closest_point - np.array([route_x_points[0], route_y_points[0]])))
        previous_closest_point = closest_point
    # Interpolate the odometry of the robot using timestamps
    odometry_interp = interp1d(odometry_times, odometry, axis=0)
    # For each particle cloud message, compute the standard deviation of the particles
    particle_trimmed_std_deviations = []
    particle_distances_along_route = []
    for i in range(len(particle_std_deviations)):
        try:
            current_odom = odometry_interp(particle_times[i])
        except ValueError:  # Happens if a particle cloud is received after the last odometry message
            continue
        closest_point, _ = find_closest_point(np.array([route_x_points, route_y_points]).T, current_odom)
        if closest_point is not None:
            particle_trimmed_std_deviations.append(particle_std_deviations[i])
            if len(particle_distances_along_route):
                particle_distances_along_route.append(particle_distances_along_route[-1] + np.linalg.norm(closest_point - previous_closest_point))
            else:
                particle_distances_along_route.append(np.linalg.norm(closest_point - np.array([route_x_points[0], route_y_points[0]])))
        previous_closest_point = closest_point

    print("Average Localization Standard Deviation:", np.mean(odom_std_deviations))
    print("Maximum Localization Standard Deviation:", np.max(odom_std_deviations))

    print("Average PF Standard Deviation:", np.mean(particle_trimmed_std_deviations))
    print("Maximum PF Standard Deviation:", np.max(particle_trimmed_std_deviations))

    plt.plot(odom_distances_along_route, odom_std_deviations)
    plt.xlabel("Downtrack (m)")
    plt.ylabel("Localization Standard Deviation (m)")
    plt.title("Localization Standard Deviation vs. Downtrack")
    plt.figure()
    plt.plot(particle_distances_along_route, particle_trimmed_std_deviations)
    plt.xlabel("Downtrack (m)")
    plt.ylabel("Particle Filter Standard Deviation (m)")
    plt.title("Particle Filter Standard Deviation vs. Downtrack")
    plt.figure()

    plt.plot(odom_distances_along_route, np.gradient(odom_distances_along_route, odometry_times_trimmed))
    plt.xlabel("Downtrack (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("Vehicle Speed vs. Downtrack")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot standard deviation of the vehicle's localization as a function of downtrack distance")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_localization(os.path.normpath(os.path.abspath(argdict["bag_in"])))
    plt.show()
