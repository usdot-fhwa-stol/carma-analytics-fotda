import sys
from mcap_ros2.reader import read_ros2_messages
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for environments without display
import matplotlib.pyplot as plt
from pathlib import Path

def extract_map_update_publication_time_from_mcap(filename):
    """
    Extract Lanelet2 map update publication time from an MCAP file.

    Args:
        filename (str): Path to the MCAP file to process

    Returns:
        list: Time relative to started recording
    """
    topic_name = "/environment/semantic_map"
    all_points = []
    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            for marker in msg.ros_msg.markers:
                # Extract x,y coordinates from each point in the marker
                marker_points = [(point.x, point.y) for point in marker.points]
                all_points.extend(marker_points)
    print(f"Processed {len(all_points)} data points from {filename} for topic {topic_name}")
    return all_points

def extract_lanelet2_map_publication_time_from_mcap(filename):
    """
    Extract Lanelet2 map publication time from an MCAP file.

    Args:
        filename (str): Path to the MCAP file to process

    Returns:
        list: Time relative to started recording
    """
    topic_name = "/environment/semantic_map"
    all_points = []
    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            for marker in msg.ros_msg.markers:
                # Extract x,y coordinates from each point in the marker
                marker_points = [(point.x, point.y) for point in marker.points]
                all_points.extend(marker_points)
    print(f"Processed {len(all_points)} data points from {filename} for topic {topic_name}")
    return all_points

def extract_lanelet2_map_from_mcap(filename):
    """
    Extract Lanelet2 map data points from an MCAP file.

    Args:
        filename (str): Path to the MCAP file to process

    Returns:
        list: List of tuples containing (x, y) coordinates of map points

    Note:
        Processes messages from the '/environment/lanelet2_map_viz' topic
        Each point represents a vertex in the Lanelet2 map visualization
    """
    topic_name = "/environment/lanelet2_map_viz"
    all_points = []
    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            for marker in msg.ros_msg.markers:
                # Extract x,y coordinates from each point in the marker
                marker_points = [(point.x, point.y) for point in marker.points]
                all_points.extend(marker_points)
    print(f"Processed {len(all_points)} data points from {filename} for topic {topic_name}")
    return all_points

def extract_pose_points_from_mcap(filename):
    """
    Extract vehicle pose data from an MCAP file.

    Args:
        filename (str): Path to the MCAP file to process

    Returns:
        tuple: (timestamps, pose_data)
            - timestamps: List of message timestamps in nanoseconds
            - pose_data: List of tuples containing (x, y) coordinates of vehicle positions

    Note:
        Processes messages from the '/localization/current_pose' topic
        Timestamps are converted to nanoseconds for precise temporal alignment
    """
    topic_name = "/localization/current_pose"
    timestamps = []
    pose_data = []
    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            # Convert ROS2 time to nanoseconds for consistency
            timestamp_ns = msg.ros_msg.header.stamp.sec * 1e9 + msg.ros_msg.header.stamp.nanosec
            timestamps.append(timestamp_ns)
            pose_data.append((msg.ros_msg.pose.position.x, msg.ros_msg.pose.position.y))
    print(f"Processed {len(pose_data)} data points from {filename} for topic {topic_name}")
    return timestamps, pose_data

def filter_map_points_for_trajectory(lanelet_points, pose_x, pose_y, buffer=50):
    """
    Filter Lanelet2 map points to only include those near the vehicle trajectory.

    Args:
        lanelet_points (list): List of (x, y) tuples representing map points
        pose_x (list): List of x-coordinates from vehicle trajectory
        pose_y (list): List of y-coordinates from vehicle trajectory
        buffer (float, optional): Additional distance in meters to include around trajectory bounds.
                                Defaults to 50 meters.

    Returns:
        tuple: (filtered_points, bounds)
            - filtered_points: List of (x, y) tuples within the bounded area
            - bounds: Tuple of (min_x, max_x, min_y, max_y) defining the filtered area
    """
    # Calculate trajectory bounds with buffer
    min_x, max_x = min(pose_x) - buffer, max(pose_x) + buffer
    min_y, max_y = min(pose_y) - buffer, max(pose_y) + buffer

    # Filter map points to only include those within bounds
    filtered_points = []
    for point in lanelet_points:
        x, y = point
        if min_x <= x <= max_x and min_y <= y <= max_y:
            filtered_points.append(point)

    return filtered_points, (min_x, max_x, min_y, max_y)

def plot_2d_map_and_pose(lanelet2_data, pose_data, output_dir=None):
    """
    Create a 2D visualization of the Lanelet2 map and vehicle trajectory.

    Args:
        lanelet2_data (list): List of (x, y) tuples representing map points
        pose_data (list): List of (x, y) tuples (or similar dimension of list or np.array)
                                  representing vehicle positions
        output_dir (str, optional): Directory to save the generated plot.
                                  If None, saves in current directory.

    Output:
        Saves a PNG file named 'lanelet2_map_section_with_trajectory.png'
        Plot includes:
        - Blue dots for map points
        - Red line for vehicle trajectory
        - Green marker for start position
        - Red marker for end position
    """
    # Create figure with large size for detail
    fig, ax = plt.subplots(figsize=(20, 16))

    if pose_data and lanelet2_data:
        # Separate x and y coordinates for plotting
        pose_x = [pose[0] for pose in pose_data]
        pose_y = [pose[1] for pose in pose_data]

        # Filter map points to show only relevant section
        filtered_points, bounds = filter_map_points_for_trajectory(lanelet2_data, pose_x, pose_y)
        min_x, max_x, min_y, max_y = bounds

        # Plot filtered lanelet2 map data
        if filtered_points:
            x, y = zip(*filtered_points)
            ax.scatter(x, y, label="Lanelet2 Map Data", alpha=0.6, s=1, c='blue')

        # Plot complete pose trajectory
        ax.plot(pose_x, pose_y, label="Vehicle Pose", alpha=0.8, linewidth=2, c='red')

        # Mark start and end points
        ax.plot(pose_x[0], pose_y[0], 'go', markersize=10, label='Start Position')
        ax.plot(pose_x[-1], pose_y[-1], 'ro', markersize=10, label='End Position')

        # Set axis limits to show only relevant section
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    # Configure plot appearance
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Lanelet2 Map Section with Vehicle Trajectory')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Save the plot
    output_file = 'lanelet2_map_section_with_trajectory.png'
    if output_dir is not None:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / output_file
    else:
        save_path = Path(output_file)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {save_path}")

if __name__ == "__main__":
    """
    Main execution block for processing MCAP files and generating visualizations.
    Command-line usage:
    python environment_scripts.py <path_to_mcap_file>
    """
    # Validate command line arguments
    if len(sys.argv) != 2:
        print("Usage: python environment_scripts.py <path_to_mcap_file1>")
        sys.exit(1)

    mcap_file = sys.argv[1]
    lanelet2_data = []
    pose_data = []

    # Process the file to extract lanelet2_map and pose data
    print(f"Processing file: {mcap_file}")
    lanelet2_data.extend(extract_lanelet2_map_from_mcap(mcap_file))
    _, pose_data_temp = extract_pose_points_from_mcap(mcap_file)
    pose_data.extend(pose_data_temp)

    # Create visualization with filtered map section
    plot_2d_map_and_pose(lanelet2_data, pose_data)
