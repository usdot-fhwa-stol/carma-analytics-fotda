import sys
from mcap_ros2.reader import read_ros2_messages
import matplotlib.pyplot as plt

def process_mcap_file(filename, topic_name):
    all_points = []

    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            for marker in msg.ros_msg.markers:
                marker_points = [(point.x, point.y) for point in marker.points]
                all_points.extend(marker_points)

    print(f"Processed {len(all_points)} data points from {filename} for topic {topic_name}")
    return all_points

def process_localization_file(filename, topic_name):
    pose_x = []
    pose_y = []

    for msg in read_ros2_messages(filename):
        if msg.channel.topic == topic_name:
            pose_x.append(msg.ros_msg.pose.position.x)
            pose_y.append(msg.ros_msg.pose.position.y)

    print(f"Processed {len(pose_x)} data points from {filename} for topic {topic_name}")
    return pose_x, pose_y

def plot_2d_data(lanelet2_data, pose_data, labels):
    fig, ax = plt.subplots(figsize=(20, 16))

    # Plot lanelet2 map data
    if lanelet2_data:
        x, y = zip(*lanelet2_data)
        ax.scatter(x, y, label=labels[0], alpha=0.6, s=1, c='blue')

    # Plot pose data
    if pose_data:
        pose_x, pose_y = pose_data
        ax.plot(pose_x, pose_y, label=labels[1], alpha=0.8, linewidth=2, c='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Lanelet2 Map and Vehicle Pose')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <path_to_mcap_file1> <path_to_mcap_file2>")
        sys.exit(1)

    lanelet2_topic = "/environment/lanelet2_map_viz"
    localization_topic = "/localization/current_pose"

    mcap_files = sys.argv[1:]
    lanelet2_data = []
    pose_data = None
    labels = ['lanelet2_map', 'vehicle_pose', 'ndt']

    for mcap_file in mcap_files:
        print(f"Processing file: {mcap_file}")
        lanelet2_data.extend(process_mcap_file(mcap_file, lanelet2_topic))
        pose_x, pose_y = process_localization_file(mcap_file, localization_topic)
        if pose_data is None:
            pose_data = (pose_x, pose_y)
        else:
            pose_data = (pose_data[0] + pose_x, pose_data[1] + pose_y)

    plot_2d_data(lanelet2_data, pose_data, labels)