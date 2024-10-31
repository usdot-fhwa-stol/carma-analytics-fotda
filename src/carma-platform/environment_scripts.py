import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from visualization_msgs.msg import MarkerArray

def plot_lanelet2_map():
    """Plot Lanelet2 map from the specified MCAP file"""
    mcap_path = "/workspaces/carma/src/analysis-data/rosbag2_2024-10-22_213643/rosbag2_2024-10-22_213643_0.mcap"
    topic = "/environment/lanelet2_map_viz"
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create reader with correct format
        storage_options = rosbag2_py.StorageOptions(
            uri=mcap_path,
            storage_id="mcap"
        )
        
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Set up plotting
        plt.figure(figsize=(15, 10))
        
        # Get topic type
        topic_types = reader.get_all_topics_and_types()
        topic_type = None
        for topic_info in topic_types:
            if topic_info.name == topic:
                topic_type = topic_info.type
                break
                
        if topic_type is None:
            raise ValueError(f"Topic {topic} not found in bag file")
            
        print(f"Found topic type: {topic_type}")
        
        # Read messages
        msg_count = 0
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            
            if topic_name == topic:
                try:
                    msg = deserialize_message(data, MarkerArray)
                    msg_count += 1
                    print(f"Processing message {msg_count}")
                    
                    # Process each marker
                    for marker in msg.markers:
                        if len(marker.points) > 0:
                            x_coords = [p.x for p in marker.points]
                            y_coords = [p.y for p in marker.points]
                            
                            # Plot based on marker type
                            if marker.type == 4:  # LINE_STRIP
                                plt.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)
                            elif marker.type == 5:  # LINE_LIST
                                for i in range(0, len(x_coords), 2):
                                    if i + 1 < len(x_coords):
                                        plt.plot(x_coords[i:i+2], y_coords[i:i+2], 'r-', linewidth=1, alpha=0.7)
                
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
        
        print(f"Processed {msg_count} messages")
        
        # Customize the plot
        plt.title('Lanelet2 Map Visualization')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        
        # Save and display
        plt.savefig('lanelet2_map.png')
        print("Plot saved as lanelet2_map.png")
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
        
    except Exception as e:
        print(f"Error reading bag file: {e}")
    finally:
        plt.close()
        rclpy.shutdown()

if __name__ == "__main__":
    plot_lanelet2_map()