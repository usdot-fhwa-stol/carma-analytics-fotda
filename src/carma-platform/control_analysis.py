from parse_ros2_bags import open_bagfile
import numpy as np
import argparse, argcomplete
import os
import yaml
import json
import tqdm
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from matplotlib import pyplot as plt

def analyze_route_state(mcap_path):
    """Analyze route state data from MCAP file"""
    if not os.path.exists(mcap_path):
        raise ValueError(f"MCAP file {mcap_path} does not exist")
    
    # Route state topic
    route_topic = '/guidance/route_state'
    
    # Open bag
    reader, type_map = open_bagfile(mcap_path, topics=[route_topic])
    
    if route_topic not in type_map:
        raise ValueError(f"Topic {route_topic} not found in MCAP file")
    
    # Store cross track data and timestamps
    cross_tracks = []
    timestamps = []
    
    # Read messages
    print("Reading messages...")
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == route_topic:
            msg_type = type_map[topic]
            msg = deserialize_message(data, get_message(msg_type))
            cross_tracks.append(msg.cross_track)
            timestamps.append(timestamp)
    
    if not cross_tracks:
        raise ValueError("No valid messages found in MCAP file")
    
    # Convert to numpy arrays
    cross_tracks = np.array(cross_tracks)
    timestamps = np.array(timestamps)
    
    # Convert timestamps to seconds from start
    timestamps = (timestamps - timestamps[0]) / 1e9
    
    # Calculate statistics
    stats = {
        'minimum': np.min(cross_tracks),
        'maximum': np.max(cross_tracks),
        'median': np.median(cross_tracks),
        'std_dev': np.std(cross_tracks),
        'mean': np.mean(cross_tracks),
        'sample_count': len(cross_tracks),
        'rms': np.sqrt(np.mean(np.square(cross_tracks)))
    }
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cross_tracks, 'b-', label='Cross Track Error', linewidth=1)
    plt.axhline(y=stats['median'], color='r', linestyle='--', label='Median')
    plt.fill_between(timestamps, 
                    stats['median'] - stats['std_dev'],
                    stats['median'] + stats['std_dev'],
                    alpha=0.2, color='r', label='±1 Std Dev')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cross Track Error (m)')
    plt.title('Route State Cross Track Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return stats, plt.gcf(), cross_tracks, timestamps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze cross track error from route state messages"
    )
    parser.add_argument("mcap_file", type=str, help="Path to MCAP file")
    parser.add_argument("--output", type=str, help="Path to save plot (optional)")
    parser.add_argument("--save-data", type=str, help="Path to save data as .npz (optional)")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    try:
        stats, fig, cross_tracks, timestamps = analyze_route_state(args.mcap_file)
        
        print("\nCross Track Error Statistics:")
        print(f"Minimum: {stats['minimum']:.4f} m")
        print(f"Maximum: {stats['maximum']:.4f} m")
        print(f"Median:  {stats['median']:.4f} m")
        print(f"Mean:    {stats['mean']:.4f} m")
        print(f"RMS:     {stats['rms']:.4f} m")
        print(f"Std Dev: {stats['std_dev']:.4f} m")
        print(f"Sample Count: {stats['sample_count']}")
        
        if args.save_data:
            np.savez(args.save_data, 
                    cross_tracks=cross_tracks, 
                    timestamps=timestamps,
                    stats=stats)
            print(f"\nData saved to: {args.save_data}")
            
        if args.output:
            plt.savefig(args.output)
            print(f"\nPlot saved to: {args.output}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)