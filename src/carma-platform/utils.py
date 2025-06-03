import numpy as np
from parse_ros2_bags import extract_mcap_data
from pathlib import Path
import json

def check_message_receipt(
    mcap_path,
    topic_name,
    start_time=None,
    end_time=None,
    min_message_count=1,
    save_stats_dir=None,
    save_data_dir=None
):
    """
    Check whether messages were received on a given topic within the specified time range.

    Args:
        mcap_path: Path to MCAP file
        topic_name: Name of the topic to check for messages
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis
        min_message_count: Minimum number of messages required (default: 1)
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data

    Returns:
        tuple: (is_successful, message_count, first_message_time, last_message_time, timestamps)
            - is_successful: Boolean - True if at least min_message_count messages were received
            - message_count: Total number of messages received on the topic
            - first_message_time: Timestamp of the first message (None if no messages)
            - last_message_time: Timestamp of the last message (None if no messages)
            - timestamps: List of all message timestamps
    """

    try:
        # Extract message timestamps from the topic
        topics = [topic_name]
        extracted_data = extract_mcap_data(
            mcap_path,
            topics,
            start_time=start_time,
            end_time=end_time,
            field_extractors={topic_name: lambda msg: True}  # We only care about timestamps
        )
        timestamps, _ = extracted_data[topic_name]
        message_count = len(timestamps)
        # Determine success based on message count
        is_successful = message_count >= min_message_count

        # Get first and last message times
        first_message_time = timestamps[0] if timestamps.any() else None
        last_message_time = timestamps[-1] if timestamps.any() else None

        # Create statistics
        stats = {
            "topic_name": topic_name,
            "message_count": message_count,
            "min_required_count": min_message_count,
            "is_successful": is_successful,
            "first_message_time": first_message_time,
            "last_message_time": last_message_time,
            "time_range": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": (end_time - start_time) if (start_time and end_time) else None
            }
        }

        # Print results
        if is_successful:
            print(f"✓ Message receipt check PASSED for topic '{topic_name}'")
            print(f"  - Found {message_count} messages (required: {min_message_count})")
            if first_message_time:
                print(f"  - First message at: {first_message_time:.3f}s")
                print(f"  - Last message at: {last_message_time:.3f}s")
        else:
            print(f"✗ Message receipt check FAILED for topic '{topic_name}'")
            print(f"  - Found {message_count} messages (required: {min_message_count})")
            if message_count == 0:
                print(f"  - No messages received on this topic")

        # Save statistics if requested
        if save_stats_dir:
            stats_path = Path(save_stats_dir) / f"message_receipt_{topic_name.replace('/', '_')}.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  - Stats saved to: {stats_path}")

        # Save data if requested
        if save_data_dir:
            data_path = Path(save_data_dir) / f"message_receipt_{topic_name.replace('/', '_')}.npz"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                data_path,
                timestamps=np.array(timestamps) if timestamps else np.array([]),
                stats=stats
            )
            print(f"  - Data saved to: {data_path}")

        return (is_successful, message_count, first_message_time, last_message_time, timestamps)

    except Exception as e:
        print(f"✗ Error checking message receipt for topic '{topic_name}': {str(e)}")
        return (False, 0, None, None, [])


def check_multiple_message_receipts(
    mcap_path,
    topic_list,
    start_time=None,
    end_time=None,
    min_message_count=1,
    save_stats_dir=None,
    save_data_dir=None
):
    """
    Check message receipt for multiple topics at once.

    Args:
        mcap_path: Path to MCAP file
        topic_list: List of topic names to check
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis
        min_message_count: Minimum number of messages required per topic (default: 1)
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data

    Returns:
        dict: Dictionary with topic names as keys and results tuples as values
    """
    results = {}
    all_successful = True

    print(f"Checking message receipt for {len(topic_list)} topics...")
    print("=" * 60)

    for topic in topic_list:
        result = check_message_receipt(
            mcap_path,
            topic,
            start_time,
            end_time,
            min_message_count,
            save_stats_dir,
            save_data_dir
        )
        results[topic] = result
        print("Got out of the function")
        if not result[0]:  # is_successful
            all_successful = False
        print()  # Add spacing between topics

    print("=" * 60)
    print(f"Overall result: {'✓ ALL PASSED' if all_successful else '✗ SOME FAILED'}")

    return results

def calculate_error_statistics(error_values: np.array, start_time=None, end_time=None) -> dict:
    """
    Calculate standard statistics for error values.

    Args:
        error_values: Numpy array of error values
        start_time: Optional start time of the analysis
        end_time: Optional end time of the analysis

    Returns:
        dict: Dictionary containing calculated statistics
    """
    stats = {
        "minimum": np.min(error_values),
        "maximum": np.max(error_values),
        "median": np.median(error_values),
        "std_dev": np.std(error_values),
        "mean": np.mean(error_values),
        "sample_count": len(error_values),
        "rms": np.sqrt(np.mean(np.square(error_values))),
        "start_time_since_recording": start_time,
        "end_time_since_recording": end_time,
    }

    return stats

def print_stats(stats: dict, title: str, decimal_places: int = 4) -> None:
    """
    Print statistics.

    Args:
        stats: Dictionary of statistics
        title: Title for the statistics block
        decimal_places: Number of decimal places (default: 4)
    """
    print(f"\n{title}:")
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, (int, bool, str)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.{decimal_places}f}")

def align_time_series(timestamps1, values1, timestamps2, values2):
    """
    Aligns two time series by interpolating to a common time base.

    Args:
        timestamps1: First series timestamps
        values1: First series values
        timestamps2: Second series timestamps
        values2: Second series values

    Returns:
        tuple: (common_timestamps, interpolated_values1, interpolated_values2)
    """
    # Find the overlapping time range
    start_time = max(timestamps1[0], timestamps2[0])
    end_time = min(timestamps1[-1], timestamps2[-1])

    # Create a common time base with the highest sampling rate
    dt1 = np.mean(np.diff(timestamps1))
    dt2 = np.mean(np.diff(timestamps2))
    dt = min(dt1, dt2)  # Use the smaller time step

    common_timestamps = np.arange(start_time, end_time, dt)

    # Interpolate both signals to the common timebase
    interpolated_values1 = np.interp(common_timestamps, timestamps1, values1)
    interpolated_values2 = np.interp(common_timestamps, timestamps2, values2)

    return common_timestamps, interpolated_values1, interpolated_values2
