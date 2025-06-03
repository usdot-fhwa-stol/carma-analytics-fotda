from parse_ros2_bags import extract_mcap_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import json
from utils import calculate_error_statistics, print_stats

STD_DEV_LABEL_STRING = "±1 Std Dev"
TIME_SECONDS_LABEL_STRING = "Time (seconds)"

def check_message_broadcast_rate(
    mcap_path,
    topic_name,
    expected_rate_hz,
    rate_tolerance_pct=0.1,  # 10% tolerance
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes the broadcast rate of messages on any given topic to verify they are
    transmitted at the expected frequency.

    Args:
        mcap_path: Path to MCAP file
        topic_name: Name of the ROS topic to analyze (e.g., "/message/incoming_mobility_operation")
        expected_rate_hz: Expected broadcast rate in Hz
        rate_tolerance_pct: Tolerance percentage for rate matching (default: 0.1 = 10%)
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots

    Returns:
        Tuple containing:
        - is_passed: Boolean indicating if broadcast rate meets requirements
        - stats: Dictionary with statistical analysis
        - figure: Matplotlib figure object
        - broadcast_intervals: Array of time intervals between messages
        - timestamps: Array of message timestamps

    Deps:
        Topics: [topic_name]
        Msgs: Any ROS message type with header.stamp field
    """

    topics = [topic_name]

    # Extract message timestamps - we only need the timing, not the content
    # Try different common timestamp field patterns
    def extract_timestamp(msg):
        try:
            # Try header.stamp first (most common)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # Try stamp field directly
            elif hasattr(msg, 'stamp'):
                return msg.stamp.sec + msg.stamp.nanosec * 1e-9
            # If no timestamp field found, return None to use message receive time
            else:
                return None
        except AttributeError:
            # Fallback to message receive time if timestamp extraction fails
            return None

    try:
        extracted_data = extract_mcap_data(
            mcap_path,
            topics,
            start_time=start_time,
            end_time=end_time,
            field_extractors={topic_name: extract_timestamp}
        )

        # If timestamp extraction failed, use message receive timestamps
        timestamps, extracted_stamps = extracted_data[topics[0]]

        # Use extracted timestamps if available, otherwise use receive timestamps
        if extracted_stamps.any() and extracted_stamps[0] is not None:
            timestamps = np.array([stamp for stamp in extracted_stamps if stamp is not None])
        else:
            timestamps = np.array(timestamps)
            print(f"Warning: Using message receive timestamps for {topic_name} (no header.stamp found)")

    except Exception as e:
        print(f"Error extracting data from topic {topic_name}: {e}")
        return False, {}, None, [], []

    if len(timestamps) < 2:
        print(f"Error: Insufficient data points for rate analysis on topic {topic_name}")
        return False, {}, None, [], []

    # Sort timestamps to ensure chronological order
    timestamps = np.sort(timestamps)

    # Calculate time intervals between consecutive messages
    broadcast_intervals = np.diff(timestamps)

    # Calculate instantaneous rates (1/interval)
    instantaneous_rates = 1.0 / broadcast_intervals

    # Calculate rolling average rate over 1-second windows
    window_size = 1.0  # 1 second window
    rolling_rates = []
    rolling_timestamps = []

    for i in range(len(timestamps)):
        # Find messages within 1 second window from current timestamp
        window_start = timestamps[i]
        window_end = window_start + window_size

        # Count messages in this window
        messages_in_window = np.sum((timestamps >= window_start) & (timestamps < window_end))

        if messages_in_window > 1:
            # Calculate rate as messages per second
            rolling_rate = messages_in_window / window_size
            rolling_rates.append(rolling_rate)
            rolling_timestamps.append(timestamps[i])

    rolling_rates = np.array(rolling_rates)
    rolling_timestamps = np.array(rolling_timestamps)

    # Calculate statistics for both instantaneous and rolling rates
    instant_stats = calculate_error_statistics(instantaneous_rates, start_time, end_time)
    rolling_stats = calculate_error_statistics(rolling_rates, start_time, end_time) if len(rolling_rates) > 0 else {}

    # Determine pass/fail criteria
    rate_lower_bound = expected_rate_hz * (1 - rate_tolerance_pct)
    rate_upper_bound = expected_rate_hz * (1 + rate_tolerance_pct)

    # Check if rolling average rate is within tolerance
    rates_within_tolerance = np.sum(
        (rolling_rates >= rate_lower_bound) & (rolling_rates <= rate_upper_bound)
    ) if len(rolling_rates) > 0 else 0

    total_windows = len(rolling_rates) if len(rolling_rates) > 0 else 1
    percentage_within_tolerance = (rates_within_tolerance / total_windows) * 100

    # Pass if at least 95% of time windows are within tolerance
    is_passed = bool(percentage_within_tolerance >= 95.0)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot instantaneous rates
    ax1.plot(
        timestamps[1:],
        instantaneous_rates,
        ".",
        markersize=2,
        label="Instantaneous Rate",
        alpha=0.6
    )
    ax1.axhline(y=expected_rate_hz, color="g", linestyle="--", label=f"Expected Rate ({expected_rate_hz} Hz)")
    ax1.axhline(y=rate_lower_bound, color="orange", linestyle=":", label=f"Tolerance Band")
    ax1.axhline(y=rate_upper_bound, color="orange", linestyle=":")
    ax1.fill_between(
        timestamps, rate_lower_bound, rate_upper_bound,
        alpha=0.2, color="orange", label="Tolerance Zone"
    )

    if len(instantaneous_rates) > 0:
        ax1.axhline(y=instant_stats["median"], color="r", linestyle="--", label="Median")

    ax1.set_title(f"Instantaneous Broadcast Rate - {topic_name}")
    ax1.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax1.set_ylabel("Rate (Hz)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, max(expected_rate_hz * 2, np.max(instantaneous_rates) * 1.1) if len(instantaneous_rates) > 0 else expected_rate_hz * 2)

    # Plot rolling average rates
    if len(rolling_rates) > 0:
        ax2.plot(
            rolling_timestamps,
            rolling_rates,
            "-",
            linewidth=1.5,
            label="1-Second Window Rate",
            color="blue"
        )
        ax2.axhline(y=rolling_stats["median"], color="r", linestyle="--", label="Median")

    ax2.axhline(y=expected_rate_hz, color="g", linestyle="--", label=f"Expected Rate ({expected_rate_hz} Hz)")
    ax2.axhline(y=rate_lower_bound, color="orange", linestyle=":", label=f"Tolerance Band")
    ax2.axhline(y=rate_upper_bound, color="orange", linestyle=":")
    ax2.fill_between(
        timestamps, rate_lower_bound, rate_upper_bound,
        alpha=0.2, color="orange", label="Tolerance Zone"
    )

    ax2.set_title(f"1-Second Window Average Broadcast Rate - {topic_name}")
    ax2.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax2.set_ylabel("Rate (Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, expected_rate_hz * 2)

    plt.tight_layout()

    # Print statistics
    print(f"\n=== Message Broadcast Rate Analysis ===")
    print(f"Topic: {topic_name}")
    print(f"Expected Rate: {expected_rate_hz} Hz")
    print(f"Tolerance: ±{rate_tolerance_pct*100:.1f}% ({rate_lower_bound:.1f} - {rate_upper_bound:.1f} Hz)")
    print(f"Total Messages: {len(timestamps)}")
    print(f"Analysis Duration: {timestamps[-1] - timestamps[0]:.2f} seconds" if len(timestamps) > 1 else "N/A")

    if len(instantaneous_rates) > 0:
        print_stats(instant_stats, "Instantaneous Rate Statistics")

    if len(rolling_rates) > 0:
        print_stats(rolling_stats, "1-Second Window Rate Statistics")
        print(f"Time windows within tolerance: {rates_within_tolerance}/{total_windows} ({percentage_within_tolerance:.1f}%)")

    print(f"\nResult: {'PASSED' if is_passed else 'FAILED'}")

    # Prepare comprehensive stats dictionary
    stats = {
        "topic_name": topic_name,
        "expected_rate_hz": expected_rate_hz,
        "rate_tolerance_pct": rate_tolerance_pct,
        "total_messages": len(timestamps),
        "analysis_duration": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "instantaneous_rates": instant_stats,
        "rolling_window_rates": rolling_stats,
        "percentage_within_tolerance": float(percentage_within_tolerance),
        "is_passed": is_passed
    }

    # Generate safe filename from topic name
    safe_topic_name = topic_name.replace("/", "_").replace(" ", "_")

    # Save results
    if save_stats_dir:
        stats_full_path = save_stats_dir / f"{safe_topic_name}_broadcast_rate_stats.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {stats_full_path}")

    if save_data_dir:
        np.savez(
            save_data_dir / f"{safe_topic_name}_broadcast_rate_data.npz",
            timestamps=timestamps,
            broadcast_intervals=broadcast_intervals,
            instantaneous_rates=instantaneous_rates,
            rolling_rates=rolling_rates,
            rolling_timestamps=rolling_timestamps,
            stats=stats,
        )
        print(f"Data saved to: {save_data_dir}")

    if save_plot_dir:
        plt.savefig(save_plot_dir / f"{safe_topic_name}_broadcast_rate_analysis.png", dpi=300)
        print(f"Plot saved to: {save_plot_dir}")
    else:
        plt.show()

    return is_passed, stats, fig, broadcast_intervals, timestamps
