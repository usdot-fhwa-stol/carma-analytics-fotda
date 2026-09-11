import json
import re

import numpy as np
import matplotlib.pyplot as plt

from parse_ros2_bags import extract_mcap_data, open_bagfile

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

def parse_kafka_log_records(kafka_log_path):
    """
    Parses a Kafka console-consumer log file into a list of message records. Each
    record is expected to start a new line of the form `CreateTime:<epoch_ms>\t{json}`;
    JSON bodies that wrap onto following lines are accumulated until the next `CreateTime`
    line is seen.

    Args:
        kafka_log_path: Path to the Kafka log file

    Returns:
        List of dicts, one per parsed log message, in file order. Each dict is the
        message's JSON body plus a "create_time_ms" key holding that record's Kafka
        CreateTime (epoch milliseconds).
    """
    records = []
    skipped_messages = 0
    pending_message = ""
    create_time_ms = None

    def flush_pending_message():
        nonlocal pending_message, skipped_messages
        if not pending_message:
            return
        try:
            msg = json.loads(pending_message)
            msg["create_time_ms"] = create_time_ms
            records.append(msg)
        except json.JSONDecodeError as e:
            print(f"Error {e.msg} extracting json from Kafka log message. Skipping message.")
            skipped_messages += 1
        pending_message = ""

    with open(kafka_log_path, encoding="utf8", errors="ignore") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("CreateTime"):
                flush_pending_message()
                create_time_ms = int(re.sub(r"[^0-9]", "", line.split(":", 1)[1].split("\t", 1)[0]))
                json_start = line.find("{")
                pending_message = line[json_start:] if json_start != -1 else ""
            else:
                pending_message += line
        flush_pending_message()

    if skipped_messages > 0:
        print(f"WARNING: Skipped {skipped_messages} Kafka log message(s) due to JSON decoding errors.")

    return records


def parse_kafka_log_timestamps(kafka_log_path, timestamp_field="timestamp"):
    """
    Parses a Kafka console-consumer log file into an array of message timestamps.
    See `parse_kafka_log_records` for the expected log format.

    Args:
        kafka_log_path: Path to the Kafka log file
        timestamp_field: JSON field to read each message's timestamp from. Falls back to
            the record's CreateTime if the field is missing from a message (default: "timestamp")

    Returns:
        np.array of epoch millisecond timestamps, sorted, one per parsed log message
    """
    records = parse_kafka_log_records(kafka_log_path)
    timestamps = [record.get(timestamp_field, record["create_time_ms"]) for record in records]

    return np.sort(np.array(timestamps, dtype=float))


def plot_message_intervals(
    title,
    timestamps,
    expected_interval_sec=0.1,
    interval_tolerance_pct=0.1,
    max_view_sec=0.5,
    detection_timestamps=None,
    output_file=None,
    ax=None,
):
    """
    Plots the number of seconds between consecutive message timestamps, highlighting
    intervals that fall outside an expected tolerance band and annotating any interval
    that exceeds the plot's y-axis view limit.

    Args:
        title: Title used for the plot and printed statistics
        timestamps: Array of message timestamps in seconds (need not be sorted)
        expected_interval_sec: Expected number of seconds between consecutive messages (default: 0.1)
        interval_tolerance_pct: Tolerance percentage around the expected interval (default: 0.1 = 10%)
        max_view_sec: Y-axis view limit in seconds; intervals beyond this are shaded red and annotated (default: 0.5)
        detection_timestamps: Optional array of object detection timestamps (seconds from start of
            recording, same time base as `timestamps`), e.g. from a Kafka object detection log. Gaps
            between consecutive detections that exceed `max_view_sec` are shaded green to show that a
            large message interval coincides with there simply being nothing detected to report, rather
            than a missed/dropped message.
        output_file: Optional path to save the plot to. If not given, the plot is shown interactively.
        ax: Optional matplotlib Axes to draw into (e.g. one panel of a combined figure). When given,
            the caller owns the figure's layout and saving, so output_file is ignored.

    Returns:
        Tuple containing:
        - figure: Matplotlib figure object
        - timestamps: Sorted array of message timestamps (seconds from start of recording)
        - intervals: Array of seconds between consecutive messages
    """
    timestamps = np.sort(np.array(timestamps))
    if len(timestamps) < 2:
        raise ValueError(f"Insufficient messages for '{title}' to compute intervals")

    intervals = np.diff(timestamps)

    interval_lower_bound = expected_interval_sec * (1 - interval_tolerance_pct)
    interval_upper_bound = expected_interval_sec * (1 + interval_tolerance_pct)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    over_max_view = np.flatnonzero(intervals > max_view_sec)
    for i, idx in enumerate(over_max_view):
        ax.axvspan(
            timestamps[idx], timestamps[idx + 1],
            color="red", alpha=0.2,
            label=f"Interval > {max_view_sec} s" if i == 0 else None
        )
        ax.annotate(
            f"{intervals[idx]:.2f}s",
            xy=(timestamps[idx + 1], max_view_sec),
            xytext=(0, 2),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=7,
            color="darkred",
        )

    if detection_timestamps is not None:
        detection_timestamps = np.sort(np.array(detection_timestamps))
        if len(detection_timestamps) >= 2:
            detection_intervals = np.diff(detection_timestamps)
            no_detection_gaps = np.flatnonzero(detection_intervals > max_view_sec)
            for i, idx in enumerate(no_detection_gaps):
                ax.axvspan(
                    detection_timestamps[idx], detection_timestamps[idx + 1],
                    color="green", alpha=0.2,
                    label=f"No Object Detected > {max_view_sec} s" if i == 0 else None
                )
            print(
                f"Object detection gaps > {max_view_sec} s: {len(no_detection_gaps)} "
                f"(out of {len(detection_timestamps)} detection log messages)"
            )

    ax.plot(timestamps[1:], intervals, ".-", markersize=4, linewidth=1, label="Seconds Since Previous Message")
    ax.axhline(y=np.median(intervals), color="r", linestyle="--", label="Median")
    ax.axhline(y=expected_interval_sec, color="g", linestyle="--", label=f"Expected Interval ({expected_interval_sec} s)")
    ax.axhline(y=interval_lower_bound, color="orange", linestyle=":", label="Tolerance Band")
    ax.axhline(y=interval_upper_bound, color="orange", linestyle=":")
    ax.fill_between(
        timestamps[1:], interval_lower_bound, interval_upper_bound,
        alpha=0.2, color="orange", label="Tolerance Zone"
    )
    ax.set_title(title, pad=20)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Seconds Since Previous Message")
    ax.set_ylim(0, max_view_sec)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if own_figure:
        fig.tight_layout()

    stats = calculate_error_statistics(intervals)
    print_stats(stats, f"{title} Interval Statistics (seconds)")
    print(f"Total messages: {len(timestamps)}")

    intervals_within_tolerance = np.sum(
        (intervals >= interval_lower_bound) & (intervals <= interval_upper_bound)
    )
    percentage_within_tolerance = (intervals_within_tolerance / len(intervals)) * 100
    print(
        f"Intervals within {interval_tolerance_pct*100:.1f}% tolerance of {expected_interval_sec} s: "
        f"{intervals_within_tolerance}/{len(intervals)} ({percentage_within_tolerance:.1f}%)"
    )

    if own_figure:
        if output_file:
            fig.savefig(output_file, dpi=300)
            print(f"Plot saved to: {output_file}")
        else:
            plt.show()

    return fig, timestamps, intervals


def extract_and_plot_message_intervals(
    mcap_path,
    topic,
    message_type=None,
    expected_interval_sec=0.1,
    interval_tolerance_pct=0.1,
    detection_log_path=None,
    detection_log_timestamp_field="timestamp",
    start_time=None,
    end_time=None,
    output_file=None,
    ax=None,
    max_view_sec=0.5,
):
    """
    Extracts message timestamps for a topic from an MCAP file and plots the seconds
    between consecutive messages. If message_type is given, the topic's messages are
    first filtered to those whose `message_type` field matches (useful for topics like
    carma_driver_msgs/msg/ByteArray that carry multiple message types on one topic).

    Args:
        mcap_path: Path to MCAP file
        topic: ROS topic to analyze
        message_type: Optional value to filter the topic's message_type field on
        expected_interval_sec: Expected number of seconds between consecutive messages (default: 0.1)
        interval_tolerance_pct: Tolerance percentage around the expected interval (default: 0.1 = 10%)
        detection_log_path: Optional path to a Kafka object detection log (e.g. from
            v2xhub_sim_sensor_detected_object). When given, gaps between consecutive object
            detections are shaded green on the plot to distinguish "nothing was detected" from
            an actual missed/dropped message.
        detection_log_timestamp_field: JSON field to read each detection's timestamp from
            (default: "timestamp")
        start_time: Time to start the analysis (seconds from start of recording)
        end_time: Time to end the analysis (seconds from start of recording)
        output_file: Optional path to save the plot to. If not given, the plot is shown interactively.
        ax: Optional matplotlib Axes to draw into (e.g. one panel of a combined figure). When given,
            the caller owns the figure's layout and saving, so output_file is ignored.
        max_view_sec: Y-axis view limit in seconds; intervals (and detection gaps) beyond this are shaded
            (default: 0.5)

    Returns:
        Tuple containing:
        - figure: Matplotlib figure object
        - timestamps: Array of message timestamps (seconds from start of recording)
        - intervals: Array of seconds between consecutive messages
    """
    if message_type:
        extracted_data = extract_mcap_data(
            mcap_path, [topic], start_time=start_time, end_time=end_time,
            field_extractors={topic: lambda msg: msg.message_type}
        )
        timestamps, message_types = extracted_data[topic]
        timestamps = np.array(timestamps)[np.array(message_types) == message_type]
        if len(timestamps) < 2:
            raise ValueError(f"Insufficient '{message_type}' messages on topic {topic} to compute intervals")
        title = f"Time Between Messages - {topic} ({message_type})"
    else:
        extracted_data = extract_mcap_data(mcap_path, [topic], start_time=start_time, end_time=end_time)
        timestamps, _ = extracted_data[topic]
        title = f"Time Between Messages - {topic}"

    detection_timestamps = None
    if detection_log_path:
        detection_timestamps_ms = parse_kafka_log_timestamps(
            detection_log_path, timestamp_field=detection_log_timestamp_field
        )
        # Recording start time (ns since epoch) so the log's absolute timestamps can be
        # placed on the same "seconds from start of recording" axis as `timestamps`.
        _, _, global_start_time_ns = open_bagfile(str(mcap_path))
        detection_timestamps = (detection_timestamps_ms * 1e6 - global_start_time_ns) / 1e9

        # Kafka logs are often a running record spanning many separate recordings/days, so
        # only keep detections that fall within this mcap's time range.
        recording_start, recording_end = np.min(timestamps), np.max(timestamps)
        detection_timestamps = detection_timestamps[
            (detection_timestamps >= recording_start) & (detection_timestamps <= recording_end)
        ]

    return plot_message_intervals(
        title=title,
        timestamps=timestamps,
        expected_interval_sec=expected_interval_sec,
        interval_tolerance_pct=interval_tolerance_pct,
        detection_timestamps=detection_timestamps,
        output_file=output_file,
        ax=ax,
        max_view_sec=max_view_sec,
    )


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
