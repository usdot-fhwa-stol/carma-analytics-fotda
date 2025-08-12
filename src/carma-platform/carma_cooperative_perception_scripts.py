from pathlib import Path
from typing import Dict
from run_all_analysis import run_all_analysis
import argparse
import argcomplete
from parse_ros2_bags import open_bagfile, extract_mcap_data
from utils import calculate_error_statistics, print_stats, align_time_series
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import json
from guidance_scripts import (
    get_engage_time,
)

# VARIOUS THRESHOLDS FOR THE METRICS
SDSM_LATENCY_THRESHOLD_IN_S = 0.01
SDSM_LATENCY_TOLERANCE_IN_S = 0.1
LATENCY_APPROXIMATION_THRESHOLD_IN_S = 0.2
INCOMING_MESSAGE_TOPIC = "/hardware_interface/comms/inbound_binary_msg"
INCOMING_SDSM_TOPIC = "/message/incoming_sdsm"
FUSED_SDSM_OBJECTS_TOPIC = "/environment/fused_external_objects"



def analyze_mcap_file_for_cp_analysis(
    mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path
) -> list:
    """Extract single MCAP file and run all cp analysis on it"""
    # 0. General steps needed for all
    try:
        engage_time, disengage_time = get_engage_time(mcap_path)
    except Exception as e:
        print(f"Error getting engage time for mcap {mcap_path}: {e}")
        return None

    all_analysis_stats = []
    intervals = [(engage_time, disengage_time)]
    for start_time, end_time in intervals:
        analysis_stats = {}
        try:
            is_passed, _, _, _ = run_sdsm_latency_analysis(
                mcap_path,
                SDSM_LATENCY_THRESHOLD_IN_S,
                SDSM_LATENCY_TOLERANCE_IN_S,
                engage_time,
                disengage_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_sdsm_latency_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_sdsm_latency_analysis: {e}"
            )
            analysis_stats["run_sdsm_latency_analysis"] = None


        try:
            is_passed, _, _, _ = run_sdsm_approximation_latency_analysis(
                mcap_path,
                LATENCY_APPROXIMATION_THRESHOLD_IN_S,
                None,
                engage_time,
                disengage_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_sdsm_approximation_latency_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_sdsm_approximation_latency_analysis: {e}"
            )
            analysis_stats["run_sdsm_approximation_latency_analysis"] = None

        all_analysis_stats.append(analysis_stats)
    return all_analysis_stats


def run_sdsm_latency_analysis(
    mcap_path,
    error_threshold_to_pass_seconds=1.0,
    threshold_percentile=None,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes latency between SDSM generation by infrastructure and receipt.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass_seconds: Threshold latency error in seconds for passing the analysis
        threshold_percentile: Threshold percentile for passing the analysis
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/message/incoming_sdsm]
        Msgs: carma_v2x_msgs/msg/SensorDataSharingMessage
    """
    try:

        topics = [INCOMING_SDSM_TOPIC]

        extracted_data = extract_mcap_data(
            mcap_path,
            topics,
            start_time=start_time,
            end_time=end_time,
            use_relative_time=False,
            field_extractors={
                INCOMING_SDSM_TOPIC: lambda msg:(
                    msg.sdsm_time_stamp.presence_vector,
                    msg.sdsm_time_stamp.year.year,
                    msg.sdsm_time_stamp.month.month,
                    msg.sdsm_time_stamp.day.day,
                    msg.sdsm_time_stamp.hour.hour,
                    msg.sdsm_time_stamp.minute.minute,
                    msg.sdsm_time_stamp.second.millisecond,
                    msg.sdsm_time_stamp.offset.offset_minute,
                    msg.objects.detected_object_data
                ),

            }
        )

        msg_timestamps, extracted_data = extracted_data[topics[0]]

        sdsm_presence_vector = extracted_data[:,0].astype(int)
        sdsm_year = extracted_data[:,1].astype(int)
        sdsm_month = extracted_data[:,2].astype(int)
        sdsm_day = extracted_data[:,3].astype(int)
        sdsm_hour = extracted_data[:,4].astype(int)
        sdsm_minute = extracted_data[:,5].astype(int)
        sdsm_millisecond = extracted_data[:,6].astype(int)
        sdsm_offset = extracted_data[:,7].astype(int)
        sdsm_objects = extracted_data[:,8]

        epoch_times = []
        for y, m, d, h, mi, ms in zip(sdsm_year, sdsm_month, sdsm_day, sdsm_hour, sdsm_minute, sdsm_millisecond):
            dt = datetime(int(y), int(m), int(d), int(h), int(mi), int(ms // 1000), microsecond=(int(ms) % 1000) * 1000, tzinfo=timezone.utc)
            epoch_times.append(dt.timestamp())
        epoch_times = np.array(epoch_times) * 1e9  + sdsm_offset*60*1e9 # Convert to nanoseconds

        incoming_object_timestamp_ns = []
        topic_timestamp_ns = []
        for message_time, encoded_msg_time, objs in zip(msg_timestamps, epoch_times, sdsm_objects):

            if len(objs) == 0:
                print(f"Warning: SDSM message at {message_time} has no objects. Skipping.")
                continue
            # Calculate total timestamp for each object
            for obj in objs:
                object_creation_time = encoded_msg_time + obj.detected_object_common_data.measurement_time.measurement_time_offset*1e6
                incoming_object_timestamp_ns.append(object_creation_time)
                topic_timestamp_ns.append(message_time)

        # Convert lists to numpy arrays
        incoming_object_timestamp_ns = np.array(incoming_object_timestamp_ns)
        topic_timestamp_ns = np.array(topic_timestamp_ns)

        # Latency: message timestamp vs encoded timestamp
        if len(topic_timestamp_ns) != len(incoming_object_timestamp_ns):
            raise ValueError("ros message timestamps and SDSM timestamps must have the same shape")
        else:
            latency = (topic_timestamp_ns - incoming_object_timestamp_ns)/1e9 # Convert to seconds

        # Calculate statistics
        stats = calculate_error_statistics(
            latency,
        )
        print_stats(stats, "SDSM Latency Analysis")

        # Pass or no pass
        if threshold_percentile == None:
            is_passed = float(stats["median"]) < error_threshold_to_pass_seconds
        elif threshold_percentile > 0:
            is_passed = np.percentile(latency, threshold_percentile) < error_threshold_to_pass_seconds

        plt.figure(figsize=(12,6))
        plt.xlabel('Message Receipt Timestamp (s)')
        plt.ylabel('Latency (s)')
        plt.title('Latency vs Message Receipt Timestamp')
        plt.plot(topic_timestamp_ns, latency, label="SDSM Receive Latency (s)", marker='o', color='blue', linestyle="solid", linewidth=2)
        plt.axhline(y=error_threshold_to_pass_seconds, color='red', linestyle='dashed', label='Error Threshold',linewidth=2)
        plt.axhline(y=stats['mean'], color='green', linestyle='dotted', label='Mean Latency',linewidth=2)
        plt.axhline(y=stats['median'], color='orange', linestyle='dashdot', label='Median Latency',linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.legend()


        # Save results
        if save_stats_dir:
            stats_full_path = save_stats_dir / f"sdsm_latency.json"
            with open(stats_full_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {stats_full_path}")

        if save_data_dir:
            np.savez(
                save_data_dir / f"sdsm_latency_data.npz",
                timestamps=topic_timestamp_ns,
                latency=latency,
                stats=stats,
            )
            print(f"Data saved to: {save_data_dir}")

        if save_plot_dir:
            plt.savefig(save_plot_dir / f"sdsm_latency_analysis.png", dpi=300)
            print(f"Plot saved to: {save_plot_dir}")
        else:
            plt.show()

        return (is_passed, stats, plt.gcf(), topic_timestamp_ns)

    except Exception as e:
        print(f"Error extracting data : {e}")
        return False, {}, None, []


def run_sdsm_approximation_latency_analysis(
    mcap_path,
    error_threshold_to_pass_seconds = 0.2,
    threshold_percentile = None,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes SDSM approximation latency from CARMA Platform's internal cooperative perception logic.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass_seconds: Threshold latency error in seconds for passing the analysis
        threshold_percentile: Threshold percentile for passing the analysis
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    """

    try:
        topics = [FUSED_SDSM_OBJECTS_TOPIC]

        # Extract data
        extracted_data = extract_mcap_data(
            mcap_path,
            topics,
            start_time=start_time,
            end_time=end_time,
            use_relative_time=True,
            field_extractors={
                FUSED_SDSM_OBJECTS_TOPIC: lambda msg: (msg.header, msg.objects),
            },
        )

        # Process extracted data
        fused_objects_timestamps, fused_objects_data = extracted_data[FUSED_SDSM_OBJECTS_TOPIC]
        fused_header, fused_objects = fused_objects_data[:, 0], fused_objects_data[:, 1]

        # Filter out empty object lists
        non_empty_mask = [len(objs) > 0 for objs in fused_objects]
        fused_header = fused_header[non_empty_mask]
        fused_objects = fused_objects[non_empty_mask]

        # Filter out instances of no received SDSM messages
        topics = [INCOMING_SDSM_TOPIC]
        extracted_data = extract_mcap_data(
            mcap_path,
            topics,
            start_time=start_time,
            end_time=end_time,
            use_relative_time=False,
            field_extractors={
                INCOMING_SDSM_TOPIC: lambda msg: (msg.msg_cnt,
                                                  msg.objects.detected_object_data),
            }
        )
        incoming_sdsm_msg_timestamps, incoming_sdsm_extracted_data = extracted_data[topics[0]]
        incoming_sdsm_objects = incoming_sdsm_extracted_data[:, 1]
        objects_count = []
        sdsm_object_timestamps = []
        for obj_timestamps, obj_data in zip(incoming_sdsm_msg_timestamps, incoming_sdsm_objects):
            objects_count.append(len(obj_data))

        # Convert to seconds
        incoming_sdsm_time_in_seconds = np.sort(incoming_sdsm_msg_timestamps/1e9)
        # Remove time ranges where no SDSM messages were received
        ranges_to_remove = detect_gap_ranges(incoming_sdsm_msg_timestamps/1e9, gap_threshold=0.1, buffer=0.0)

        # Calculate approximation latency (in seconds)
        approximation_latency_in_s = []
        msg_timestamps = []
        stats_vals = []
        sdsm_drops = []
        max_obj_timestamp = float('-inf')
        track_objects_count = []
        track_objects_timestamps = []
        prev_obj_timestamp = float('-inf')
        for msg_header, objs in zip(fused_header, fused_objects):
            msg_timestamp = msg_header.stamp.sec * 1e9 + msg_header.stamp.nanosec

            for obj in objs:
                obj_timestamp = obj.header.stamp.sec * 1e9 + obj.header.stamp.nanosec

                # Calculate statistics only for objects that are not in the removed ranges
                # if obj_timestamp/1e9 < incoming_sdsm_time_in_seconds[-1] and not any(start_time < obj_timestamp/1e9 < end_time for start_time, end_time in ranges_to_remove):

                if obj_timestamp > max_obj_timestamp: # Take maximum object timestamp since that is closest to confirmed detection
                    max_obj_timestamp = obj_timestamp

            if max_obj_timestamp ==  prev_obj_timestamp:
                sdsm_drops.append(msg_timestamp/1e9)
                continue
            latency = (msg_timestamp - max_obj_timestamp) / 1e9
            if latency < 0:
                print(f"Warning: Negative latency detected for object with timestamp {obj_timestamp}.")
                exit -1
                # return False, {}, None, [], []
            stats_vals.append(latency)


            track_objects_count.append(len(objs))
            track_objects_timestamps.append(msg_timestamp/1e9)
            approximation_latency_in_s.append( (msg_timestamp - max_obj_timestamp) / 1e9)
            msg_timestamps.append(msg_timestamp/1e9)
            prev_obj_timestamp = max_obj_timestamp


            if len(objs) == 0:
                track_objects_count.append(0)
                track_objects_timestamps.append(msg_timestamp/1e9)


        # Calculate statistics
        stats = calculate_error_statistics(
            stats_vals,
        )
        print_stats(stats, "SDSM Latency Approximation Analysis")

        # Pass or no pass
        if threshold_percentile == None:
            is_passed = float(stats["median"]) < error_threshold_to_pass_seconds
        elif threshold_percentile > 0:
            is_passed = np.percentile(latency, threshold_percentile) < error_threshold_to_pass_seconds

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(msg_timestamps, approximation_latency_in_s, linestyle='solid',linewidth=2, marker='o', label='Latency approximation (s)')

        ax1.set_ylabel('Latency approximation (s)')
        ax1.set_title('Latency approximation per fused object')
        ax1.plot(sdsm_drops, np.ones(len(sdsm_drops)) * -0.001, '*',linestyle='None', label='Outdated detections (Ignored)')
        ax1.axhline(y=error_threshold_to_pass_seconds, color='red', linestyle='dashed',linewidth=2, label='Error Threshold')
        ax1.axhline(y=stats['mean'], color='green', linestyle='dashdot',linewidth=2, label='Mean Latency')
        ax1.axhline(y=stats['median'], color='orange', linestyle='dotted',linewidth=2, label='Median Latency')
        ax1.grid(True)
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(incoming_sdsm_time_in_seconds,objects_count, linestyle='solid',marker='o',linewidth=2, label='SDSM Objects count per message')
        ax2.plot(msg_timestamps,track_objects_count, linestyle='dashed',linewidth=2,marker='*', label='Track Objects count per message')
        ax2.set_title('Object count per message')
        ax2.set_ylabel('Object Count')
        ax2.set_xlabel('Message Receipt Timestamp (s)')
        ax2.grid(True)
        ax2.legend()

        # Save results
        if save_stats_dir:
            stats_full_path = save_stats_dir / f"sdsm_approximation_latency.json"
            with open(stats_full_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {stats_full_path}")

        if save_data_dir:
            np.savez(
                save_data_dir / f"sdsm_approximation_latency_data.npz",
                timestamps=msg_timestamps,
                latency=approximation_latency_in_s,
                stats=stats,
            )
            print(f"Data saved to: {save_data_dir}")

        if save_plot_dir:
            plt.savefig(save_plot_dir / f"sdsm_approximation_latency_analysis.png", dpi=300)
            print(f"Plot saved to: {save_plot_dir}")
        else:
            plt.show()

        return (is_passed, stats, plt.gcf(), msg_timestamps)

    except Exception as e:
        print(f"Error extracting data for SDSM detection analysis: {e}")
        return False, {}, None, []


def detect_gap_ranges(timestamps, gap_threshold=0.1, buffer=0.00):
    """
    Detect gap ranges from timestamp differences

    Parameters:
    - timestamps: array of timestamps
    - gap_threshold: minimum gap size to consider
    - buffer: extra time to add before/after each gap

    Returns:
    - list of (start_time, end_time) tuples for gaps
    """
    diffs = np.diff(timestamps)
    large_gaps = diffs > gap_threshold
    print("Gap threshold: " + str(gap_threshold))
    gap_indices = np.where(large_gaps)[0]

    ranges_to_remove = []

    for gap_idx in gap_indices:
        # Gap is between timestamps[gap_idx] and timestamps[gap_idx + 1]
        gap_start = timestamps[gap_idx] - buffer      # Start a bit before the gap
        gap_end = timestamps[gap_idx + 1] + buffer    # End a bit after the gap

        ranges_to_remove.append((gap_start, gap_end))
        gap_size = diffs[gap_idx]

    return ranges_to_remove


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all cooperative perception analysis on multiple MCAP files in a given directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MCAP files to analyze",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for saving analysis results (optional)",
        default=None,
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        run_all_analysis(
            args.input_dir, analyze_mcap_file_for_cp_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
