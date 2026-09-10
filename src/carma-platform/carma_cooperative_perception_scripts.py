from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict
from run_all_analysis import run_all_analysis
import argparse
import argcomplete
from parse_ros2_bags import open_bagfile, extract_mcap_data
from utils import calculate_error_statistics, print_stats, align_time_series, parse_kafka_log_records
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
INCOMING_MESSAGE_TOPIC = "/hardware_interface/comms/inbound_binary_msg"
INCOMING_SDSM_TOPIC = "/message/incoming_sdsm"
FUSED_SDSM_OBJECTS_TOPIC = "/environment/fused_external_objects"

# DT-05: Median latency from object detection (by the infrastructure sensor) to CARMA Platform receiving it
# in an SDSM should be less than 0.3 s
DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_IN_S = 0.3

# CP-02: Raw detection to SDSM drop rate should be less than 2%
SDSM_DROP_RATE_THRESHOLD_PCT = 2.0
# Object times are decoded to within ~1ms (sdsm_time_stamp only carries millisecond
# resolution), so a real match lands well inside this; anything wider risks bridging over
# a genuine drop (raw detections are ~100ms apart in the sample data).
SDSM_DROP_RATE_MATCH_TOLERANCE_IN_S = 0.05

# CP-03: SDSMs broadcast by the RSU (RSU pcap) that CARMA Platform never receives
# (/hardware_interface/comms/inbound_binary_msg) should be less than 2%
RSU_SDSM_TRANSMISSION_DROP_RATE_THRESHOLD_PCT = 2.0
# pcap <-> mcap correlator (correlate_pcap_mcap.py) - j2735-pcap is a hyphenated directory, not a package
J2735_PCAP_TOOLS_DIR = Path(__file__).resolve().parent.parent / "j2735-pcap"



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
            analysis_stats["run_sdsm_latency_analysis"] = bool(is_passed)
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_sdsm_latency_analysis: {e}"
            )
            analysis_stats["run_sdsm_latency_analysis"] = None


        try:
            is_passed, _, _, _ = run_sdsm_approximation_latency_analysis(
                mcap_path,
                0.2,
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
    Analyzes latency from each object's detection by the infrastructure sensor until CARMA Platform receives
    it in an SDSM (DT-05).

    An SDSM object's detection time is its SDSM's sdsm_time_stamp minus the object's measurement_time_offset
    (carma_v2x_msgs/MeasurementTimeOffset, already in seconds): the offset is how long before the SDSM was
    generated that the object was detected. Receipt time is when INCOMING_SDSM_TOPIC was recorded on the
    vehicle, so the latency spans the whole detection -> SDSM generation -> broadcast -> receipt pipeline, and
    assumes the infrastructure and vehicle clocks are synchronized.

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
        plt.close('all')

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
        # DDateTime offset is the local time's offset from UTC in minutes, so subtract it to get UTC
        epoch_times = np.array(epoch_times) * 1e9 - sdsm_offset*60*1e9 # Convert to nanoseconds

        incoming_object_timestamp_ns = []
        topic_timestamp_ns = []
        for message_time, encoded_msg_time, objs in zip(msg_timestamps, epoch_times, sdsm_objects):

            if len(objs) == 0:
                print(f"Warning: SDSM message at {message_time} has no objects. Skipping.")
                continue
            # Detection time of each object: measurement_time_offset (s) is how long before the SDSM it was detected
            for obj in objs:
                object_creation_time = encoded_msg_time - obj.detected_object_common_data.measurement_time.measurement_time_offset*1e9
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
        print_stats(stats, "DT-05: Detection to CARMA Platform SDSM Receipt Latency Analysis",decimal_places = 10)

        # Pass or no pass
        if threshold_percentile == None:
            is_passed = float(stats["median"]) < error_threshold_to_pass_seconds
        elif threshold_percentile > 0:
            is_passed = np.percentile(latency, threshold_percentile) < error_threshold_to_pass_seconds

        # Same x-axis as the other dt_wz plots: seconds since the MCAP recording started
        _, _, global_start_time_ns = open_bagfile(str(mcap_path))
        plt.figure(figsize=(12,6))
        plt.xlabel('Time Since Start of Recording (seconds)')
        plt.ylabel('Latency (s)')
        plt.title('DT-05: Latency from Object Detection to CARMA Platform SDSM Receipt')
        plt.plot((topic_timestamp_ns - global_start_time_ns) / 1e9, latency, label="Detection to SDSM Receipt Latency (s)", marker='o', color='blue', linestyle="solid", linewidth=2)
        plt.axhline(y=error_threshold_to_pass_seconds, color='red', linestyle='dashed', label=f'Threshold ({error_threshold_to_pass_seconds} s)',linewidth=2)
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
        plt.close('all')

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
        print_stats(stats, "SDSM Latency Approximation Analysis",decimal_places = 10)

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

        # Ensure proper formatting of output plots
        plt.tight_layout()

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


def _decode_sdsm_object_detections(mcap_path, start_time=None, end_time=None):
    """
    Extracts every detected object reported across all SDSM messages in an MCAP file,
    decoding each object's absolute detection time as the message's sdsm_time_stamp minus
    the object's measurement_time_offset (carma_v2x_msgs/MeasurementTimeOffset is already in
    seconds - how long before the message was generated the object was actually measured).

    Args:
        mcap_path: Path to MCAP file
        start_time: Time to start the analysis (seconds from start of recording)
        end_time: Time to end the analysis (seconds from start of recording)

    Returns:
        Tuple containing:
        - object_ids: Array of detected_id.object_id values, one per detected object instance
        - object_times_sec: Array of each object's absolute detection time (epoch seconds)
        - global_start_time_ns: Recording start time (ns since epoch), for placing other time
            sources (e.g. a Kafka log's absolute timestamps) on the same time base
        - msg_times_sec: Array of each SDSM message's receipt time (epoch seconds)
    """
    _, _, global_start_time_ns = open_bagfile(str(mcap_path))

    extracted_data = extract_mcap_data(
        mcap_path,
        [INCOMING_SDSM_TOPIC],
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            INCOMING_SDSM_TOPIC: lambda msg: (
                msg.sdsm_time_stamp.year.year,
                msg.sdsm_time_stamp.month.month,
                msg.sdsm_time_stamp.day.day,
                msg.sdsm_time_stamp.hour.hour,
                msg.sdsm_time_stamp.minute.minute,
                msg.sdsm_time_stamp.second.millisecond,
                msg.sdsm_time_stamp.offset.offset_minute,
                msg.objects.detected_object_data,
            ),
        },
    )
    msg_times_sec, extracted_data = extracted_data[INCOMING_SDSM_TOPIC]
    msg_times_sec = (msg_times_sec * 1e9 + global_start_time_ns) / 1e9

    sdsm_year = extracted_data[:, 0].astype(int)
    sdsm_month = extracted_data[:, 1].astype(int)
    sdsm_day = extracted_data[:, 2].astype(int)
    sdsm_hour = extracted_data[:, 3].astype(int)
    sdsm_minute = extracted_data[:, 4].astype(int)
    sdsm_millisecond = extracted_data[:, 5].astype(int)
    sdsm_offset = extracted_data[:, 6].astype(int)
    sdsm_objects = extracted_data[:, 7]

    object_ids = []
    object_times_sec = []
    for y, m, d, h, mi, ms, off, objs in zip(
        sdsm_year, sdsm_month, sdsm_day, sdsm_hour, sdsm_minute, sdsm_millisecond, sdsm_offset, sdsm_objects
    ):
        dt = datetime(
            int(y), int(m), int(d), int(h), int(mi), int(ms // 1000),
            microsecond=(int(ms) % 1000) * 1000, tzinfo=timezone.utc,
        )
        msg_time_sec = dt.timestamp() + off * 60

        for obj in objs:
            common_data = obj.detected_object_common_data
            object_ids.append(common_data.detected_id.object_id)
            object_times_sec.append(msg_time_sec - common_data.measurement_time.measurement_time_offset)

    return np.array(object_ids), np.array(object_times_sec), global_start_time_ns, msg_times_sec


def _match_detections_to_sdsm(raw_times_sec, sdsm_times_sec, match_tolerance_sec):
    """
    Greedily matches each raw detection time to the closest not-yet-used SDSM detection
    time (of the same object ID - both arrays are expected to already be filtered down to
    a single object ID) within match_tolerance_sec. Both inputs are expected to be roughly
    periodic and time-ordered, so a single forward pass over each array is sufficient.

    Args:
        raw_times_sec: Sorted array of raw detection times (epoch seconds) for one object ID
        sdsm_times_sec: Sorted array of SDSM-reported detection times (epoch seconds) for the same object ID
        match_tolerance_sec: Maximum time difference to consider a raw/SDSM pair matched

    Returns:
        Tuple containing:
        - matched_count: Number of raw detections matched to an SDSM detection
        - dropped_times_sec: Array of raw detection times with no matching SDSM detection
    """
    sdsm_idx = 0
    n_sdsm = len(sdsm_times_sec)
    matched_count = 0
    dropped_times_sec = []

    for raw_time in raw_times_sec:
        while sdsm_idx < n_sdsm and sdsm_times_sec[sdsm_idx] < raw_time - match_tolerance_sec:
            sdsm_idx += 1

        if sdsm_idx < n_sdsm and abs(sdsm_times_sec[sdsm_idx] - raw_time) <= match_tolerance_sec:
            matched_count += 1
            sdsm_idx += 1
        else:
            dropped_times_sec.append(raw_time)

    return matched_count, np.array(dropped_times_sec)


def run_sdsm_detection_drop_rate_analysis(
    mcap_path,
    detection_log_path,
    max_drop_rate_pct=SDSM_DROP_RATE_THRESHOLD_PCT,
    match_tolerance_sec=SDSM_DROP_RATE_MATCH_TOLERANCE_IN_S,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
    ax=None,
):
    """
    CP-02: Verifies that fewer than max_drop_rate_pct of the raw object detections logged by
    v2xhub (the v2xhub_sim_sensor_detected_object Kafka topic) fail to appear in any SDSM
    broadcast received by the vehicle (INCOMING_SDSM_TOPIC).

    Each raw detection's "objectId" is the same temporary ID reported as an SDSM object's
    detected_id, so raw detections and SDSM-reported detections are grouped by that ID, then
    each raw detection is matched to the closest not-yet-matched SDSM detection of the same ID
    within match_tolerance_sec (see _match_detections_to_sdsm). Raw detections with no match are
    counted as dropped between the sensor and the vehicle's outgoing SDSM.

    Args:
        mcap_path: Path to MCAP file containing INCOMING_SDSM_TOPIC
        detection_log_path: Path to the v2xhub_sim_sensor_detected_object Kafka log corresponding
            to mcap_path
        max_drop_rate_pct: Maximum allowed percentage of raw detections missing from SDSM for the
            analysis to pass (default: 2.0)
        match_tolerance_sec: Maximum time difference to consider a raw detection matched to an
            SDSM-reported detection of the same object ID (default: 0.15 s)
        start_time: Time to start the analysis (seconds from start of recording)
        end_time: Time to end the analysis (seconds from start of recording)
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
        ax: Optional matplotlib Axes to draw into (e.g. one panel of a combined figure). When given,
            the caller owns the figure's layout and saving, so save_plot_dir is ignored.

    Returns:
        Tuple containing:
        - is_passed: Boolean - True if the overall drop rate is below max_drop_rate_pct
        - stats: Dictionary with overall and per-object-ID drop rate statistics
        - figure: Matplotlib figure object
        - dropped_detections: Array of (object_id, epoch_time_sec) for unmatched raw detections

    Deps:
        Topics: [/message/incoming_sdsm]
        Msgs: carma_v2x_msgs/msg/SensorDataSharingMessage
    """
    own_figure = ax is None
    if own_figure:
        plt.close('all')

    sdsm_object_ids, sdsm_object_times_sec, global_start_time_ns, msg_times_sec = _decode_sdsm_object_detections(
        mcap_path, start_time, end_time
    )
    # Same time origin as plot_message_time_intervals's x-axis (seconds since the MCAP
    # recording started), so the two plots can be stacked and compared directly.
    recording_origin_sec = global_start_time_ns / 1e9

    # Bound raw detections to this mcap's (possibly start_time/end_time restricted) analysis
    # window - a Kafka log is often a running record spanning many separate recordings/days.
    recording_start_sec = np.min(msg_times_sec)
    recording_end_sec = np.max(msg_times_sec)

    detection_records = parse_kafka_log_records(detection_log_path)
    raw_object_ids = []
    raw_times_sec = []
    for record in detection_records:
        time_sec = record.get("timestamp", record["create_time_ms"]) / 1e3
        if recording_start_sec - match_tolerance_sec <= time_sec <= recording_end_sec + match_tolerance_sec:
            raw_object_ids.append(record.get("objectId"))
            raw_times_sec.append(time_sec)
    raw_object_ids = np.array(raw_object_ids)
    raw_times_sec = np.array(raw_times_sec)

    if len(raw_times_sec) == 0:
        print(f"Error: No raw detections found in {detection_log_path} within the mcap's recording window")
        return False, {}, None, []

    total_matched = 0
    dropped_detections = []
    per_object_stats = {}
    for object_id in np.unique(raw_object_ids):
        object_mask = raw_object_ids == object_id
        object_raw_times = np.sort(raw_times_sec[object_mask])
        object_sdsm_times = np.sort(sdsm_object_times_sec[sdsm_object_ids == object_id])

        matched_count, dropped_times_sec = _match_detections_to_sdsm(
            object_raw_times, object_sdsm_times, match_tolerance_sec
        )
        total_matched += matched_count
        dropped_detections.extend((int(object_id), t) for t in dropped_times_sec)

        per_object_stats[str(int(object_id))] = {
            "raw_detections": len(object_raw_times),
            "matched": matched_count,
            "dropped": len(dropped_times_sec),
            "drop_rate_pct": (len(dropped_times_sec) / len(object_raw_times)) * 100,
        }

    total_raw = len(raw_times_sec)
    total_dropped = total_raw - total_matched
    drop_rate_pct = (total_dropped / total_raw) * 100
    is_passed = bool(drop_rate_pct < max_drop_rate_pct)

    print(f"\n=== CP-02: FLIR Detection to CARMA Platform SDSM Drop Rate Analysis ===")
    print(f"Total raw detections: {total_raw}")
    print(f"Matched to an SDSM broadcast: {total_matched}")
    print(f"Dropped (no matching SDSM detection): {total_dropped}")
    print(f"Drop rate: {drop_rate_pct:.2f}% (threshold: < {max_drop_rate_pct}%)")
    print(f"Result: {'PASSED' if is_passed else 'FAILED'}")

    stats = {
        "total_raw_detections": total_raw,
        "total_matched": total_matched,
        "total_dropped": total_dropped,
        "drop_rate_pct": float(drop_rate_pct),
        "max_drop_rate_pct": max_drop_rate_pct,
        "match_tolerance_sec": match_tolerance_sec,
        "is_passed": is_passed,
        "per_object_id": per_object_stats,
    }

    # Visualize matched vs. dropped raw detections over time
    dropped_times_sec = np.array([t for _, t in dropped_detections])
    matched_times_sec = np.setdiff1d(raw_times_sec, dropped_times_sec)

    if own_figure:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    ax.plot(
        matched_times_sec - recording_origin_sec, np.ones(len(matched_times_sec)),
        ".", color="green", markersize=4, label=f"Received in SDSM ({total_matched})"
    )
    if len(dropped_times_sec) > 0:
        ax.plot(
            dropped_times_sec - recording_origin_sec, np.ones(len(dropped_times_sec)),
            "x", color="red", markersize=6, label=f"Not received ({total_dropped})"
        )
    ax.set_title(
        "CP-02: FLIR Camera Detections Received as SDSM by CARMA Platform\n"
        f"(end to end via V2X-Hub, TENA and RSU broadcast) - Drop Rate {drop_rate_pct:.2f}%"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    if own_figure:
        fig.tight_layout()

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "sdsm_detection_drop_rate.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {stats_full_path}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "sdsm_detection_drop_rate_data.npz",
            raw_object_ids=raw_object_ids,
            raw_times_sec=raw_times_sec,
            dropped_detections=np.array(dropped_detections, dtype=object),
            stats=stats,
        )
        print(f"Data saved to: {save_data_dir}")

    if own_figure:
        if save_plot_dir:
            save_plot_dir = Path(save_plot_dir)
            save_plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_plot_dir / "sdsm_detection_drop_rate_analysis.png", dpi=300)
            print(f"Plot saved to: {save_plot_dir}")
        else:
            plt.show()

    return is_passed, stats, fig, np.array(dropped_detections, dtype=object)


def _import_pcap_mcap_correlator():
    """Import the j2735-pcap correlator modules on first use, so their extra dependencies
    (tshark, pycrate, mcap, mcap_ros2) are only needed by analyses that use them."""
    if str(J2735_PCAP_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(J2735_PCAP_TOOLS_DIR))
    import correlate_j2735_latency
    import correlate_pcap_mcap
    return correlate_j2735_latency, correlate_pcap_mcap


@lru_cache(maxsize=None)
def _extract_pcap_messages_by_direction(pcap_path):
    """J2735 messages in a pcap by direction ("incoming"/"outgoing"/"other"), as correlator message dicts.
    Cached since one session's pcaps are checked against every MCAP in the session."""
    pcap_base, _ = _import_pcap_mcap_correlator()
    messages_by_direction, _ = pcap_base.extract_messages(pcap_path)
    return messages_by_direction


def extract_pcap_messages(pcap_path, direction, msg_type):
    """Timestamped messages of one J2735 type and direction in a pcap (e.g. an RSU's outgoing SDSMs or an
    OBU's outgoing BSMs), as correlator message dicts."""
    return [
        message for message in _extract_pcap_messages_by_direction(str(pcap_path))[direction]
        if message["msg_type"] == msg_type and message["timestamp"] is not None
    ]


def run_rsu_sdsm_transmission_drop_rate_analysis(
    mcap_path,
    rsu_pcap_paths,
    max_drop_rate_pct=RSU_SDSM_TRANSMISSION_DROP_RATE_THRESHOLD_PCT,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_plot_dir=None,
    ax=None,
):
    """
    CP-03: Verifies that fewer than max_drop_rate_pct of the SDSMs broadcast by the RSU fail to be
    received by CARMA Platform, i.e. the drop rate of the RSU -> OBU -> CARMA Platform transmission.

    Uses the j2735-pcap correlator (correlate_pcap_mcap.py): the RSU pcap's outgoing SDSMs are matched
    byte-for-byte against the J2735 SDSMs CARMA Platform received on INCOMING_MESSAGE_TOPIC, each to the
    closest-in-time unconsumed identical payload within the correlator's match window. Broadcasts that
    match nothing are dropped; matches later than the correlator's drop latency threshold (200 ms) are
    counted as received late rather than dropped. Broadcasts outside the MCAP's recording window can't
    be checked and are excluded.

    Args:
        mcap_path: Path to MCAP file containing INCOMING_MESSAGE_TOPIC
        rsu_pcap_paths: RSU pcaps to take SDSM broadcasts from. Only broadcasts within this MCAP's analysis
            window are used, so every RSU pcap in a session can be passed for every MCAP.
        max_drop_rate_pct: Maximum allowed percentage of RSU SDSM broadcasts not received for the analysis to
            pass (default: 2.0)
        start_time: Time to start the analysis (seconds from start of recording)
        end_time: Time to end the analysis (seconds from start of recording)
        save_stats_dir: Directory to save analysis stats
        save_plot_dir: Directory to save generated plots
        ax: Optional matplotlib Axes to draw into (e.g. one panel of a combined figure). When given,
            the caller owns the figure's layout and saving, so save_plot_dir is ignored.

    Returns:
        Tuple containing:
        - is_passed: Boolean - True if the drop rate is below max_drop_rate_pct
        - stats: Dictionary with drop rate and receive latency statistics
        - figure: Matplotlib figure object

    Deps:
        Topics: [/hardware_interface/comms/inbound_binary_msg]
        Msgs: carma_driver_msgs/msg/ByteArray
        Tools: tshark, and the j2735-pcap requirements (pycrate, mcap, mcap-ros2-support)
    """
    pcap_base, pcap_mcap = _import_pcap_mcap_correlator()

    # Same time origin as the other dt_wz plots' x-axis (seconds since the MCAP recording started)
    _, _, global_start_time_ns = open_bagfile(str(mcap_path))
    recording_origin_sec = global_start_time_ns / 1e9
    window_start_sec = recording_origin_sec + start_time if start_time is not None else -np.inf
    window_end_sec = recording_origin_sec + end_time if end_time is not None else np.inf

    broadcasts = [
        message
        for rsu_pcap_path in rsu_pcap_paths
        for message in extract_pcap_messages(rsu_pcap_path, "outgoing", "SDSM")
        if window_start_sec <= message["timestamp"] <= window_end_sec
    ]
    if not broadcasts:
        raise ValueError(f"No RSU SDSM broadcasts in {[str(p) for p in rsu_pcap_paths]} within {mcap_path}'s analysis window")

    received_messages = pcap_mcap.extract_mcap_binary_messages(mcap_path)["inbound"]
    received, dropped, received_late, out_of_window = pcap_mcap.correlate_across_boundary(
        broadcasts, received_messages, pcap_base.DROP_LATENCY_THRESHOLD_MS
    )

    total_checked = len(received) + len(received_late) + len(dropped)
    if total_checked == 0:
        raise ValueError(f"No RSU SDSM broadcasts fall within {mcap_path}'s recording window")
    drop_rate_pct = len(dropped) / total_checked * 100
    is_passed = bool(drop_rate_pct < max_drop_rate_pct)

    print(f"\n=== CP-03: RSU SDSM Broadcast to CARMA Platform Receipt Drop Rate Analysis ===")
    print(f"RSU SDSM broadcasts checked: {total_checked} "
          f"(plus {len(out_of_window)} outside the mcap's recording window, not counted)")
    print(f"Received by CARMA Platform: {len(received)}")
    print(f"Received late (> {pcap_base.DROP_LATENCY_THRESHOLD_MS} ms): {len(received_late)}")
    print(f"Dropped (never received): {len(dropped)}")
    print(f"Drop rate: {drop_rate_pct:.2f}% (threshold: < {max_drop_rate_pct}%)")
    print(f"Result: {'PASSED' if is_passed else 'FAILED'}")

    stats = {
        "rsu_pcaps": [str(p) for p in rsu_pcap_paths],
        "total_broadcasts_checked": total_checked,
        "total_received": len(received),
        "total_received_late": len(received_late),
        "total_dropped": len(dropped),
        "total_outside_recording_window": len(out_of_window),
        "drop_rate_pct": float(drop_rate_pct),
        "max_drop_rate_pct": max_drop_rate_pct,
        "late_threshold_ms": pcap_base.DROP_LATENCY_THRESHOLD_MS,
        "receive_latency_ms": pcap_base.summarize(received) if received else None,
        "is_passed": is_passed,
    }

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    for events, marker, color, markersize, label in (
        (received, ".", "green", 4, f"Received ({len(received)})"),
        (received_late, "^", "orange", 6, f"Received late > {pcap_base.DROP_LATENCY_THRESHOLD_MS} ms ({len(received_late)})"),
        (dropped, "x", "red", 6, f"Not received ({len(dropped)})"),
    ):
        if events:
            times = np.array([event[0] for event in events]) - recording_origin_sec
            ax.plot(times, np.ones(len(times)), marker, color=color, markersize=markersize, linestyle="none", label=label)
    ax.set_title(
        "CP-03: SDSMs Broadcast by RSU Received by CARMA Platform\n"
        f"(RSU pcap -> OBU -> {INCOMING_MESSAGE_TOPIC}) - Drop Rate {drop_rate_pct:.2f}%"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    if own_figure:
        fig.tight_layout()

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "rsu_sdsm_transmission_drop_rate.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {stats_full_path}")

    if own_figure:
        if save_plot_dir:
            save_plot_dir = Path(save_plot_dir)
            save_plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_plot_dir / "rsu_sdsm_transmission_drop_rate_analysis.png", dpi=300)
            print(f"Plot saved to: {save_plot_dir}")
        else:
            plt.show()

    return is_passed, stats, fig


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
        