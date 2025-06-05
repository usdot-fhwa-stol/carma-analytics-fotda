from parse_ros2_bags import extract_mcap_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import json
from utils import calculate_error_statistics, print_stats
import re
import os
from datetime import datetime
from pathlib import Path

STD_DEV_LABEL_STRING = "±1 Std Dev"
TIME_SECONDS_LABEL_STRING = "Time (seconds)"
INCOMING_GEOFENCE_CONTROL_TOPIC = "/message/incoming_geofence_control"
OUTGOING_GEOFENCE_REQUEST_TOPIC = "/message/outgoing_geofence_request"
OUTGOING_MOBILITY_OPERATION_TOPIC = "/message/outgoing_mobility_operation"

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

def process_cc_logs_for_tcr_tcm_data(
    cc_data_path,
    log_date,
    max_delay,
    expected_rate,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None
):
    """
    Reads .log files to process the TrafficControlRequests and TrafficControlMessages from Carma Cloud.
    Assumes every file in the directory is a .log file and that all .log files were taken on the same day.

    Args:
        cc_data_path: Path to directory containing cc.log files
        log_data: String with the day the logs were taken to convert to log times to Unix Epoch Time
        max_delay: Maximum allowed delay between TCR recipt and TCM broadcasted
        expected_rate: Rate in Hz at which any TCM is expected to be broadcasted

    Returns:
        all_results: Dictionary of dictionaries containing:
            'reqid': ID of the TCR received and TCM(s) broadcasted
                'tcr_time': Time the TCR was received
                'first_tcm_time': Time the first TCM was broadcasted
                'response_delay': Time between receiving the TCR and broadcasting the first TCM
                'tcm_1': ID of a TCM broadcasted in response to the TCR
                    'timestamps': Array of times this exact TCM was broadcasted
                    'msgnum': Message number of the TCM
                    'count': Number of times that TCM was broadcasted
                    'rate': Rate at which that TCM was broadcasted
                'tcm_2': ...
    """
    all_results = {}

    # Formats to determine reqid, time of log, and type of log message in a file
    time_pattern = re.compile(r'\[DEBUG (\d{2}:\d{2}:\d{2}\.\d{3})')
    reqid_pattern = r'<reqid>([A-F0-9]+)</reqid>'
    tcmid_pattern = r'<id>([a-fA-F0-9]+)</id>'
    msgnum_pattern = r'<msgnum>(\d+)</msgnum>'
    message_type_pattern = r'<(TrafficControlRequest|TrafficControlMessage)'

    # Parses every file in the passed in directory
    for filename in os.listdir(cc_data_path):
        file_path = os.path.join(cc_data_path, filename)
        if os.path.isfile(file_path):
            print(f"Opening file {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    if 'TrafficControlRequest' not in line and 'TrafficControlMessage' not in line:
                        continue

                    # Extract timestamp
                    time_match = re.search(time_pattern, line)
                    if not time_match:
                        continue
                    timestamp_str = time_match.group(1)
                    timestamp = datetime.strptime(f"{log_date} {timestamp_str}", "%Y-%m-%d %H:%M:%S.%f")
                    timestamp_epoch = timestamp.timestamp()

                    # Extract reqid
                    reqid_match = re.search(reqid_pattern, line)
                    if not reqid_match:
                        continue
                    reqid = reqid_match.group(1)

                    # Determine if line contains TCR or TCM
                    msg_match = re.search(message_type_pattern, line)
                    if not msg_match:
                        continue
                    msg_type = msg_match.group(1)

                    # Initialize reqid entry
                    if reqid not in all_results:
                        all_results[reqid] = {
                            'tcr_time': None,
                            'first_tcm_time': None,
                            'response_delay': None
                        }

                    if msg_type == 'TrafficControlRequest' and all_results[reqid]['tcr_time'] is None:
                        all_results[reqid]['tcr_time'] = timestamp_epoch
                    elif msg_type == 'TrafficControlMessage':
                        tcmid_match = re.search(tcmid_pattern, line)
                        if not tcmid_match:
                            continue
                        tcmid = tcmid_match.group(1)

                        if tcmid not in all_results[reqid]:
                            all_results[reqid][tcmid] = {
                                "timestamps": [],
                                "msgnum": 0,
                                "count": 0,
                                "rate": 0.0
                            }

                        msgnum_match = re.search(msgnum_pattern, line)
                        if not msgnum_match:
                            continue
                        msgnum = int(msgnum_match.group(1))

                        entry = all_results[reqid][tcmid]
                        entry['timestamps'].append(timestamp_epoch)
                        entry['count'] += 1
                        entry['msgnum'] = msgnum

                        if all_results[reqid]['first_tcm_time'] is None:
                            all_results[reqid]['first_tcm_time'] = timestamp_epoch

                            if all_results[reqid]['tcr_time'] is not None:
                                delay = (all_results[reqid]['first_tcm_time'] - all_results[reqid]['tcr_time'])
                                all_results[reqid]['response_delay'] = delay

                        timestamps = all_results[reqid][tcmid]['timestamps']
                        if len(timestamps) >= 2:
                            duration = (timestamps[-1] - timestamps[0])
                            entry['rate'] = entry['count'] / duration if duration > 0 else float('inf')
                        else:
                            entry['rate'] = 0.0

                        all_results[reqid][tcmid] = entry

    # Pull data to be graphed
    req_times = []
    response_delays = []

    broadcast_rates = []
    broadcast_times = []


    for reqid, data in all_results.items():
        req_time = data['tcr_time']
        delay = data.get('response_delay')

        if req_time is not None and delay is not None:
            req_times.append(req_time)
            response_delays.append(delay)

        for msg_id, msg_data in data.items():
            if msg_id in {'tcr_time', 'first_tcm_time', 'response_delay'}:
                continue

            timestamps = msg_data.get('timestamps', [])
            rate = msg_data.get('rate', 0)

            if timestamps:
                first_msg_time = min(timestamps)
                broadcast_times.append(first_msg_time)
                broadcast_rates.append(rate)

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    ax1.plot(req_times, response_delays, color='blue', label='Response Delay')
    ax1.axhline(max_delay, color='green', linestyle='--', label=f"Maximum Allowed Delay: {max_delay} s")
    ax1.set_ylabel("Response Delay (s)")
    ax1.set_title("(FWZ-4) TCR Receipt to TCM Response Delay Over Time")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(broadcast_times, broadcast_rates, color='red', label='Broadcast Rate')
    ax2.axhline(expected_rate, color='orange', linestyle='--', label=f"Expected Broadcast Rate: {expected_rate} Hz")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Broadcast Rate (Hz)")
    ax2.set_title("(FWZ-19) TCM Broadcast Rate Over Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_plot_dir:
        save_plot_dir = Path(save_plot_dir)
        save_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_dir / "cc_TCM_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    # Run statistics on delay and rate
    response_delays = np.array(response_delays)
    delay_stats = calculate_error_statistics(response_delays)
    broadcast_rates = np.array(broadcast_rates)
    rate_stats = calculate_error_statistics(broadcast_rates)

    # Print statistics
    print_stats(delay_stats, "CC TCM Delay Statistics")
    print_stats(rate_stats, "CC TCM Rate Statistics")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "cc_TCM_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(delay_stats, f, indent=2)
            json.dump(rate_stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    # Save data if requested
    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "cc_TCM_data.npz",
            all_results=all_results,
            delay_stats=delay_stats,
            rate_stats=rate_stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    return all_results, plt.gcf()

def check_cc_response_delay(all_cc_data, expected_delay, save_stats_dir, save_data_dir):
    """
    Checks the delay between Carma Cloud receiving a TCR from the vehicle and sending a TCM

    Args:
        all_cc_data: Dictionary of dictionaries containing:
            'reqid': ID of the TCR received and TCM(s) broadcasted
                'tcr_time': Time the TCR was received
                'first_tcm_time': Time the first TCM was broadcasted
                'response_delay': Time between receiving the TCR and broadcasting the first TCM
                'tcm_1': ID of a TCM broadcasted in response to the TCR
                    'timestamps': Array of times this exact TCM was broadcasted
                    'msgnum': Message number of the TCM
                    'count': Number of times that TCM was broadcasted
                    'rate': Rate at which that TCM was broadcasted
                'tcm_2': ...
        expected_delay: Maximum expected delay between TCR receipt and sending TCM

    Returns:
        is_successful: Boolean - True if all TCR -> TCM delays are within expected_delay
    """
    is_successful = True
    delays = []
    failed_delays = []

    for reqid in all_cc_data:
        data = all_cc_data[reqid]
        delays.append(data['response_delay'])
        if data['response_delay'] > expected_delay:
            is_successful = False
            failed_delays.append((reqid, data['response_delay']))

    if is_successful:
        print(f"FWZ-4 Succeeded: All TCR's received had a TCM broadcasted within {expected_delay} seconds.")
    else:
        print(f"FWZ-4: Failed - {len(failed_delays)} TCMs were broadcasted more than {expected_delay} seconds after TCR receipt.")

    delays = np.array(delays)
    stats = calculate_error_statistics(delays)
    print_stats(stats, 'CC TCM Broadcast Delay Statistics')

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "cc_tcm_broadcast_delay_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "cc_tcm_broadcast_delay_data.npz",
            delays=delays,
            failed_delays=failed_delays,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    return is_successful

def check_tcm_acknowledgement_delay(
    mcap_path,
    max_delay,
    save_stats_dir=None,
    save_data_dir=None,
    save_plots_dir=None
):
    """
    Check that all TCMs received by the vehicle are acknowledged within max_delay

    Args:
        mcap_path: Path to MCAP file
        max_delay: Maximum amount of time vehicle can take to acknowledged TCM

    Returns:
        is_successful: Boolean - True if all TCMs are acknowledged within max_delay
        tcm_acknowledgements: List of tuples containing
            key[0]: reqid of the TCM
            key[1]: msgnum of the TCM
            tcm_time: time the TCM was received
            ack_time: time the TCM was acknowledged
    """
    topics = [INCOMING_GEOFENCE_CONTROL_TOPIC, OUTGOING_MOBILITY_OPERATION_TOPIC]

    received_tcms = {}
    acknowledged_tcms = {}

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        field_extractors={
            INCOMING_GEOFENCE_CONTROL_TOPIC: lambda msg: msg.tcm_v01,
            OUTGOING_MOBILITY_OPERATION_TOPIC: lambda msg: msg
        }
    )

    # Get the time that each TCM was received
    tcm_timestamps, tcms = extracted_data[topics[0]]
    for timestamp, tcm in zip(tcm_timestamps, tcms):
        reqid_hex = ''.join(f'{byte:02X}' for byte in tcm.reqid.id)
        msgnum = tcm.msgnum
        received_tcms[(reqid_hex, msgnum)] = timestamp

    # Get the time that each TCM was acknowledged
    ack_timestamps, acks = extracted_data[topics[1]]
    for timestamp, ack in zip(ack_timestamps, acks):
        # We only want the Geofence Acknowledgement ones
        if ack.strategy == "carma3/Geofence_Acknowledgement":
            match = re.search(r"traffic_control_id:([0-9A-Fa-f]+),\s*msgnum:(\d+)", ack.strategy_params)
            if match:
                reqid_hex = match.group(1).upper()
                msgnum = int(match.group(2))
                acknowledged_tcms[(reqid_hex, msgnum)] = timestamp

    tcm_acknowledgements = []
    is_successful = True
    times_ack_delays = []
    # Check if every TCM is acknowledged and if so, within max_delay
    for key, tcm_time in received_tcms.items():
        if key in acknowledged_tcms:
            ack_time = acknowledged_tcms[key]
            time_to_ack = ack_time - tcm_time
            times_ack_delays.append((ack_time, time_to_ack))
            if time_to_ack > max_delay:
                is_successful = False
                print(f"TCM reqid: {key[0]} msgnum: {key[1]} wasn't acknowledged by threshold. Was {time_to_ack} s, expected {max_delay} s.")
            # Add tcm acknowledgement tuple to output list
            tcm_acknowledgements.append((key[0], key[1], tcm_time, ack_time))
        else:
            is_successful = False
            print(f"TCM reqid: {key[0]} msgnum: {key[1]} was not acknowledged")

    if is_successful:
        print(f"FWZ-9 Succeeded - All TCMs were acknowledged within {max_delay} s.")
    else:
        print(f"FWZ-9 Failed: Not all TCMs were acknowledged or acknowledged within {max_delay} s.")

    # Create visualizations
    times, ack_delays = zip(*times_ack_delays)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, ack_delays, label='TCM Acknowledgement Delay', color='blue')
    ax.axhline(y=max_delay, linestyle='--', color='gray', label=f'Max Expected TCM Acknowledgement Delay')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acknowledgement Delay (s)')
    ax.set_title('TCM Acknowledgement Delay Over Time')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_plots_dir:
        save_plots_dir = Path(save_plots_dir)
        save_plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plots_dir / "tcm_acknowledgement_analysis.png")
        print(f"\nPlot saved to: {save_plots_dir}")
    else:
        plt.show()

    # Calculate, print, and save statistics
    ack_delays = np.array(ack_delays)
    stats = calculate_error_statistics(ack_delays)
    print_stats(stats, "TCM Acknowledgement Delay Statistics")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "tcm_acknowledgement_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "tcm_acknowledgement_data.npz",
            tcm_acknowledgements=tcm_acknowledgements,
            ack_delays=ack_delays,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")


    return is_successful, tcm_acknowledgements, plt.gcf(), stats

def check_tcm_broadcast_count(all_cc_data, tcm_acknowledgements, expected_broadcasts, save_stats_dir, save_data_dir):
    """
    Verify that each TCM was broadcasted {expected_broadcasts} times or if less than 10, that the vehicle had acknowledged the TCM

    Args:
        all_cc_data: Dictionary of dictionaries containing:
            'reqid': ID of the TCR received and TCM(s) broadcasted
                'tcr_time': Time the TCR was received
                'first_tcm_time': Time the first TCM was broadcasted
                'response_delay': Time between receiving the TCR and broadcasting the first TCM
                'tcm_1': ID of a TCM broadcasted in response to the TCR
                    'timestamps': Array of times this exact TCM was broadcasted
                    'msgnum': Message number of the TCM
                    'count': Number of times that TCM was broadcasted
                    'rate': Rate at which that TCM was broadcasted
                'tcm_2': ...
        tcm_acknowledgements: list of tuples containing (reqid, msgnum, tcm_time, ack_time)
            reqid: reqid of the TCM
            msgnum: msgnum of the TCM
            tcm_time: time the TCM was received
            ack_time: time the TCM was acknowledged
        expected_broadcasts: Number of times a TCM is expected to be broadcasted

    Returns:
        is_successful: Boolean - True if all TCMs were broadcasted the expected number of times, or broadcasted less and acknowledged by the vehicle
    """
    messages_acknowledged = {(reqid, msgnum) for (reqid, msgnum, _, _) in tcm_acknowledgements}
    is_successful = True
    tcm_counts = []

    for req_id, data in all_cc_data.items():
        for key, value in data.items():
            if key in {'tcr_time', 'first_tcm_time', 'response_delay'}:
                continue
            count = value.get('count')
            tcm_counts.append(count)
            msgnum = value.get('msgnum')
            if count > expected_broadcasts:
                is_successful = False
                print(f"FWZ-18 Failed - TCM reqid: {req_id} msgnum: {msgnum} was broadcasted {count} times. {expected_broadcasts} expected.")
            elif count < expected_broadcasts:
                if (req_id, msgnum) not in messages_acknowledged:
                    is_successful = False
                    print(f"FWZ-18 Failed - TCM reqid: {req_id} msgnum: {msgnum} was broadcasted less than {expected_broadcasts} times w/o acknowledgement from CMV")

    if is_successful:
        print(f"FWZ-18 Succeeded - All TCMs were broadcasted {expected_broadcasts} times or less than {expected_broadcasts} times with acknowledgement from CMV")

    tcm_counts = np.array(tcm_counts)
    stats = calculate_error_statistics(tcm_counts)
    print_stats(stats, "TCM Broadcast Count")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "tcm_broadcast_count_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "tcm_broadcast_count_data.npz",
            tcm_counts=tcm_counts,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")


    return is_successful

def check_tcm_broadcast_rate(all_cc_data, tcm_acknowledgements, expected_rate, save_stats_dir, save_data_dir):
    """
    Verifies that all TCMs are broadcasted at the expected rate

    Args:
        all_cc_data: Dictionary of dictionaries containing:
            'reqid': ID of the TCR received and TCM(s) broadcasted
                'tcr_time': Time the TCR was received
                'first_tcm_time': Time the first TCM was broadcasted
                'response_delay': Time between receiving the TCR and broadcasting the first TCM
                'tcm_1': ID of a TCM broadcasted in response to the TCR
                    'timestamps': Array of times this exact TCM was broadcasted
                    'msgnum': Message number of the TCM
                    'count': Number of times that TCM was broadcasted
                    'rate': Rate at which that TCM was broadcasted
                'tcm_2': ...
        tcm_acknowledgements: list of tuples containing (reqid, msgnum, tcm_time, ack_time)
            reqid: reqid of the TCM
            msgnum: msgnum of the TCM
            tcm_time: time the TCM was received
            ack_time: time the TCM was acknowledged
        expected_rate: expected broadcast rate of TCMs in Hz

    Returns:
        is_successful: Boolean - True if all TCMs are broadcasted at expected_rate
    """
    messages_acknowledged = {(reqid, msgnum) for (reqid, msgnum, _, _) in tcm_acknowledgements}
    is_successful = True
    tcm_rates = []

    for req_id, data in all_cc_data.items():
        for key, value in data.items():
            if key in {'tcr_time', 'first_tcm_time', 'response_delay'}:
                continue
            rate = value.get('rate')
            tcm_rates.append(rate)
            msgnum = value.get('msgnum')
            if rate != expected_rate:
                if (req_id, msgnum) not in messages_acknowledged:
                    is_successful = False
                    print(f"TCM reqid: {req_id} msgnum: {msgnum} was broadcasted at {rate} Hz. {expected_rate} Hz expected.")
                else:
                    print(f"TCM reqid: {req_id} msgnum: {msgnum} was broadcasted at {rate} Hz. Acknowledged by vehicle.")

    if is_successful:
        print(f"FWZ-19 Succeeded - All TCMs were broadcasted at {expected_rate} Hz or were acknowledged by the vehicle.")
    else:
        print(f"FWZ-19 Failed - TCMs listed above were not broadcasted at {expected_rate} Hz and were not acknowledged by the vehicle.")

    tcm_rates = np.array(tcm_rates)
    stats = calculate_error_statistics(tcm_rates)
    print_stats(stats, "TCM Broadcast Rate")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "tcm_broadcast_rate_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "tcm_broadcast_rate_data.npz",
            tcm_rates=tcm_rates,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    return is_successful

def check_tcm_response_time(mcap_path, expected_tcr_to_tcm_duration, save_stats_dir, save_data_dir, save_plots_dir):
    """
    Verifies that after sending a TCR, the vehicle receives a TCM from Carma Cloud within a specified duration

    Args:
        mcap_path: Path to the MCAP file
        expected_tcr_to_tcm_duration: expected max duration between sending a TCR to receiving a TCM (sec)

    Returns:
        is_successful: Boolean - True if all TCRs sent have a received TCM within the expected duration
    """
    # reqid_v2x_timestamps: 0 is reqid; 1 is tcr receive time; 2-10 are FIRST tcm tx times for msgnums 0 to 9
    topics = [OUTGOING_GEOFENCE_REQUEST_TOPIC, INCOMING_GEOFENCE_CONTROL_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        field_extractors={
            OUTGOING_GEOFENCE_REQUEST_TOPIC: lambda msg: msg.tcr_v01,
            INCOMING_GEOFENCE_CONTROL_TOPIC: lambda msg: msg.tcm_v01
        }
    )
    tcr_timestamps, tcrs = extracted_data[topics[0]]
    tcm_timestamps, tcms = extracted_data[topics[1]]
    tcm_receipt_delay = []

    tcrs_to_tcms = []
    for timestamp, tcr in zip(tcr_timestamps, tcrs):
        tcr_id_hex = ''.join(f'{b:02X}' for b in tcr.reqid.id)
        tcm_times = []
        for t, tcm in zip(tcm_timestamps, tcms):
            tcm_id_hex = ''.join(f'{b:02X}' for b in tcm.reqid.id)
            if tcm_id_hex == tcr_id_hex:
                tcm_times.append(t)
        tcrs_to_tcms.append((tcr_id_hex, timestamp, tcm_times))

    is_successful = True
    for tcr_id, time, tcm_times in tcrs_to_tcms:
        duration = tcm_times[0] - time
        tcm_receipt_delay.append((time, duration))
        if duration > expected_tcr_to_tcm_duration:
            is_successful = False
            print(f"FWZ-31 Failed: TCM response for TCR reqid: {tcr_id} was received {duration} seconds after being sent. Expected less than {expected_tcr_to_tcm_duration} seconds.")

    if is_successful:
        print(f"FWZ-31 Succeeded: All TCRs sent received TCM response within {expected_tcr_to_tcm_duration} seconds.")

    # Create visualizations
    times, tcm_receipt_delays = zip(*tcm_receipt_delay)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, tcm_receipt_delays, label='TCM Receipt Delay', color='blue')
    ax.axhline(y=expected_tcr_to_tcm_duration, linestyle='--', color='gray', label=f'Max Expected TCR to TCM Delay')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Delay (s)')
    ax.set_title('TCR Broadcast to TCM Receipt Delay over Time')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_plots_dir:
        save_plots_dir = Path(save_plots_dir)
        save_plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plots_dir / "tcm_receipt_analysis.png")
        print(f"\nPlot saved to: {save_plots_dir}")
    else:
        plt.show()

    # Calculate, print, and save statistics
    tcm_receipt_delays = np.array(tcm_receipt_delays)
    stats = calculate_error_statistics(tcm_receipt_delays)
    print_stats(stats, "TCM Receipt Delay Statistics")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "tcm_receipt_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "tcm_receipt_data.npz",
            tcm_receipt_delay=tcm_receipt_delay,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    return is_successful

def main():
    """
    Main function to run the analysis scritps.
    """
    # Example usage of the functions
    mcap_path = "/path/to/your/mcap_file.mcap"


if __name__ == "__main__":
    main()
