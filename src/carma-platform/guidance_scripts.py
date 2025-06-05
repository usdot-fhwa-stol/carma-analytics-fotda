from parse_ros2_bags import open_bagfile, extract_mcap_data
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.spatial import KDTree
import json
from utils import calculate_error_statistics, print_stats, align_time_series
from scipy.spatial.transform import Rotation as r
from bisect import bisect_left

DEG_TO_RAD = 0.0174533
MPH_TO_MPS = 0.44704
STD_DEV_LABEL_STRING = "±1 Std Dev"
TIME_SECONDS_LABEL_STRING = "Time (seconds)"
# ROS Topics Constants
GUIDANCE_ROUTE_TOPIC = "/guidance/route"
GUIDANCE_STATE_TOPIC = "/guidance/state"
GUIDANCE_ROUTE_STATE_TOPIC = "/guidance/route_state"
GUIDANCE_PLAN_TRAJECTORY_TOPIC = "/guidance/plan_trajectory"
GUIDANCE_CONTROL_CMD_TOPIC = "/guidance/ctrl_cmd"

LOCALIZATION_POSE_TOPIC = "/localization/current_pose"

HARDWARE_VEHICLE_STATUS_TOPIC = "/hardware_interface/vehicle_status"
HARDWARE_VEHICLE_TWIST_TOPIC = "/hardware_interface/vehicle/twist"
# Lexus and Freightliners control topics
HARDWARE_PACMOD_STEER_REPORT_TOPIC = "/hardware_interface/as/pacmod/parsed_tx/steer_rpt" #pacmod3_msgs/msg/SystemRptFloat
# Fusion control topics
HARDWARE_DATASPEED_STEER_REPORT_TOPIC = "/hardware_interface/ds_fusion/steering_report" #dbw_mkz_msgs/SteeringReport
# Pacifica control topics
HARDWARE_NEWEAGLE_STEER_REPORT_TOPIC = "/hardware_interface/steering_report" #raptor_dbw_msgs/SteeringReport
# Geofence topics
ACTIVE_GEOFENCE_TOPIC = "/environment/active_geofence"
INCOMING_GEOFENCE_CONTROL_TOPIC = "/message/incoming_geofence_control"
OUTGOING_MOBILITY_OPERATION_TOPIC = "/message/outgoing_mobility_operation"
OUTGOING_GEOFENCE_REQUEST_TOPIC = "/message/outgoing_geofence_request"


def get_engage_time(mcap_path):
    """
    Get the (engage, disengage_time) as a tuple
    Returns last recorded time if no disengaged.
    NOTE: If there are multiple engage operations, it will only take the first engage time as the start_time.
    Args:
        mcap_path: Path to MCAP file
    Deps:
        Topics: [/guidance/state]
    """
    DRIVERS_READY = 2
    ACTIVE = 3
    ENGAGED = 4
    INACTIVE = 5
    ENTER_PARK = 6
    SHUTDOWN = 0

    not_engaged_anymore = {DRIVERS_READY, ACTIVE, INACTIVE, ENTER_PARK, SHUTDOWN}

    topics = [GUIDANCE_STATE_TOPIC]
    extracted_data = extract_mcap_data(
        mcap_path, topics, field_extractors={topics[0]: lambda msg: msg.state}
    )
    timestamps, states = extracted_data[topics[0]]

    start_time = None
    end_time = timestamps[-1]  # pick last available timestamp by default

    for timestamp, state in zip(timestamps, states):
        if state == ENGAGED:
            start_time = timestamp
            break

    if start_time is None:
        raise ValueError("Cannot find CARMA engage time in this recording...")

    for timestamp, state in zip(timestamps, states):
        if timestamp > start_time and state in not_engaged_anymore:
            end_time = timestamp
            break

    print(f"Engage time: {start_time} and disengage time: {end_time}")
    return (start_time, end_time)


def run_crosstrack_analysis(
    mcap_path,
    error_threshold_to_pass_meter=2.0,
    threshold_percentile=None,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes cross trask error from CARMA Platform's internal route logic.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass_meter: Threshold crosstrack error in meters for passing the analysis
        threshold_percentile: Threshold percentile for passing the analysis
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/localization/current_pose]
        Msgs: carma_planning_msgs
    """

    topics = [GUIDANCE_ROUTE_STATE_TOPIC]
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={GUIDANCE_ROUTE_STATE_TOPIC: lambda msg: msg.cross_track},
    )
    timestamps, cross_tracks = extracted_data[topics[0]]

    # Calculate statistics
    stats = calculate_error_statistics(cross_tracks, start_time, end_time)

    # Pass or no pass
    if threshold_percentile == None:
        is_passed = float(stats["median"]) < error_threshold_to_pass_meter
    elif threshold_percentile > 0:
        is_passed = np.percentile(cross_tracks, threshold_percentile) < error_threshold_to_pass_meter

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        timestamps,
        cross_tracks,
        ".",
        markersize=2,
        label="Cross Track Error",
        linewidth=1,
    )
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.axhline(
        y=error_threshold_to_pass_meter,
        color="g",
        linestyle="--",
        label="Crosstrack Error Threshold",
    )
    plt.axhline(y=-error_threshold_to_pass_meter, color="g", linestyle="--")
    plt.fill_between(
        timestamps,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label=STD_DEV_LABEL_STRING,
    )

    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Cross Track Error (m)")
    plt.title("Route State Cross Track Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Print Stats
    print_stats(stats, "Cross Track Error Statistics")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "cross_track_stats_result.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "cross_track_extracted_numpy_data.npz",
            timestamps=timestamps,
            cross_tracks=cross_tracks,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    if save_plot_dir:
        save_plot_dir = Path(save_plot_dir)
        save_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_dir / "cross_track_error_over_time.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (is_passed, stats, plt.gcf(), cross_tracks, timestamps)


def process_actual_path(odom_data):
    """Helper function that processes actual path data."""
    return np.array([[point[0], point[1]] for point in odom_data])


def process_planned_path(traj_plans):
    """Helper function that processes planned path data with duplicate removal."""
    planned_path = []
    last_planned_point = None

    for plan in traj_plans:
        for point in plan:
            if last_planned_point is None:
                planned_path.append(point)
                last_planned_point = point
            else:
                dist = np.linalg.norm(np.array(point) - np.array(last_planned_point))
                if dist > 0.25:  # 0.25m threshold
                    planned_path.append(point)
                    last_planned_point = point
                    break

    return np.array(planned_path)


def run_turn_accuracy_analysis(
    mcap_path,
    error_threshold_to_pass_meter=2.0,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes turn accuracy by comparing actual path to planned trajectory using spline fitting.

    Args:
        mcap_path: Path to MCAP file
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/localization/current_pose, /guidance/plan_trajectory]
        Msgs: carma_planning_msgs
    """
    # Extract actual and planned paths
    actual_path = []
    planned_path = []

    # Extract messages from MCAP
    topics = [LOCALIZATION_POSE_TOPIC, GUIDANCE_PLAN_TRAJECTORY_TOPIC]
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            LOCALIZATION_POSE_TOPIC: lambda msg: (
                msg.pose.position.x,
                msg.pose.position.y,
            ),
            GUIDANCE_PLAN_TRAJECTORY_TOPIC: lambda msg: [
                (p.x, p.y) for p in msg.trajectory_points[1:]
            ],  # Skip first point
        },
    )

    # Process actual path
    timestamps, odom = extracted_data[topics[0]]

    actual_path = process_actual_path(odom)

    # Process planned path with duplicate removal
    # timestamp source here is not too important
    _, traj_plans = extracted_data[topics[1]]

    planned_path = process_planned_path(traj_plans)

    # Fit spline to planned path
    t = np.linspace(0, 1, len(planned_path))
    cs_x = CubicSpline(t, planned_path[:, 0])
    cs_y = CubicSpline(t, planned_path[:, 1])

    # Generate higher resolution points along spline
    t_dense = np.linspace(0, 1, len(planned_path) * 5)  # 5x more points
    spline_points = np.column_stack((cs_x(t_dense), cs_y(t_dense)))

    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(spline_points)

    # Calculate distances from actual path to spline
    distances = []
    for point in actual_path:
        dist, _ = tree.query(point)
        distances.append(dist)

    distances = np.array(distances)

    # Calculate statistics
    stats = calculate_error_statistics(distances, start_time, end_time)

    # Pass or no pass
    is_passed = float(stats["median"]) < error_threshold_to_pass_meter

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot paths
    plt.subplot(2, 1, 1)
    plt.plot(planned_path[:, 0], planned_path[:, 1], label="Planned Path", linewidth=1)
    plt.plot(
        spline_points[:, 0],
        spline_points[:, 1],
        "g-",
        label="Fitted Spline",
        linewidth=1,
    )
    plt.plot(
        actual_path[:, 0], actual_path[:, 1], "r-", label="Actual Path", linewidth=1
    )
    plt.title("Path Comparison")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot error over time
    plt.subplot(2, 1, 2)
    plt.plot(
        timestamps, distances, ".", markersize=2, label="Distance Error", linewidth=1
    )
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.axhline(
        y=error_threshold_to_pass_meter,
        color="g",
        linestyle="--",
        label="Distance Error Threshold",
    )
    plt.fill_between(
        timestamps,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label=STD_DEV_LABEL_STRING,
    )

    plt.title("Turn Accuracy Error Over Time")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Print statistics
    print_stats(stats, "Turn Accuracy Statistics")

    if save_stats_dir:
        stats_full_path = save_stats_dir / "turn_accuracy_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    # Save data if requested
    if save_data_dir:
        save_path = Path(save_data_dir)
        np.savez(
            save_path / "turn_accuracy_data.npz",
            actual_path=actual_path,
            planned_path=planned_path,
            spline_points=spline_points,
            distances=distances,
            timestamps=timestamps,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    # Save plot if requested
    if save_plot_dir:
        save_path = Path(save_plot_dir)
        plt.savefig(save_path / "turn_accuracy_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (
        is_passed,
        stats,
        plt.gcf(),
        actual_path,
        planned_path,
        distances,
        timestamps,
    )


def calculate_instant_acceleration(timestamps, speeds):
    """
    Calculate instantaneous acceleration from speed data.

    Args:
        timestamps: Array of timestamps
        speeds: Array of corresponding speeds

    Returns:
        tuple: (accelerations, time_points)
    """
    dt = np.diff(timestamps)
    dv = np.diff(speeds)
    accelerations = dv / dt
    # Use timestamps[1:] because the acceleration values correspond to the later point in each pair
    return accelerations, timestamps[1:]


def calculate_window_average(timestamps, values, window_size=1.0):
    """
    Calculate time-weighted average values over specified time windows, moving forward by 1 point each time.
    Uses the formula: Average = Σ(value_i * Δt_i) / Σ(Δt_i)

    Args:
        timestamps: Array of timestamps in seconds
        values: Array of corresponding values to be averaged
        window_size: Size of window in seconds (default: 1.0)

    Returns:
        tuple: (window_averages, avg_timestamps)
    """
    window_averages = []
    avg_timestamps = []

    for i in range(len(timestamps) - 1):
        # Find all points within window_size seconds from current point
        mask = (timestamps > timestamps[i]) & (
            timestamps <= timestamps[i] + window_size
        )

        if len(timestamps[mask]) > 1:  # Need at least 2 points for average
            window_values = values[mask]
            window_times = timestamps[mask]

            # Calculate time intervals between consecutive measurements
            delta_t = np.diff(window_times)

            # Calculate value * dt for each interval
            # Use values[:-1] because we have one less interval than values
            value_time_products = window_values[:-1] * delta_t

            # Calculate time-weighted average using the formula
            avg_value = np.sum(value_time_products) / np.sum(delta_t)

            window_averages.append(avg_value)
            avg_timestamps.append(timestamps[i])

    return np.array(window_averages), np.array(avg_timestamps)


def plot_acceleration_analysis(
    time_points, accelerations, stats, title, ylabel, comfort_threshold=2.0, ax=None
):
    """
    Plot acceleration analysis on given axes.

    Args:
        time_points: Array of time points
        accelerations: Array of acceleration values
        stats: Dictionary of statistics
        title: Plot title
        ylabel: Y-axis label
        comfort_threshold: Comfort threshold value
        ax: Matplotlib axes object (creates new if None)
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        time_points, accelerations, ".", markersize=2, label="Acceleration", linewidth=1
    )
    ax.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    ax.fill_between(
        time_points,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label=STD_DEV_LABEL_STRING,
    )

    ax.axhline(
        y=comfort_threshold, color="g", linestyle="--", label="Comfort Threshold"
    )
    ax.axhline(y=-comfort_threshold, color="g", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def run_acceleration_comfort_analysis(
    mcap_path,
    comfort_deceleration_threshold_to_pass=3.0,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Main function to analyze acceleration comfort.

    Args:
        mcap_path: Path to MCAP file
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        topics = ["/hardware_interface/vehicle_status"]
        autoware_msgs need to be built and sourced
    """
    # Extract vehicle state data
    topics = [HARDWARE_VEHICLE_STATUS_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time,
        end_time,
        {HARDWARE_VEHICLE_STATUS_TOPIC: lambda msg: msg.speed},
    )

    timestamps, speeds = extracted_data[topics[0]]
    timestamps = np.array(timestamps)
    speeds = np.array(speeds)

    # Calculate instant accelerations
    accelerations, time_points = calculate_instant_acceleration(timestamps, speeds)
    instant_stats = calculate_error_statistics(accelerations, start_time, end_time)

    # Calculate 1-second average accelerations
    avg_accelerations, avg_timepoints = calculate_window_average(
        time_points, accelerations
    )
    avg_stats = calculate_error_statistics(avg_accelerations, start_time, end_time)

    # Create visualization
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot both analyses
    plot_acceleration_analysis(
        time_points,
        accelerations,
        instant_stats,
        "Instantaneous Acceleration Over Time",
        "Instant Acceleration (m/s²)",
        comfort_threshold=comfort_deceleration_threshold_to_pass,
        ax=ax1,
    )

    plot_acceleration_analysis(
        avg_timepoints,
        avg_accelerations,
        avg_stats,
        "1-Second Average Acceleration Over Time",
        "Average Acceleration (m/s²)",
        comfort_threshold=comfort_deceleration_threshold_to_pass,
        ax=ax2,
    )

    plt.tight_layout()

    # Print statistics
    print_stats(instant_stats, "Instantaneous Acceleration Statistics")
    print_stats(avg_stats, "1-Second Average Acceleration Statistics")

    # Calculate comfort metrics
    instant_discomfort = np.sum(
        np.abs(accelerations) > comfort_deceleration_threshold_to_pass
    )
    avg_discomfort = np.sum(
        np.abs(avg_accelerations) > comfort_deceleration_threshold_to_pass
    )
    is_passed = bool(instant_discomfort == 0 and avg_discomfort == 0)

    print("\nComfort Analysis:")
    print(f"Instantaneous Discomfort Events: {instant_discomfort}")
    print(
        f"Instantaneous Percentage Uncomfortable: {(instant_discomfort/len(accelerations))*100:.2f}%"
    )
    print(f"1-Second Average Discomfort Events: {avg_discomfort}")
    print(
        f"1-Second Average Percentage Uncomfortable: {(avg_discomfort/len(avg_accelerations))*100:.2f}%"
    )

    if save_stats_dir:
        stats_full_path = save_stats_dir / "instantaneous_acceleration_stats.json"
        with open(stats_full_path, "w") as f:
            json.dump(instant_stats, f, indent=2)
        stats_full_path = save_stats_dir / "1sec_average_acceleration_stats.json"
        with open(stats_full_path, "w") as f:
            json.dump(avg_stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    # Save data if requested
    if save_data_dir:
        save_path = Path(save_data_dir)
        np.savez(
            save_path / "acceleration_comfort_data.npz",
            timestamps=time_points,
            speeds=speeds,
            instantaneous_accelerations=accelerations,
            avg_accelerations=avg_accelerations,
            instant_stats=instant_stats,
            avg_stats=avg_stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    # Save plot if requested
    if save_plot_dir:
        save_path = Path(save_plot_dir)
        plt.savefig(save_path / "acceleration_comfort_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (
        is_passed,
        instant_stats,
        avg_stats,
        plt.gcf(),
        accelerations,
        avg_accelerations,
        time_points,
        avg_timepoints,
    )


def calculate_instant_lateral_values(long_velocities, ang_velocities, timestamps):
    """
    Calculate instantaneous lateral acceleration and jerk.

    Args:
        long_velocities: Array of longitudinal velocities
        ang_velocities: Array of angular velocities
        timestamps: Array of timestamps

    Returns:
        tuple: (lateral_acc, lateral_jerk, acc_timestamps, jerk_timestamps)
    """
    # Calculate lateral acceleration
    lateral_acc = long_velocities * ang_velocities

    # Calculate lateral jerk
    dt = np.diff(timestamps)
    jerk = np.diff(lateral_acc) / dt

    return lateral_acc, jerk, timestamps, timestamps[1:]


def run_lateral_analysis(
    mcap_path,
    acc_threshold_to_pass=2.0,  # m/s^2
    jerk_threshold_to_pass=2.0,  # m/s^3
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes lateral acceleration and jerk from vehicle state data using both
    instantaneous and 1-second window averages.

    Args:
        mcap_path: Path to MCAP file
        acc_threshold_to_pass: Maximum acceptable lateral acceleration in m/s^2
        jerk_threshold_to_pass: Maximum acceptable lateral jerk in m/s^3
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots

    Deps:
        Topics: [/hardware_interface/vehicle/twist]
        Msgs: geometry_msgs/Twist
    """

    topics = [HARDWARE_VEHICLE_TWIST_TOPIC]

    # Extract linear and angular velocities
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            HARDWARE_VEHICLE_TWIST_TOPIC: lambda msg: (
                msg.twist.linear.x,  # longitudinal velocity
                msg.twist.angular.z,  # angular velocity
            )
        },
    )

    timestamps, velocity_data = extracted_data[topics[0]]
    long_velocities = np.array([v[0] for v in velocity_data])
    ang_velocities = np.array([v[1] for v in velocity_data])

    # Calculate instantaneous values
    lateral_acc_inst, lateral_jerk_inst, acc_timestamps, jerk_timestamps = (
        calculate_instant_lateral_values(long_velocities, ang_velocities, timestamps)
    )

    # Calculate 1-second window averages
    lateral_acc_avg, acc_timestamps_avg = calculate_window_average(
        acc_timestamps, lateral_acc_inst, window_size=1.0
    )
    lateral_jerk_avg, jerk_timestamps_avg = calculate_window_average(
        jerk_timestamps, lateral_jerk_inst, window_size=1.0
    )

    # Calculate statistics
    acc_inst_stats = calculate_error_statistics(lateral_acc_inst)
    acc_avg_stats = calculate_error_statistics(lateral_acc_avg)
    jerk_inst_stats = calculate_error_statistics(lateral_jerk_inst)
    jerk_avg_stats = calculate_error_statistics(lateral_jerk_avg)

    # Calculate comfort metrics
    acc_inst_discomfort = np.sum(np.abs(lateral_acc_inst) > acc_threshold_to_pass)
    acc_avg_discomfort = np.sum(np.abs(lateral_acc_avg) > acc_threshold_to_pass)
    jerk_inst_discomfort = np.sum(np.abs(lateral_jerk_inst) > jerk_threshold_to_pass)
    jerk_avg_discomfort = np.sum(np.abs(lateral_jerk_avg) > jerk_threshold_to_pass)

    is_passed = all(
        [
            acc_inst_discomfort == 0,
            acc_avg_discomfort == 0,
            jerk_inst_discomfort == 0,
            jerk_avg_discomfort == 0,
        ]
    )

    # Create two separate figures
    # Figure 1: Acceleration measurements
    fig_acc, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot instantaneous acceleration
    plot_acceleration_analysis(
        acc_timestamps,
        lateral_acc_inst,
        acc_inst_stats,
        "Instantaneous Lateral Acceleration",
        "Lateral Acceleration (m/s²)",
        comfort_threshold=acc_threshold_to_pass,
        ax=ax1,
    )

    # Plot 1-second average acceleration
    plot_acceleration_analysis(
        acc_timestamps_avg,
        lateral_acc_avg,
        acc_avg_stats,
        "1-Second Average Lateral Acceleration",
        "Lateral Acceleration (m/s²)",
        comfort_threshold=acc_threshold_to_pass,
        ax=ax2,
    )

    fig_acc.suptitle("Lateral Acceleration Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    # Figure 2: Jerk measurements
    fig_jerk, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot instantaneous jerk
    plot_acceleration_analysis(
        jerk_timestamps,
        lateral_jerk_inst,
        jerk_inst_stats,
        "Instantaneous Lateral Jerk",
        "Lateral Jerk (m/s³)",
        comfort_threshold=jerk_threshold_to_pass,
        ax=ax3,
    )

    # Plot 1-second average jerk
    plot_acceleration_analysis(
        jerk_timestamps_avg,
        lateral_jerk_avg,
        jerk_avg_stats,
        "1-Second Average Lateral Jerk",
        "Lateral Jerk (m/s³)",
        comfort_threshold=jerk_threshold_to_pass,
        ax=ax4,
    )

    fig_jerk.suptitle("Lateral Jerk Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    # Print statistics
    print_stats(acc_inst_stats, "Instantaneous Lateral Acceleration Statistics")
    print_stats(acc_avg_stats, "1-Second Average Lateral Acceleration Statistics")
    print_stats(jerk_inst_stats, "Instantaneous Lateral Jerk Statistics")
    print_stats(jerk_avg_stats, "1-Second Average Lateral Jerk Statistics")

    print("\nComfort Analysis:")
    print(f"Instantaneous Acceleration Discomfort Events: {acc_inst_discomfort}")
    print(f"1-Second Average Acceleration Discomfort Events: {acc_avg_discomfort}")
    print(f"Instantaneous Jerk Discomfort Events: {jerk_inst_discomfort}")
    print(f"1-Second Average Jerk Discomfort Events: {jerk_avg_discomfort}")

    # Save statistics if requested
    if save_stats_dir:
        stats = {
            "instantaneous_acceleration": acc_inst_stats,
            "average_acceleration": acc_avg_stats,
            "instantaneous_jerk": jerk_inst_stats,
            "average_jerk": jerk_avg_stats,
        }
        stats_full_path = save_stats_dir / "lateral_analysis_stats.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    # Save data if requested
    if save_data_dir:
        np.savez(
            save_data_dir / "lateral_analysis_data.npz",
            timestamps=timestamps,
            lateral_acc_inst=lateral_acc_inst,
            lateral_acc_avg=lateral_acc_avg,
            lateral_jerk_inst=lateral_jerk_inst,
            lateral_jerk_avg=lateral_jerk_avg,
            acc_inst_stats=acc_inst_stats,
            acc_avg_stats=acc_avg_stats,
            jerk_inst_stats=jerk_inst_stats,
            jerk_avg_stats=jerk_avg_stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    # Save plots if requested
    if save_plot_dir:
        fig_acc.savefig(save_plot_dir / "lateral_acceleration_analysis.png")
        fig_jerk.savefig(save_plot_dir / "lateral_jerk_analysis.png")
        print(f"\nPlots saved to: {save_plot_dir}")
    else:
        plt.show()

    return (
        is_passed,
        acc_inst_stats,
        acc_avg_stats,
        jerk_inst_stats,
        jerk_avg_stats,
        (fig_acc, fig_jerk),  # Return both figures
        lateral_acc_inst,
        lateral_acc_avg,
        lateral_jerk_inst,
        lateral_jerk_avg,
        timestamps,
    )


def run_guidance_steering_analysis(
    mcap_path,
    error_threshold_to_pass_radian=0.1,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes steering performance by comparing commanded vs actual steering angles at guidance level.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass_radian: Maximum allowed steering error in radians
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/guidance/ctrl_cmd, /hardware_interface/vehicle_status]
    """
    topics = ["/guidance/ctrl_cmd", "/hardware_interface/vehicle_status"]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: msg.cmd.steering_angle,
            topics[1]: lambda msg: msg.angle,
        },
    )

    cmd_timestamps, cmd_angles = extracted_data[topics[0]]
    actual_timestamps, actual_angles = extracted_data[topics[1]]

    # Convert to numpy arrays
    cmd_timestamps = np.array(cmd_timestamps)
    cmd_angles = np.array(cmd_angles)
    actual_timestamps = np.array(actual_timestamps)
    actual_angles = np.array(actual_angles)

    # Align the time series
    common_timestamps, aligned_cmd_angles, aligned_actual_angles = align_time_series(
        cmd_timestamps, cmd_angles, actual_timestamps, actual_angles
    )

    # Calculate differences between commanded and actual angles
    error_angles = np.abs(aligned_cmd_angles - aligned_actual_angles)

    # Calculate statistics
    stats = calculate_error_statistics(error_angles, start_time, end_time)

    # Pass or no pass
    is_passed = float(stats["median"]) < error_threshold_to_pass_radian

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot steering angles
    plt.subplot(2, 1, 1)

    # Also plot original data as dots to show sampling
    plt.plot(
        cmd_timestamps,
        cmd_angles,
        "b.",
        markersize=2,
        alpha=0.3,
        label="Commanded Samples",
    )
    plt.plot(
        actual_timestamps,
        actual_angles,
        "r.",
        markersize=2,
        alpha=0.3,
        label="Actual Samples",
    )

    plt.title("Steering Angle Comparison")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Steering Angle (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot error over time
    plt.subplot(2, 1, 2)
    plt.plot(
        common_timestamps,
        error_angles,
        ".",
        markersize=2,
        label="Steering Error",
        linewidth=1,
    )
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.axhline(
        y=error_threshold_to_pass_radian,
        color="g",
        linestyle="--",
        label="Error Threshold",
    )
    plt.fill_between(
        common_timestamps,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label=STD_DEV_LABEL_STRING,
    )

    plt.title("Steering Error Over Time")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Error (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Print statistics
    print_stats(stats, "Guidance Steering Analysis Statistics")

    if save_stats_dir:
        stats_full_path = save_stats_dir / "guidance_steering_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        np.savez(
            save_data_dir / "guidance_steering_data.npz",
            common_timestamps=common_timestamps,
            cmd_angles_aligned=aligned_cmd_angles,
            actual_angles_aligned=aligned_actual_angles,
            original_cmd_timestamps=cmd_timestamps,
            original_cmd_angles=cmd_angles,
            original_actual_timestamps=actual_timestamps,
            original_actual_angles=actual_angles,
            error_angles=error_angles,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    if save_plot_dir:
        plt.savefig(save_plot_dir / "guidance_steering_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (is_passed, stats, plt.gcf(), error_angles, common_timestamps)


def extract_steering_data(mcap_path, start_time=None, end_time=None):
    """
    Extracts steering data from MCAP file using only steering report topics.
    Returns the extracted data and used topic for the first valid topic found.
    Supports PACMOD, DATASPEED, and NEWEAGLE steering report topics.

    Args:
        mcap_path: Path to MCAP file
        start_time: Time to start the extraction
        end_time: Time to end the extraction

    Returns:
        tuple: lists of timestamps, actual steering angles, and commanded steering angles, and used topic

    Raises:
        ValueError: If no valid data found in any report topics
    """
    # Define report topics and their corresponding extractors
    report_topics = [
        # PACMod report topic
        {
            'report_topic': HARDWARE_PACMOD_STEER_REPORT_TOPIC,
            'extractors': {
                HARDWARE_PACMOD_STEER_REPORT_TOPIC: lambda msg: (
                    msg.output,  # rad
                    msg.command  # rad
                )
            }
        },
        # Fusion report topic
        {
            'report_topic': HARDWARE_DATASPEED_STEER_REPORT_TOPIC,
            'extractors': {
                HARDWARE_DATASPEED_STEER_REPORT_TOPIC: lambda msg: (
                    msg.steering_wheel_angle,  # rad
                    msg.steering_wheel_cmd  # rad
                )
            }
        },
        # Pacifica report topic
        {
            'report_topic': HARDWARE_NEWEAGLE_STEER_REPORT_TOPIC,
            'extractors': {
                HARDWARE_NEWEAGLE_STEER_REPORT_TOPIC: lambda msg: (
                    msg.steering_wheel_angle * DEG_TO_RAD,  # deg to rad
                    msg.steering_wheel_angle_cmd * DEG_TO_RAD  # deg to rad
                )
            }
        }
    ]

    # Try each report topic until we find one with data
    for topic_info in report_topics:
        report_topic = topic_info['report_topic']

        try:
            current_extracted_data = extract_mcap_data(
                mcap_path,
                [report_topic],
                start_time=start_time,
                end_time=end_time,
                field_extractors=topic_info['extractors']
            )

            # Check if we got any data if so return
            if len(current_extracted_data[report_topic][0]) > 0:
                timestamps, angle_values = current_extracted_data[report_topic]
                actual_sw_angle = np.array([v[0] for v in angle_values])
                commanded_sw_angle = np.array([v[1] for v in angle_values])
                return timestamps, actual_sw_angle, commanded_sw_angle, report_topic

        except Exception as e:
            print(f"Warning: Could not extract data for topic {report_topic}: {str(e)}")
            continue

    raise ValueError("No valid data found in any of the report topics")

def run_steering_wheel_analysis(
    mcap_path,
    error_threshold_to_pass=0.1,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes steering performance by comparing commanded vs actual steering values.
    Supports multiple vehicle types by checking different topic pairs.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass: Maximum allowed steering error
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    """
    # Extract data from 3 possible topic pairs
    timestamps, actual_values, cmd_values, used_topic  = extract_steering_data(mcap_path, start_time, end_time)

    # Convert to numpy arrays
    actual_timestamps = np.array(timestamps)
    actual_values = np.array(actual_values)
    cmd_values = np.array(cmd_values)

    # Calculate differences between commanded and actual steering wheel values
    error_values = np.abs(cmd_values - actual_values)

    # Calculate statistics
    stats = calculate_error_statistics(error_values, start_time, end_time)

    # Pass or no pass
    is_passed = float(stats["median"]) < error_threshold_to_pass

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot steering Wheel values
    plt.subplot(2, 1, 1)

    # Also plot original data as dots to show sampling
    plt.plot(
        timestamps,
        cmd_values,
        "b.",
        markersize=2,
        alpha=0.3,
        label="Commanded Samples",
    )
    plt.plot(
        actual_timestamps,
        actual_values,
        "r.",
        markersize=2,
        alpha=0.3,
        label="Actual Samples",
    )

    plt.title(f"Steering Wheel Value Comparison\nUsing topic: {used_topic}")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Steering Wheel Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot error over time
    plt.subplot(2, 1, 2)
    plt.plot(
        timestamps,
        error_values,
        ".",
        markersize=2,
        label="Steering Error",
        linewidth=1,
    )
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.axhline(
        y=error_threshold_to_pass, color="g", linestyle="--", label="Error Threshold"
    )
    plt.fill_between(
        timestamps,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label=STD_DEV_LABEL_STRING,
    )

    plt.title("Steering Wheel Error Over Time")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Print statistics
    print_stats(stats, "Steering Wheel Analysis Statistics")

    if save_stats_dir:
        stats_full_path = save_stats_dir / "steering_wheel_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        np.savez(
            save_data_dir / "steering_wheel_data.npz",
            timestamps=timestamps,
            cmd_values=cmd_values,
            actual_values=actual_values,
            error_values=error_values,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    if save_plot_dir:
        plt.savefig(save_plot_dir / "steering_wheel_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (is_passed, stats, plt.gcf(), error_values, timestamps)


def get_planner_trajectory_intervals(
    mcap_path,
    planner_plugin_name,
    start_time=None,
    end_time=None,
):
    """
    Extract time intervals when a specific planner was active based on trajectory plans.
    Uses header stamp time and first point of each plan for decision making.

    Args:
        mcap_path: Path to MCAP file
        planner_plugin_name: Name of the planner plugin to track (e.g. "guidance/plugins/inlanecruising_plugin")
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis

    Returns:
        List of tuples [(start_time1, end_time1), (start_time2, end_time2), ...] representing
        time intervals when the specified planner was active

    Deps:
        Topics: [/guidance/plan_trajectory]
        Msgs: carma_planning_msgs/msg/TrajectoryPlan
    """
    topics = [GUIDANCE_PLAN_TRAJECTORY_TOPIC]

    # Extract timestamp and planner name from each trajectory plan
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: (
                # trajecteory_plan on this topic is guaranteed to have minimum 2 points
                msg.trajectory_points[0].planner_plugin_name
            )
        },
    )

    timestamps, plan_data = extracted_data[topics[0]]

    # Initialize variables
    intervals = []
    current_start = None

    # Process each trajectory plan
    for i in range(len(timestamps)):
        # Check if this is the planner we're looking for
        is_target_planner = plan_data[i] == planner_plugin_name

        if is_target_planner and current_start is None:
            # Start of a new interval
            current_start = timestamps[i]
        elif not is_target_planner and current_start is not None:
            # End of current interval
            intervals.append((current_start, timestamps[i]))
            current_start = None

    # Handle case where planner was active at end of data
    if current_start is not None:
        # Add 0.1s just to account for the fact that trajectories last for 0.1s
        new_end_time = timestamps[-1] + 0.1
        intervals.append((current_start, new_end_time))

    # Print summary
    print(f"\nFound {len(intervals)} intervals for planner: {planner_plugin_name}")
    for i, (start, end) in enumerate(intervals):
        duration = end - start
        print(f"Interval {i+1}: {duration:.2f} seconds (from {start:.2f} to {end:.2f})")

    return intervals

def run_speed_limit_change_response_analysis(
    mcap_path,
    response_time_threshold=0.2,  # seconds
    steady_state_indication_time=3.0,    # seconds
    speed_tolerance_pct=0.05,      # 5% tolerance for speed match
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None
):
    """
    Analyze vehicle's response to speed limit changes in the map.
    Passes if for each new speed limit change, the vehicle is able
    to get into a steady state within acceptable tolerance percentage
    of the new speed limit (can't be exact due to geometry of the road)
    and within configurable parameter of duration. Also requires that speed
    command should be applied within threshold after the speed limit
    changes. For example: True if after new speed limit change,
    vehicle's commanded speed is within 5% of target for at least 3 consecutive seconds
    and starts commanding different speed within 0.1s
    NOTE: This script should be used for straightaways and speed limit change segments
          that would last at least steady_state_indication_time for best characterization
    Args:
        mcap_path: Path to MCAP file
        response_time_threshold: Maximum acceptable response time to speed changes (seconds)
        steady_state_indication_time: Minimum duration required within speed tolerance to consider
            steady state achieved
        speed_tolerance_pct: Tolerance percentage for speed matching (to account for road geometry)
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis
        save_stats_dir: Directory to save statistics
        save_data_dir: Directory to save analysis data
        save_plot_dir: Directory to save plots

    Returns:
        Tuple containing:
        - pass_results: Pass/Fail if all ciriterias pass/fail
        - statistics: Detailed statistics about the analysis
        - figure: Matplotlib figure object
        - speed_changes: Detected speed limit change events
        - response_times: Response times for each speed change

    Deps:
        Topics: [/hardware_interface/vehicle/twist, /guidance/route_state, /guidance/control_cmd]
        Msgs: geometry_msgs/Twist, custom_msgs/RouteState, custom_msgs/ControlCommand
    """

    # Extract data from MCAP
    topics = [HARDWARE_VEHICLE_TWIST_TOPIC, GUIDANCE_ROUTE_STATE_TOPIC, GUIDANCE_CONTROL_CMD_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: (
                msg.twist.linear.x  # longitudinal velocity in m/s
            ),
            topics[1]: lambda msg: (
                msg.speed_limit     # Actual speed limit in m/s
            ),
            topics[2]: lambda msg: (
                msg.cmd.linear_velocity # Commanded speed in m/s
            )
        },
    )

    # Extract time series data
    timestamps_vel, velocity_data = extracted_data[topics[0]]
    long_velocities = velocity_data

    timestamps_speed_limit, speed_limit_data = extracted_data[topics[1]]
    speed_limits =  speed_limit_data

    timestamps_cmd, cmd_data = extracted_data[topics[2]]
    cmd_velocities = cmd_data

    # Approximate aligned timestamps for all variables for plotting
    # Align time series data for consistent comparison using existing function twice
    first_aligned_timestamps, first_aligned_velocities, aligned_speed_limits = align_time_series(
        timestamps_vel, long_velocities,
        timestamps_speed_limit, speed_limits
    )

    # Second alignment to get all three series aligned
    timestamps, aligned_cmd_velocities, aligned_velocities = align_time_series(
        timestamps_cmd, cmd_velocities,
        first_aligned_timestamps, first_aligned_velocities
    )

    # Now get aligned speed limits with the same timestamps
    aligned_speed_limits = np.interp(timestamps, first_aligned_timestamps, aligned_speed_limits)

    # Reassign to variables used later in the function
    long_velocities = aligned_velocities
    speed_limits = aligned_speed_limits
    cmd_velocities = aligned_cmd_velocities

    # Detect speed limit changes only use raw speed limit data for accuracy
    speed_limit_changes = detect_speed_limit_changes(timestamps_speed_limit, speed_limit_data)

    # Analyze response to speed changes only use raw cmd data for accuracy
    response_times, steady_state_indication_periods = analyze_speed_responses(
        timestamps,
        aligned_cmd_velocities,
        aligned_velocities,
        speed_limit_changes,
        steady_state_indication_time,
        speed_tolerance_pct
    )

    # Calculate statistics
    statistics = {
        "num_speed_changes": len(speed_limit_changes),
        "response_times": {
            "mean": float(np.mean(response_times)) if len(response_times) > 0 else None,
            "median": float(np.median(response_times)) if len(response_times) > 0 else None,
            "min": float(np.min(response_times)) if len(response_times) > 0 else None,
            "max": float(np.max(response_times)) if len(response_times) > 0 else None,
            "values": response_times.tolist() if len(response_times) > 0 else []
        },
        "steady_state_indication": {
            "total_periods": len(steady_state_indication_periods),
            "periods": steady_state_indication_periods
        }
    }

    # Determine pass/fail for each criterion
    response_time_pass = all(
        rt <= response_time_threshold for rt in response_times
        ) if len(response_times) > 0 else False
    steady_state_pass = len(steady_state_indication_periods) >= len(speed_limit_changes)

    pass_results = {
        "response_time": response_time_pass,
        "steady_state_indication": steady_state_pass,
        "overall": response_time_pass and steady_state_pass
    }

    fig = plt.figure(figsize=(15, 10))
    # Plot vehicle speed, speed limit, and commanded speed
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(timestamps, long_velocities, 'b-', linewidth=2, label='Vehicle Speed')
    ax1.plot(timestamps, speed_limits, 'r--', alpha=0.5,
            linewidth=2, label='Speed Limit')
    ax1.plot(timestamps, cmd_velocities, 'g-.', alpha=0.5,
            label='Commanded Speed')
    # Highlight speed change events
    for event in speed_limit_changes:
        event_time, old_limit, new_limit = event
        # Add vertical line to mark the event
        ax1.axvline(x=event_time, color='k', linestyle=':', alpha=0.7)
        # Add text indicating the time point on x-axis
        ax1.text(event_time, min(long_velocities) - 1, f"{event_time:.1f}s",
                rotation=90, verticalalignment='top')
        # Add text indicating the speed limit on y-axis
        ax1.text(event_time - 0.5, new_limit, f"{new_limit:.1f} m/s",
                horizontalalignment='right', verticalalignment='center')
    # Highlight steady state periods
    steady_state_period_added = False
    for period in steady_state_indication_periods:
        start_time, end_time, speed = period
        if not steady_state_period_added:
            ax1.axvspan(start_time, end_time, alpha=0.2, color='g',
                        label='Steady State Indication Periods')
            steady_state_period_added = True
        else:
            ax1.axvspan(start_time, end_time, alpha=0.2, color='g')
    ax1.set_title("Speed Limit vs. Commanded Speed vs. Actual Speed")
    ax1.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Plot speed difference between vehicle speed and speed limit
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    speed_diff = long_velocities - speed_limits
    ax2.plot(timestamps, speed_diff, 'r-', linewidth=1.5,
            label='Speed Difference (Vehicle Speed - Speed Limit)')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    # Also show vertical lines for speed change events in the bottom plot
    for event in speed_limit_changes:
        event_time, _, _ = event
        ax2.axvline(x=event_time, color='k', linestyle=':', alpha=0.7)
        ax2.text(event_time, min(speed_diff) - 0.5, f"{event_time:.1f}s",
                rotation=90, verticalalignment='top')

    tolerance_added_to_legend = False

    for i, event in enumerate(speed_limit_changes):
        event_time, old_limit, new_limit = event

        # Calculate tolerance bands for this speed limit
        upper_tolerance = new_limit * speed_tolerance_pct
        lower_tolerance = -new_limit * speed_tolerance_pct

        # Determine the end time for this segment
        if i < len(speed_limit_changes) - 1:
            # End at the next speed limit change
            end_time = speed_limit_changes[i+1][0]
        else:
            # For the last change, go to the end of the data
            end_time = timestamps[-1]

        # Draw tolerance bands only for this time segment
        if not tolerance_added_to_legend:
            # First time, add to legend
            ax2.hlines(y=upper_tolerance, xmin=event_time, xmax=end_time,
                    color='g', linestyle='--', alpha=0.5,
                    label=f'Tolerance (±{speed_tolerance_pct*100:.0f}%)')
            ax2.hlines(y=lower_tolerance, xmin=event_time, xmax=end_time,
                    color='g', linestyle='--', alpha=0.5)
            tolerance_added_to_legend = True
        else:
            # Subsequent times, don't add to legend
            ax2.hlines(y=upper_tolerance, xmin=event_time, xmax=end_time,
                    color='g', linestyle='--', alpha=0.5)
            ax2.hlines(y=lower_tolerance, xmin=event_time, xmax=end_time,
                    color='g', linestyle='--', alpha=0.5)

    ax2.set_title("Speed Difference")
    ax2.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax2.set_ylabel("Difference (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save results if directories are provided
    if save_stats_dir:
        stats_dir = Path(save_stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / "speed_change_analysis.json"
        with open(stats_path, "w") as f:
            json.dump({
                "pass_results": pass_results,
                "statistics": statistics
            }, f, indent=2)
        print(f"Stats saved to: {stats_path}")

    if save_data_dir:
        data_dir = Path(save_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "speed_change_analysis_data.npz"
        np.savez(
            data_path,
            timestamps=timestamps,
            velocities=long_velocities,
            speed_limits=speed_limits,
            cmd_velocities=cmd_velocities,
            speed_limit_changes=speed_limit_changes,
            response_times=response_times,
            steady_state_indication_periods=steady_state_indication_periods
        )
        print(f"Data saved to: {data_path}")

    if save_plot_dir:
        plot_dir = Path(save_plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "speed_change_analysis.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")
    else:
        plt.show()

    return (pass_results["overall"], statistics, fig, speed_limit_changes, response_times)


def detect_speed_limit_changes(timestamps, speed_limits, min_change=0.5):
    """
    Detect significant changes in speed limits.

    Args:
        timestamps: Array of timestamps
        speed_limits: Array of speed limits
        min_change: Minimum change in speed limit to be considered significant (m/s)

    Returns:
        List of tuples (timestamp, old_limit, new_limit) for each detected change
    """
    changes = []
    prev_limit = speed_limits[0]

    for i in range(1, len(speed_limits)):
        current_limit = speed_limits[i]
        if prev_limit < 0.01:
            prev_limit = current_limit
            continue
        if abs(current_limit - prev_limit) >= min_change:
            changes.append((timestamps[i], prev_limit, current_limit))
            prev_limit = current_limit

    return changes


def analyze_speed_responses(timestamps, cmd_velocities, velocities, speed_limit_changes,
                           steady_state_indication_time, speed_tolerance_pct,
                           min_speed_change_detection_threshold=0.067):
    """
    Analyze vehicle's response to speed limit changes.
    Args:
        timestamps: Array of timestamps
        cmd_velocities: Array of commanded vehicle velocities
        velocities: Array of vehicle velocities
        speed_limit_changes: List of detected speed limit changes
        steady_state_indication_time: Duration required at new speed for steady state
        speed_tolerance_pct: Tolerance percentage for speed matching (used only for steady state)
        min_speed_change_detection_threshold: Minimum speed change required to register as a valid
            response to a speed limit change (default is 0.067 m/s for 2m/s^2 change in 0.033s
            (30Hz) is 0.067 m/s)
    Returns:
        Tuple containing:
        - Array of response times for each speed change
        - List of first X seconds (based on steady_state_indication_time) of steady state
            [(start_time, end_time, speed_limit), ...]
    """
    response_time_idxs = []
    response_times = np.array([])
    steady_state_indication_periods = []

    # Get the response times and indexes for each speed limit change
    for i, change in enumerate(speed_limit_changes):
        change_time, old_limit, new_limit = change

        # Find the index of the change in the timestamps array
        change_idx = np.argmin(np.abs(timestamps - change_time))

        # Calculate response time: time when speed differs by at least
        # min_speed_change_detection_threshold from original
        response_time_idx = change_idx
        init_cmd_velocity = cmd_velocities[change_idx]

        for j in range(change_idx + 1, len(timestamps)):
            # Check if velocity has changed by at least min_speed_change_detection_threshold
            # from the initial value
            if abs(cmd_velocities[j] - init_cmd_velocity) >= min_speed_change_detection_threshold:
                response_time_idx = j
                break  # from inner loop

        # Calculate response time
        response_time = timestamps[response_time_idx] - timestamps[change_idx]
        response_times = np.append(response_times, response_time)
        response_time_idxs.append(response_time_idx)

    # Analyze steady states for each response
    # Here we assume velocities and cmd_velocities are aligned
    for i, response_time_idx in enumerate(response_time_idxs):
        # Get current speed limit
        _, _, new_limit = speed_limit_changes[i]

        # Define tolerance band for the new speed limit (for steady state)
        upper_bound = new_limit * (1 + speed_tolerance_pct)
        lower_bound = new_limit * (1 - speed_tolerance_pct)

        # Analyze steady state (duration at new speed limit within tolerance)
        steady_state_start = None
        next_event_time_idx = len(timestamps)

        if (i + 1 < len(response_time_idxs)):
            # Check until the next speed change
            next_event_time_idx = response_time_idxs[i + 1]

        # Steady state check only until the next speed change
        for k in range(response_time_idx, next_event_time_idx):
            if lower_bound <= velocities[k] <= upper_bound:
                if steady_state_start is None:
                    steady_state_start = timestamps[k]

                # Check if we've had enough consecutive time in the band
                if timestamps[k] - steady_state_start >= steady_state_indication_time:
                    steady_state_indication_periods.append((
                        steady_state_start,
                        timestamps[k],
                        new_limit
                    ))
                    break
            else:
                # Reset if we go out of band
                steady_state_start = None

    return response_times, steady_state_indication_periods

def run_guidance_speed_analysis(
    mcap_path,
    error_threshold_to_pass_mph=0.5,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):

    """
    Extract time intervals when a specific planner was active based on trajectory plans.
    Uses header stamp time and first point of each plan for decision making.

    Args:
        mcap_path: Path to MCAP file
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis

    Returns:
        List of tuples [(start_time1, end_time1), (start_time2, end_time2), ...] representing
        time intervals when the specified planner was active

    Deps:
        Topics: [/hardware_interface/vehicle/twist]
        Msgs: geometry_msgs/Twist
    """

    topics = [HARDWARE_VEHICLE_TWIST_TOPIC, GUIDANCE_ROUTE_STATE_TOPIC, GUIDANCE_CONTROL_CMD_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: (
                msg.twist.linear.x,  # longitudinal velocity
            ),
            topics[1]: lambda msg: (
                msg.speed_limit     # Actual speed limit in m/s
            ),
            topics[2]: lambda msg: (
                msg.cmd.linear_velocity # Commanded speed limit in m/s
            )
        },
    )

    timestamps_vel, velocity_data = extracted_data[topics[0]]
    long_velocities = np.array([v[0] for v in velocity_data])

    timestamps_speed_limit, speed_limits = extracted_data[topics[1]]
    timestamps_cmd_vel, long_cmd_velocities = extracted_data[topics[2]]

    # Align timeseries
    timestamps, long_velocities, speed_limits = align_time_series(timestamps_vel, long_velocities, timestamps_speed_limit, speed_limits)

    # Calculate statistics
    speed_limit_error = np.abs(long_velocities - speed_limits)
    speed_limit_error_stats = calculate_error_statistics(speed_limit_error, start_time, end_time)

    # Pass or no pass
    error_threshold_to_pass_mps = error_threshold_to_pass_mph * MPH_TO_MPS
    is_passed = float(speed_limit_error_stats["median"]) < ( error_threshold_to_pass_mps )


    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot speed error values
    plt.subplot(2, 1, 1)

    plt.plot(
        timestamps,
        long_velocities,
        "b.",
        markersize=2,
        alpha=0.3,
        label="Speed",
    )

    plt.plot(
        timestamps,
        speed_limits,
        "r.",
        alpha=0.3,
        label="Speed Limit"
    )

    plt.title(f"Vehicle Speed and Speed Limit")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Speed (mps)")
    plt.grid(True, alpha=0.3)
    plt.legend()


    if save_stats_dir:
        stats_full_path = save_stats_dir / "guidance_speed_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(speed_limit_error_stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        np.savez(
            save_data_dir / "guidance_speed_analysis_data.npz",
            timestamps=timestamps,
            cmd_values=long_cmd_velocities,
            actual_values=long_velocities,
            error_values=speed_limit_error,
            stats=speed_limit_error_stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    if save_plot_dir:
        plt.savefig(save_plot_dir / "guidance_speed_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    return (is_passed, speed_limit_error_stats, plt.gcf(), speed_limit_error, timestamps)

def compute_turn_reference_speed(steering_angle_rad, wheelbase_m=2.7, a_max=2.5):
    if abs(steering_angle_rad) < 1e-3:
        return float('inf')  # straight line, no turn limit
    radius = wheelbase_m / np.tan(steering_angle_rad)
    return np.sqrt(a_max * abs(radius))

def compute_turn_lateral_acceleration(steering_angle_rad, vehicle_speed_mps, wheelbase_m):
    if abs(steering_angle_rad) < 1e-3:
        return 0.0  # no turning, no lateral acceleration
    radius = wheelbase_m / np.tan(steering_angle_rad)
    a_lat = (vehicle_speed_mps ** 2) / abs(radius)
    return a_lat

def run_turn_acceleration_analysis(
    mcap_path,
    turn_deceleration_threshold_to_pass=3.0,
    turn_threshold=0.2,  # radians
    wheelbase_m=2.7,     # meters
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Main function to analyze turn lateral acceleration.

    Args:
        mcap_path: Path to MCAP file
        turn_deceleration_threshold_to_pass: Max allowed lateral acceleration (m/s^2)
        turn_threshold: Steering angle threshold to consider as a turn (rad)
        wheelbase_m: Vehicle wheelbase in meters
    """

    topics = [GUIDANCE_CONTROL_CMD_TOPIC]
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time,
        end_time,
        field_extractors={
            topics[0]: lambda msg: (
                msg.cmd.linear_velocity,  # m/s
                msg.cmd.steering_angle             # rad
            ),
        },
    )

    # Extract data
    timestamps_cmd, cmd_data = extracted_data[topics[0]]
    speeds = np.array([d[0] for d in cmd_data])
    steering_angles = np.array([d[1] for d in cmd_data])

    # Identify turns
    turn_indices = np.where(np.abs(steering_angles) > turn_threshold)[0]
    if turn_indices.size == 0:
        print("Warning: No Turn events found above threshold. Skipping analysis.")
        return
    turn_times = np.array(timestamps_cmd)[turn_indices]
    turn_speeds = speeds[turn_indices]
    turn_angles = steering_angles[turn_indices]

    # Compute lateral acceleration
    lateral_accels = np.array([
    compute_turn_lateral_acceleration(angle, speed, wheelbase_m)
    for angle, speed in zip(turn_angles, turn_speeds)])
    lateral_accels = np.abs(lateral_accels)  # consider magnitude only

    # Compare with threshold
    acc_violations = lateral_accels > turn_deceleration_threshold_to_pass

    turn_acc_stats = calculate_error_statistics(lateral_accels, start_time, end_time)
    is_passed = float(turn_acc_stats["median"]) < turn_deceleration_threshold_to_pass

    print(f"\nFound {len(turn_indices)} turn samples above threshold angle ({turn_threshold} rad).")
    print(f"{np.sum(acc_violations)} exceeded {turn_deceleration_threshold_to_pass} m/s² lateral accel.")

    # Plot
    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(turn_times, lateral_accels, 'b-', label="Lateral Acceleration")
    ax1.axhline(y=turn_deceleration_threshold_to_pass, color='g', linestyle='--', label="Threshold")
    ax1.axhline(y=turn_acc_stats["median"], color="r", linestyle="--", label="Median")
    ax1.set_title("Lateral Acceleration During Turns")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(turn_times, turn_angles, 'g-', label="Steering Angle (rad)")
    ax2.axhline(y=turn_threshold, color='gray', linestyle='--', label="Turn Threshold")
    ax2.axhline(y=-turn_threshold, color='gray', linestyle='--')
    ax2.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax2.set_ylabel("Steering Angle (rad)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if save_plot_dir:
        fig.savefig(save_plot_dir / "turn_lateral_acceleration_analysis.png")
        print(f"Turn lateral acceleration plot saved to: {save_plot_dir}")
    else:
        plt.show()

    if save_stats_dir:
        stats_full_path = save_stats_dir / "turn_lateral_acceleration_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(turn_acc_stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    return (is_passed, turn_acc_stats, plt.gcf(), lateral_accels, timestamps_cmd)

def run_turn_speed_analysis(
    mcap_path,
    turn_threshold = 0.2,
    wheelbase = 2.75,
    lateral_acc = 2.5,
    execc_turn_speed_threshold = 0.1,
    start_time=None,
    end_time=None,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):

    """
    Extract time intervals when a specific planner was active based on trajectory plans.
    Uses header stamp time and first point of each plan for decision making.

    Args:
        mcap_path: Path to MCAP file
        start_time: Optional start time to begin analysis
        end_time: Optional end time to end analysis

    Returns:
        List of tuples [(start_time1, end_time1), (start_time2, end_time2), ...] representing
        time intervals when the specified planner was active

    Deps:
        Topics: [/hardware_interface/vehicle/twist]
        Msgs: geometry_msgs/Twist
    """
    topics = [HARDWARE_VEHICLE_TWIST_TOPIC, GUIDANCE_CONTROL_CMD_TOPIC, GUIDANCE_ROUTE_STATE_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: (
                msg.twist.linear.x,  # longitudinal velocity
            ),
            topics[1]: lambda msg: (
                msg.cmd.linear_velocity,
                msg.cmd.steering_angle,
            ),
            topics[2]: lambda msg: (
                msg.speed_limit,  # longitudinal velocity
            ),
        },
    )

    timestamps_vel, velocity_data = extracted_data[topics[0]]
    long_velocities = np.array([v[0] for v in velocity_data])

    timestamps_ctrl_cmd, ctrl_cmd_data = extracted_data[topics[1]]
    long_cmd_velocities = np.array([d[0] for d in ctrl_cmd_data])
    steering_angles = np.array([d[1] for d in ctrl_cmd_data])

    timestamps_speed_limit, speed_limits = extracted_data[topics[2]]
    # Align steering with speed
    timestamps_steer_aligned, steering_angles_aligned, velocity_for_steer = align_time_series(
        timestamps_ctrl_cmd, steering_angles, timestamps_vel, long_velocities
    )

    # Compute desired speed when steering is high

    # Find indices of high steering angles (turns)
    high_steering_indices = np.where(np.abs(steering_angles_aligned) > turn_threshold)[0]

    if high_steering_indices.size == 0:
        print("Warning: No Turn events found above threshold. Skipping analysis.")
        return


    # Extract data for those moments
    steer_times = timestamps_steer_aligned[high_steering_indices]
    steer_angles_during_turns = steering_angles_aligned[high_steering_indices]
    vehicle_speeds_during_turns = velocity_for_steer[high_steering_indices]
    # Compute reference speeds based on turn radius and compare
    turn_speed_refs = np.array([
        compute_turn_reference_speed(angle, wheelbase, lateral_acc)
        for angle in steer_angles_during_turns
    ])

    speed_excess = vehicle_speeds_during_turns - turn_speed_refs


    speed_excess_stats = calculate_error_statistics(speed_excess, start_time, end_time)
    is_passed = abs(float(speed_excess_stats["median"])) < execc_turn_speed_threshold


    # --- Visualization: Actual vs Reference Speed During Turns ---

    # Create figure and axes for 3 subplots
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    timestamps_speed_limit = np.array(timestamps_speed_limit)
    speed_limits = np.array(speed_limits)
    steer_times = np.array(steer_times)

    # Create a mask where timestamps_speed_limit values fall within the range of steer_times
    mask = (timestamps_speed_limit >= steer_times[0]) & (timestamps_speed_limit <= steer_times[-1])

    # Slice to get matching values
    speed_limits_trimmed = speed_limits[mask]
    timestamps_trimmed = timestamps_speed_limit[mask]

    # Then plot using the trimmed timestamps
    ax0.plot(steer_times, vehicle_speeds_during_turns, 'g-', label="Vehicle Speed")
    ax0.plot(timestamps_trimmed, speed_limits_trimmed, 'k--', label="Speed Limit")
    ax0.set_title("Vehicle Speed vs Speed Limit During Turn")
    ax0.set_ylabel("Speed (m/s)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # Top subplot: Actual and reference speeds
    ax1.plot(steer_times, vehicle_speeds_during_turns, 'b-', label="Actual Speed")
    ax1.plot(steer_times, turn_speed_refs, 'g--', label="Reference Turn Speed")


    ax1.fill_between(
        steer_times,
        turn_speed_refs,
        vehicle_speeds_during_turns,
        where=(vehicle_speeds_during_turns > turn_speed_refs),
        color='orange',
        alpha=0.3,
        label="Overspeed"
    )
    ax1.set_title("Actual vs Reference Speed During Turns")
    ax1.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom subplot: Speed Excess (Actual - Reference)
    ax2.plot(steer_times, speed_excess, 'm-', label="Speed Excess (Actual - Reference)")
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(y=speed_excess_stats["median"], color="r", linestyle="--", label="Median")
    ax2.set_title("Speed Excess During Turns")
    ax2.set_xlabel(TIME_SECONDS_LABEL_STRING)
    ax2.set_ylabel("Speed Excess (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Save or show
    if save_plot_dir and isinstance(save_plot_dir, (str, Path)):
        save_plot_path = Path(save_plot_dir) / "turn_speed_analysis.png"
        fig.savefig(save_plot_path)
        print(f"Turn speed comparison plot saved to: {save_plot_path}")
    else:
        plt.show()

    if save_stats_dir:
        stats_full_path = save_stats_dir / "turn_speed_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(speed_excess_stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    return (is_passed, speed_excess_stats, plt.gcf(), speed_excess, steer_times)

def get_geofence_entrance_and_exit_times(mcap_path):
    """
    Extract the first time a vehicle enters and exits a geofenced area

    Args:
        mcap_path: Path to MCAP file

    Returns:
        time_enter_geofence: The time the vehicle entered the geofence
        time_exit_geofence: The time the vehicle exited the geofence
        found_geofence_tiems: Boolean; True if time_enter_geofence and time_exit_geofence are found
    """
    is_on_active_geofence = False
    found_geofence_entrance_time = False
    found_geofence_exit_time = False

    topics = [ACTIVE_GEOFENCE_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        field_extractors={ACTIVE_GEOFENCE_TOPIC: lambda msg: msg.is_on_active_geofence}
    )
    timestamps, states = extracted_data[topics[0]]

    for timestamp, geofence_state in zip(timestamps, states):
        # Check if is_on_active_geofence parameter is true and aren't currently in a geofence
        if(geofence_state and not is_on_active_geofence):
            time_enter_active_geofence = timestamp
            print("Entered geofence at " + str(timestamp))
            found_geofence_entrance_time = True
            is_on_active_geofence = True

        # Check if is_on_active_geofence parameter is false and are currently in a geofence
        if(not geofence_state and is_on_active_geofence):
            time_exit_active_geofence = timestamp
            found_geofence_exit_time = True
            time_in_geofence = time_exit_active_geofence - time_enter_active_geofence
            print("Spent " + str(time_in_geofence) + " sec in geofence. Started at " + str(time_enter_active_geofence))
            is_on_active_geofence = False

    found_geofence_times = False
    if (found_geofence_entrance_time and found_geofence_exit_time):
        found_geofence_times = True

    if not found_geofence_times:
        return None, None, False

    return time_enter_active_geofence, time_exit_active_geofence, found_geofence_times

def get_route_original_speed(mcap_path, start_time=None):
    """
    Get the speed limit of the first route

    Args:
        mcap_path: Path to MCAP file
        start_time: Start time to begin analysis

    Returns:
        Speed limit of the first route in m/s
    """
    topics = [GUIDANCE_ROUTE_STATE_TOPIC]
    original_speed_limit = 0

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        field_extractors={GUIDANCE_ROUTE_STATE_TOPIC: lambda msg: msg.speed_limit}
    )

    timestamps, speed_limits = extracted_data[topics[0]]
    for timestamp, speed_limit in zip(timestamps, speed_limits):
        original_speed_limit = speed_limit
        break

    return original_speed_limit


def check_geofence_in_reroute(
    mcap_path,
    closed_lanelets,
    save_data_dir=None,
):
    """
    Checks whether a closed lanelet is present in either the original route (FWZ-1) or the reroute (FWZ-8)

    Args:
        mcap_path: Path to MCAP file
        closed_lanelets: list of closed lanelets

    Returns:
        initial_route_includes_closed_lane: Boolean - True if closed lanelet was present in original route
        map_updated_for_closed_lane: Boolean - True if closed lanelet is not present in reroute
    """
    topics = [GUIDANCE_ROUTE_TOPIC]
    shortest_path_lanelets = []

    if not closed_lanelets:
        print(f"FWZ-1 Failed: Passed in list of closed lanelets is empty. Can not validate closed lanelet is in original route. Please populate list with expected closed lanelets.")
        print(f"FWZ-8 Failed: Passed in list of closed lanelets is empty. Can not validate closed lanelet is removed in re-route. Please populate list with expected closed lanelets.")
        return False, False

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        field_extractors={GUIDANCE_ROUTE_TOPIC: lambda msg: msg.shortest_path_lanelet_ids}
    )
    timestamps, paths = extracted_data[topics[0]]

    for timestamp, path in zip(timestamps, paths):
        print(f"Shortest Path Route Update at {timestamp}: {path}")

        shortest_path_lanelets.append([])
        for lanelet in path:
            shortest_path_lanelets[-1].append(lanelet)

    # If there are two route paths, check that the first (original) route contains the closed lanelet(s) and the second route doesn't
    # Note: Assumes there should be only two routes: (1) the initial route and (2) the re-routed route
    initial_route_includes_closed_lane = False
    map_is_updated_for_closed_lane = False
    if (len(shortest_path_lanelets) > 1):
        original_shortest_path = shortest_path_lanelets[0]
        rerouted_shortest_path = shortest_path_lanelets[-1]

        for lanelet_id in closed_lanelets:
            if lanelet_id in original_shortest_path:
                initial_route_includes_closed_lane = True
            else:
                initial_route_includes_closed_lane = False
                break

        for lanelet_id in closed_lanelets:
            if lanelet_id not in rerouted_shortest_path:
                map_is_updated_for_closed_lane = True
            else:
                map_is_updated_for_closed_lane = False
                break
    else:
        print(f"Invalid quantity of route updates found in bag file ({str(len(shortest_path_lanelets))} found, more than 1 expected)")

    # Print result statements and return success flags
    if initial_route_includes_closed_lane:
        print(f"FWZ-1 succeeded: all closed lanelets {str(closed_lanelets)} were in the initial route.")
    else:
        print(f"FWZ-1 failed: not all closed lanelets {str(closed_lanelets)} were in the initial route.")

    if map_is_updated_for_closed_lane:
        print(f"FWZ-8 succeeded: no closed lanelets {str(closed_lanelets)} were in the re-routed route.")
    else:
        print(f"FWZ-8 failed: at least 1 closed lanelet {str(closed_lanelets)} was in the re-routed route.")

    # Save shortest path data
    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "shortest_path_data.npz",
            closed_lanelets=closed_lanelets,
            shortest_path_lanelets=shortest_path_lanelets,
        )
        print(f"\nReroute data saved to: {save_data_dir}/shortest_path_data.npz")

    return initial_route_includes_closed_lane, map_is_updated_for_closed_lane


def check_speed_limits_in_geofence(
    mcap_path,
    time_enter_geofence,
    time_exit_geofence,
    advisory_speed_limit,
    save_data_dir=None
):
    """
    Checks that the vehicle processes the new speed limit after receiving a TCM with new workzone speed limit

    Args:
        mcap_path: Path to MCAP file
        time_enter_geofence: Time the vehicle entered the geofence
        time_exit_geofence: Time the vehicle exited the geofence
        advisory_speed_limit: New speed limit within the geofence

    Returns:
        is_successful: Boolean - True if lanelets travelled through within geofence have the advisory speed limit applied
    """
    geofence_topics = [INCOMING_GEOFENCE_CONTROL_TOPIC]
    route_state_topics = [GUIDANCE_ROUTE_STATE_TOPIC]

    speed_tolerance_ms = 0.03

    if not time_enter_geofence or not time_exit_geofence:
        print("FWZ-7 Failed: Vehicle never entered geofence - can not determine if workzone speed limit was processed")
        return False

    extracted_geofence_data = extract_mcap_data(
        mcap_path,
        geofence_topics,
        field_extractors={INCOMING_GEOFENCE_CONTROL_TOPIC: lambda msg: msg.tcm_v01}
    )
    incoming_geofence_timestamps, tcm_v01s = extracted_geofence_data[geofence_topics[0]]

    time_buffer_sec = 2 # Buffer in seconds after entering geofence and before exiting geofence for which advisory speed limit is observed
    extracted_route_state_data = extract_mcap_data(
        mcap_path,
        route_state_topics,
        start_time = (time_enter_geofence + time_buffer_sec),
        end_time = (time_exit_geofence - time_buffer_sec),
        field_extractors={GUIDANCE_ROUTE_STATE_TOPIC: lambda msg: (
                msg.speed_limit,
                msg.lanelet_id
            )}
    )

    guidance_route_timestamps, guidance_route_states = extracted_route_state_data[route_state_topics[0]]

    # Check that a TrafficControlMessage was published using the correct advisory speed limit
    has_communicated_advisory_speed_limit = False
    for tcm_v01 in tcm_v01s:
        if (tcm_v01.params.detail.choice == 12) and (advisory_speed_limit - speed_tolerance_ms <= tcm_v01.params.detail.maxspeed <= advisory_speed_limit + speed_tolerance_ms):
            has_communicated_advisory_speed_limit = True

    # Check that lanelets travelled through within the geofence have the expected advisory speed limit applied
    lanelet_speed_limits = []
    has_correct_geofence_lanelet_speed_limits = True
    for state in guidance_route_states:
        speed_limit = state[0]
        lanelet_id = state[1]
        lanelet_speed_limits.append((lanelet_id, speed_limit))
        if(abs(speed_limit-advisory_speed_limit) >= speed_tolerance_ms):
            print(f"Lanelet ID {lanelet_id} has speed limit of {speed_limit} m/s.")
            print(f"Does not match advisory speed limit of {advisory_speed_limit} m/s.")
            has_correct_geofence_lanelet_speed_limits = False
            break

    if has_communicated_advisory_speed_limit and has_correct_geofence_lanelet_speed_limits:
        print(f"FWZ-7 succeeded; System received and processed an advisory speed limit of {advisory_speed_limit} m/s.")
        is_successful = True
    else:
        print(f"FWZ-7 failed; System did not receive and process an advisory speed limit of {advisory_speed_limit} m/s.")
        is_successful = False

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "geofence_speed_limit_data.npz",
            advisory_speed_limit=advisory_speed_limit,
            lanelet_speed_limits=lanelet_speed_limits,
        )
        print(f"\nLanelet speed limit data saved to: {save_data_dir}")

    return is_successful


def check_reroute_duration(
    mcap_path,
    max_duration,
    save_data_dir=None
):
    """
    Check that after receiving a TCM with work zone information, the vehicle updates its route within max_duration

    Args:
        mcap_path: Path to MCAP file
        max_duration: Max amount of time (seconds) vehicle can take to update route

    Returns:
        is_successful: Boolean - True if vehicle updates route within max_duration seconds of receiving TCM with work zone information
    """
    topics=[INCOMING_GEOFENCE_CONTROL_TOPIC, GUIDANCE_ROUTE_TOPIC]

    # Obtain timestamps of each closed/restricted lane TCM
    closed_lane_tcm_receive_time = None
    restricted_lane_tcm_receive_time = None
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        field_extractors={
            INCOMING_GEOFENCE_CONTROL_TOPIC: lambda msg: msg.tcm_v01,
            GUIDANCE_ROUTE_TOPIC: lambda msg: msg.route_path_lanelet_ids
        }
    )
    timestamps, tcm_v01s = extracted_data[topics[0]]

    restricted_lane_present = False
    for timestamp, tcm_v01 in zip(timestamps, tcm_v01s):
        print(f"FWZ-11 (DEBUG): Received TCM at {timestamp} with detail choice {tcm_v01.params.detail.choice}")
        # Evaluate a received TCM for a closed lane
        if tcm_v01.params.detail.choice == 5:
            # Determine whether the closed lane is closed to passenger vehicles
            # Note: Lane is considered restricted if it is not closed to passenger vehicles
            is_restricted_lane = True
            for value in tcm_v01.params.vclasses:
                # vehicle_class 5 is passenger vehicles, 0 is any vehicle
                if value.vehicle_class == 5 or value.vehicle_class == 0:
                    is_restricted_lane = False

            # Set boolean flags for metric
            if is_restricted_lane:
                restricted_lane_present = True
                if restricted_lane_tcm_receive_time is None:
                    restricted_lane_tcm_receive_time = timestamp
                    print(f"FWZ-11 (DEBUG): Received restricted lane TCM at {timestamp}")
            else:
                if closed_lane_tcm_receive_time is None:
                    closed_lane_tcm_receive_time = timestamp
                    print(f"FWZ-11 (DEBUG): Received closed lane TCM at {timestamp}")

    # Get the time of each re-route
    route_generation_times = []
    timestamps, paths = extracted_data[topics[1]]
    for timestamp in timestamps:
        print(f"FWZ-11 (DEBUG): Generated route at {timestamp}")
        route_generation_times.append(timestamp)

    is_successful = True
    duration_reroute_after_closed_lane_tcm_received = None
    duration_reroute_after_restricted_lane_tcm_received = None
    # Make sure there is more than just the initial generated route
    if len(route_generation_times) > 1:
        # If there is a restricted lane, determine whether the closed or restricted came first. Then duration for each
        if restricted_lane_present:
            if closed_lane_tcm_receive_time <= restricted_lane_tcm_receive_time:
                closed_lane_tcm_received_first = True
            else:
                closed_lane_tcm_received_first = False

            if closed_lane_tcm_received_first:
                duration_reroute_after_closed_lane_tcm_received = (route_generation_times[1] - closed_lane_tcm_receive_time)
                duration_reroute_after_restricted_lane_tcm_received = (route_generation_times[2] - restricted_lane_tcm_receive_time)
            else:
                duration_reroute_after_closed_lane_tcm_received = (route_generation_times[2] - closed_lane_tcm_receive_time)
                duration_reroute_after_restricted_lane_tcm_received = (route_generation_times[1] - restricted_lane_tcm_receive_time)
        else:
            duration_reroute_after_closed_lane_tcm_received = (route_generation_times[1] - closed_lane_tcm_receive_time)

        # Determine whether the reroute duration is within the appropriate time
        if duration_reroute_after_closed_lane_tcm_received <= max_duration:
            print(f"FWZ-11 succeeded; rerouted {duration_reroute_after_closed_lane_tcm_received} sec after receiving closed lane TCM")
        else:
            print(f"FWZ-11 failed; rerouted {duration_reroute_after_closed_lane_tcm_received} sec after receiving closed lane TCM")
            is_successful = False
        if restricted_lane_present:
            if duration_reroute_after_restricted_lane_tcm_received <= max_duration:
                print(f"FWZ-11 succeeded; rerouted {duration_reroute_after_restricted_lane_tcm_received} sec after receiving restricted lane TCM")
            else:
                print(f"FWZ-11 failed; rerouted {duration_reroute_after_restricted_lane_tcm_received} sec after receiving restricted lane TCM")
                is_successful = False
    else:
        print(f"FWZ-11 failed; Invalid quantity of route updates found in bag file ({len(route_generation_times)} found, more than 1 expected)")
        is_successful = False

    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "reroute_duration_data.npz",
            paths=paths,
            route_generation_times=route_generation_times,
            duration_reroute_after_closed_lane_tcm_received=duration_reroute_after_closed_lane_tcm_received,
            duration_reroute_after_restricted_lane_tcm_received=duration_reroute_after_restricted_lane_tcm_received,
        )
        print(f"\nReroute duration data saved to: {save_data_dir}")

    return is_successful

def get_lateral_velocities(mcap_path, start_time=None, end_time=None):
    """
    Get lateral velocity of the vehicle with linear twist and pose orientation data

    Args:
        mcap_path: Path to MCAP file

    Returns:
        List of tuples containing (timestamp, lateral velocity)
    """
    topics = [LOCALIZATION_POSE_TOPIC, HARDWARE_VEHICLE_TWIST_TOPIC]
    lane_change_velocities = []

    # Get lateral velocity for every time stamp given vehicle velocity(x,y,z)
    #   and current orientation (x,y,z,w)
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            LOCALIZATION_POSE_TOPIC: lambda msg: msg.pose.orientation,
            HARDWARE_VEHICLE_TWIST_TOPIC: lambda msg: msg.twist.linear
        }
    )
    orientation_timestamps, orientations = extracted_data[topics[0]]
    twist_timestamps, twists = extracted_data[topics[1]]

    # Get reference heading (initial heading before lane change)
    if not orientation_timestamps.any():
        return []

    # Find the reference orientation 1 index before the lanechange starts
    ref_idx = max(0, bisect_left(orientation_timestamps, start_time) -1)
    reference_orientation = orientations[ref_idx] # First orientation as reference
    print(f"Reference Orientation: {reference_orientation} at {orientation_timestamps[ref_idx]} seconds")
    reference_quat = [reference_orientation.x, reference_orientation.y,
                      reference_orientation.z, reference_orientation.w]
    reference_rotation = r.from_quat(reference_quat)
    reference_yaw = reference_rotation.as_euler('xyz')[2] # Z-axis yaw

    print(f"Reference Yaw: {reference_yaw} radians")
    # For every twist, find the orientation by nearest timestamp
    for twist_timestamp, twist in zip(twist_timestamps, twists):
        idx = bisect_left(orientation_timestamps, twist_timestamp)
        if idx == 0:
            nearest_idx = 0
        elif idx == len(orientation_timestamps):
            nearest_idx = idx - 1
        else:
            before = orientation_timestamps[idx - 1]
            after = orientation_timestamps[idx]
            nearest_idx = idx - 1 if abs(twist_timestamp - before) < abs(twist_timestamp - after) else idx

        closest_orientation = orientations[nearest_idx]
        # Transform body velocity to world frame
        body_velocity = [twist.x, twist.y, twist.z]
        current_orientation = [closest_orientation.x, closest_orientation.y,
                                closest_orientation.z, closest_orientation.w]
        rotation = r.from_quat(current_orientation)
        world_velocity = rotation.apply(body_velocity)

        # Calculate lateral velocity relative to reference heading
        # Project world velocity onto lateral axis of reference heading
        lateral_velocity = (-world_velocity[0] * np.sin(reference_yaw) +
                            world_velocity[1] * np.cos(reference_yaw))
        lane_change_velocities.append((twist_timestamp, lateral_velocity))

    return lane_change_velocities

def check_lanechange_lateral_velocity(
    mcap_path,
    min_lat_velocity,
    max_lat_velocity,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None
):
    """
    Verifies that lateral velocity during a lane change is between min/max lateral velocity

    Args:
        mcap_path: Path to MCAP file
        min_lat_velocity: Minimum lateral velocity value during lane change
        max_lat_velocity: Maximum lateral velocity value during lane change

    Returns:
        is_successful: Boolean - True if all lateral velocities during lange change are between min/max lateral velocity
    """
    # Get times that vehicle is changing lanes
    planner_plugin = "cooperative_lanechange"
    intervals = get_planner_trajectory_intervals(mcap_path, planner_plugin)
    start_time = intervals[0][0] if intervals else 0
    # Get the lateral velocities starting at the first lane change
    lane_change_velocities = get_lateral_velocities(mcap_path, start_time)
    print("Got lateral velocities")

    lane_changes = []

    is_successful = True
    for idx, (lanechange_start, lanechange_end) in enumerate(intervals):
        lane_changes.append((lanechange_start, lanechange_end))
        for t, v in lane_change_velocities:
            if lanechange_start <= t <= lanechange_end:
                if min_lat_velocity >= abs(v) or max_lat_velocity <= abs(v):
                    is_successful = False
                    print(f"FWZ-13 Failed: Lateral velocity during lanechange {idx+1} was {abs(v)} m/s at {t} seconds. Not in the 0.5-1.25 m/s threshold")

    if is_successful and lane_changes:
        print(f"FWZ-13 Succeeded: All lateral velocities during lanechanges were within the 0.5-1.25 m/s threshold")
    else:
        is_successful = False
        print(f"FWZ-13 Failed: No lane changes recorded, can not evaluate lateral velocity during lane change")

    # Create visualizations
    times, velocities = zip(*lane_change_velocities)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, velocities, label='Lateral Velocity (m/s)', color='blue')

    for start, end in lane_changes:
        ax.axvspan(start, end, color='orange', alpha=0.3, label='Lane Change')
        dur = end - start
        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            ax.get_ylim()[1] * 0.9,
            f'{dur:.2f}s',
            ha='center',
            va='top',
            fontsize=9,
            color='black',
            backgroundcolor='white',
            alpha=0.7
        )

    ax.axhline(y=min_lat_velocity, color='green', linestyle='--')
    ax.axhline(y=max_lat_velocity, color='green', linestyle='--')
    ax.axhline(y=(min_lat_velocity*-1), color='green', linestyle='--')
    ax.axhline(y=(max_lat_velocity*-1), color='green', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lateral Velocity (m/s)')
    ax.set_title('Lateral Velocity over Time')
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()

    if save_plot_dir:
        save_plot_dir = Path(save_plot_dir)
        save_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_dir / "lateral_velocity_analysis.png")
        print(f"\nPlot saved to: {save_plot_dir}")
    else:
        plt.show()

    # Calculate, print, and save statistics
    velocities = np.array(velocities)
    stats = calculate_error_statistics(velocities)
    print_stats(stats, "Lateral Velocity Statistics")

    if save_stats_dir:
        save_stats_dir = Path(save_stats_dir)
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_full_path = save_stats_dir / "lateral_velocity_analysis.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    # Save data if requested
    if save_data_dir:
        save_data_dir = Path(save_data_dir)
        save_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_data_dir / "lateral_velocity_data.npz",
            lane_change_velocities=lane_change_velocities,
            intervals=intervals,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    return is_successful, plt.gcf(), stats

def check_lanechange_duration(
    mcap_path,
    start_time,
    max_lanechange_duration,
    save_stats_dir,
    save_data_dir
):
    """
    Verifies that vehicle completes all lane changes within max_lanechange_duration

    Args:
        mcap_path: Path to MCAP file
        start_time: Start time to begin analysis
        max_lanechange_duration: Maximum amount of time (sec) to complete lane change

    Returns:
        is_successful: Boolean - True if all lane changes are completed within max_lanechange_duration
    """
    # Get all lane change times
    planner_plugin = "cooperative_lanechange"
    intervals = get_planner_trajectory_intervals(mcap_path, planner_plugin, start_time)
    durations = []
    stats = None

    is_successful = True
    for i, (start, end) in enumerate(intervals):
        duration = end - start
        durations.append(duration)
        if duration > max_lanechange_duration:
            print(f"FWZ-14 (LC {i+1}) failed; lane change completed in {duration:.2f} seconds")
            is_successful = False

    if is_successful and durations:
        print(f"FWZ-14 Succeeded: all lane changes completed in less than {max_lanechange_duration} seconds")

        durations = np.array(durations)
        stats = calculate_error_statistics(durations)
        print_stats(stats, "Lane Change Duration Statistics")

        if save_stats_dir:
            save_stats_dir = Path(save_stats_dir)
            save_stats_dir.mkdir(parents=True, exist_ok=True)
            stats_full_path = save_stats_dir / "lane_change_duration_analysis.json"
            with open(stats_full_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Stats saved to: {save_stats_dir}")

        if save_data_dir:
            save_data_dir = Path(save_data_dir)
            save_data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_data_dir / "lane_change_duration_data.npz",
                intervals=intervals,
                durations=durations,
                stats=stats,
            )
            print(f"\nData saved to: {save_data_dir}")
    else:
        print(f"FWZ-14 Failed: No lane changes recorded, can not evaluate duration of lane change")
        return False, None

    return is_successful, stats


def find_accel_period(accelerations, time_start, deceleration):
    """
    Helper function to find the beginning and end of acceleration/deceleration periods as well as the values during that time

    Args:
        accelerations: Tuple of lists with timestamps and accelerations/decelerations
        time_start: Beginning of the time frame to be analyzed
        deceleration: Boolean - True if looking for deceleration period

    Returns:
        time_start_period: start of the acceleration/deceleration period
        time_end_period: end of the acceleration/deceleration period
        accels: list of acceleration/deceleration values
    """
    # Determines if a acceleration value is valid based on whether we want acceleration or deceleration
    is_valid = (lambda x: x < 0) if deceleration else (lambda x: x > 0)

    # Arbitrary number of consecutive accelerations/decelerations needed to be considered the start
    num_consecutive = 10
    accels = []
    consec_count = 0
    time_begin_period = None
    time_end_period = None

    # Pulls accelerations starting at time_start
    filtered_accelerations = [entry for entry in accelerations if entry[0] > time_start]

    for timestamp, accel in filtered_accelerations:
        if is_valid(accel):
            if consec_count == 0:
                time_begin_period = timestamp
                accels = [accel]
            else:
                accels.append(accel)
            consec_count += 1
        else:
            if consec_count >= num_consecutive:
                time_end_period = timestamp
                return time_begin_period, time_end_period, accels
            consec_count = 0
            time_begin_period = None
            accels = []

    # Handle case where sequence continues to end
    if consec_count >= num_consecutive:
        time_end_period = filtered_accelerations[-1][0]
        return time_begin_period, time_end_period, accels

    return None, None, []

def check_time_to_begin_deceleration(speed_limit_changes, response_times, response_threshold, save_stats_dir, save_data_dir):
    """
    Verifies that all slow down speed limit changes are responded to within a threshold

    Args:
        speed_limit_changes: List of tuple containing (time of speed limit change, old speed limit, new speed limit)
        response_times: List of speed limit change response times
        response_threshold: Max value vehicle can take to respond to speed limit change (sec)

    Returns:
        is_successful: Boolean - True if all slow down speed limit change responses are within the threshold
    """
    if not speed_limit_changes:
        print(f"FWZ-22 Failed: No speed limit changes recorded. Can not evaluate response time to deceleration command.")
        return False

    is_successful = True
    deceleration_responses = []
    for i, (speed_change, response_time) in enumerate(zip(speed_limit_changes, response_times)):
        if speed_change[1] > speed_change[2]:
            deceleration_responses.append(response_time)
            if response_time > response_threshold:
                is_successful = False
                index = i

    if is_successful:
        print(f"FWZ-22 Succeeded: All deceleration commands were issued less than {response_threshold} sec after entering the geofenced area.")
    else:
        print(f"FWZ-22 Failed: Speed limit change {index} deceleration command was issued late. Expected {response_threshold} sec after entering the geofenced area, was {response_time} sec.")

    if deceleration_responses:

        deceleration_responses = np.array(deceleration_responses)
        stats = calculate_error_statistics(deceleration_responses)
        print_stats(stats, 'Deceleration Command Response Time')

        if save_stats_dir:
            save_stats_dir = Path(save_stats_dir)
            save_stats_dir.mkdir(parents=True, exist_ok=True)
            stats_full_path = save_stats_dir / "deceleration_response_analysis.json"
            with open(stats_full_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Stats saved to: {save_stats_dir}")

        if save_data_dir:
            save_data_dir = Path(save_data_dir)
            save_data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_data_dir / "deceleration_response_data.npz",
                speed_limit_changes=speed_limit_changes,
                deceleration_responses=deceleration_responses,
                stats=stats,
            )
            print(f"\nData saved to: {save_data_dir}")

    return is_successful

def check_speed_before_workzone(
    mcap_path,
    start_time,
    end_time,
    workzone_lanelet_id,
    advisory_speed_limit_ms,
    speed_limit_threshold_ms
):
    """
    Verifies that vehicle speed matches the advisory speed limit upon entering geofenced area

    Args:
        mcap_path: Path to MCAP file
        start_time: Start time to look at
        end_time: End time to look at
        workzone_lanelet_id: List of workzone lanelet ids
        advisory_speed_limit_ms: Advisory speed limit of workzone in m/s
        speed_limit_threshold_ms: Threshold vehicle speed must be within the advisory speed limit

    Returns:
        is_successful: Boolean - True if vehicle speed is at advisory speed limit +- threshold
    """
    route_state_topics = [GUIDANCE_ROUTE_STATE_TOPIC]
    vehicle_twist_topics = [HARDWARE_VEHICLE_TWIST_TOPIC]
    min_speed_limit_ms = advisory_speed_limit_ms - speed_limit_threshold_ms
    max_speed_limit_ms = advisory_speed_limit_ms + speed_limit_threshold_ms
    time_enter_workzone = 0.0

    if not workzone_lanelet_id:
        print(f"FWZ-23 Failed: Passed in list of closed lanelets was empty. Can not evaluate if advisory speed limit was achieved upon entering geofence. Please populate closed lanelets")
        return False

    extracted_data = extract_mcap_data(
        mcap_path,
        route_state_topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={GUIDANCE_ROUTE_STATE_TOPIC: lambda msg: msg.lanelet_id}
    )
    timestamps, lanelets = extracted_data[route_state_topics[0]]

    # Get the time the vehicle entered the workzone lanelet
    for timestamp, lanelet in zip(timestamps, lanelets):
        if lanelet == workzone_lanelet_id:
            time_enter_workzone = timestamp


    extracted_data = extract_mcap_data(
        mcap_path,
        vehicle_twist_topics,
        start_time=time_enter_workzone,
        end_time=end_time,
        field_extractors={HARDWARE_VEHICLE_TWIST_TOPIC: lambda msg: msg.twist}
    )

    # Get the first speed the vehicle was traveling in the workzone lanelet
    timestamps, twists = extracted_data[vehicle_twist_topics[0]]
    for timestamp, twist in zip(timestamps, twists):
        vehicle_speed_workzone_entrance_ms = twist.linear.x
        break

    is_successful = False
    if(min_speed_limit_ms <= vehicle_speed_workzone_entrance_ms <= max_speed_limit_ms):
        print(f"FWZ-23 succeeded: Vehicle travelling at {vehicle_speed_workzone_entrance_ms} m/s when entering the workzone.")
        is_successful = True
    else:
        print(f"FWZ-23 failed: Vehicle travelling at {vehicle_speed_workzone_entrance_ms} m/s when entering the workzone. Should be between {min_speed_limit_ms} m/s and {max_speed_limit_ms}.")

    return is_successful

def check_steady_state_after_geofence(
    mcap_path,
    time_begin_acceleration_after_geofence,
    time_end_engagement,
    original_speed_limit_ms,
    min_time_at_steady_state=5.0,
    threshold_speed_limit_offset=0.89408
):
    """
    Verifies that vehicle maintains steady state for at least 5 seconds after exiting geofenced area

    Args:
        mcap_path: Path to MCAP file
        time_begin_acceleration_after_geofence: Start time to look for steady state
        time_end_engagement: End time of engagement
        original_speed_limit_ms: Original speed limit in m/s
        min_time_at_steady_state: Minimum time required at steady state in seconds (default: 5.0)
        threshold_speed_limit_offset: Speed threshold offset in m/s for steady state detection (default: 0.89408 m/s = 2 mph)

    Returns:
        is_successful: Boolean - True if vehicle was at steady state for at least the minimum required time
    """
    # (m/s) Threshold offset of vehicle speed to speed limit to be considered at steady state
    min_steady_state_speed = original_speed_limit_ms - threshold_speed_limit_offset
    max_steady_state_speed = original_speed_limit_ms + threshold_speed_limit_offset

    vehicle_twist_topics = [HARDWARE_VEHICLE_TWIST_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        vehicle_twist_topics,
        start_time=time_begin_acceleration_after_geofence,
        end_time=time_end_engagement,
        field_extractors={HARDWARE_VEHICLE_TWIST_TOPIC: lambda msg: msg.twist}
    )

    timestamps, twists = extracted_data[vehicle_twist_topics[0]]

    has_reached_steady_state = False
    time_start_steady_state = 0.0
    time_end_steady_state = 0.0

    for timestamp, twist in zip(timestamps, twists):
        current_speed = twist.linear.x

        if (min_steady_state_speed <= current_speed <= max_steady_state_speed) and not has_reached_steady_state:
            time_start_steady_state = timestamp
            has_reached_steady_state = True

        if not (min_steady_state_speed <= current_speed <= max_steady_state_speed) and has_reached_steady_state:
            time_end_steady_state = timestamp
            break
        elif has_reached_steady_state:
            time_end_steady_state = timestamp

    if has_reached_steady_state:
        time_at_steady_state = time_end_steady_state - time_start_steady_state
    else:
        time_at_steady_state = 0.0

    is_successful = False
    if time_at_steady_state >= min_time_at_steady_state:
        print(f"FWZ-29 succeeded: Vehicle was at steady state for {time_at_steady_state} seconds after exiting the geofence (required: {min_time_at_steady_state} seconds)")
        is_successful = True
    else:
        print(f"FWZ-29 failed: Vehicle was at steady state for {time_at_steady_state} seconds after exiting the geofence (required: {min_time_at_steady_state} seconds)")

    return is_successful

def create_geofence_acceleration_plot(accelerations, sec_accelerations, time_enter_geofence, time_exit_geofence, save_plots_dir=None):
    """
    Creates plots comparing instantaneous acceleration and 1-sec average acceleration over time. Marks the times the vehicle entered & exited the geofence
    Saves the plot to save_plots_dir

    Args:
        accelerations: List of tuples containing (timestamp, instantaneous acceleration)
        sec_accelerations: List of tuples containing (timestamp, 1-sec average acceleration)
        time_enter_geofence: Timestamp the vehicle entered the geofence
        time_exit_geofence: Timestamp the vehicle exited the geofence

    """
    if time_enter_geofence and time_exit_geofence:
        acc_times = []
        acc_values = []
        sec_acc_times = []
        sec_acc_values = []

        for time, acc in accelerations:
            if ((time_enter_geofence - 5) <= time <= (time_exit_geofence + 5)):
                acc_times.append(time)
                acc_values.append(acc)

        for time, acc in sec_accelerations:
            if ((time_enter_geofence - 5) <= time <= (time_exit_geofence + 5)):
                sec_acc_times.append(time)
                sec_acc_values.append(acc)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(acc_times, acc_values, label='Instantaneous Acceleration', color='blue')
        ax1.axvline(x=time_enter_geofence, color='gray', linestyle='--')
        ax1.axvline(x=time_exit_geofence, color='gray', linestyle='--')
        ax1.text(time_enter_geofence, ax1.get_ylim()[1], f'Entered Geofence: {time_enter_geofence:.2f}', color='gray', ha='center', va='bottom')
        ax1.text(time_exit_geofence, ax1.get_ylim()[1], f'Exited Geofence: {time_exit_geofence:.2f}', color='gray', ha='center', va='bottom')
        ax1.set_ylabel('Instantaneous Acceleration (m/s^2)')
        ax1.set_title('Instantaneous Acceleration over Time', pad=20)
        ax1.grid(True)

        ax2.plot(sec_acc_times, sec_acc_values, label='1-Sec Average Acceleration', color='red')
        ax2.axvline(x=time_enter_geofence, color='gray', linestyle='--')
        ax2.axvline(x=time_exit_geofence, color='gray', linestyle='--')
        ax2.text(time_enter_geofence, ax2.get_ylim()[1], f'Entered Geofence: {time_enter_geofence:.2f}', color='gray', ha='center', va='bottom')
        ax2.text(time_exit_geofence, ax2.get_ylim()[1], f'Exited Geofence: {time_exit_geofence:.2f}', color='gray', ha='center', va='bottom')
        ax2.set_ylabel('1-Sec Average Acceleration')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('1-Sec Average Acceleration over Time', pad=20)
        ax2.grid(True)

        plt.legend()
        plt.tight_layout()

        if save_plots_dir:
            save_plots_dir = Path(save_plots_dir)
            save_plots_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_plots_dir / "geofence_acceleration.png")
            print(f"\nPlot saved to: {save_plots_dir}")
        else:
            plt.show()
    else:
        print(f"Vehicle never entered geofence, can not create geofence plots.")


def check_deceleration_for_geofence(time_enter_geofence, accelerations, max_deceleration):
    """
    Verifies that the average acceleration over a deceleration period is not greater than the max

    Args:
        time_enter_geofence: time the vehicle entered the geofence
        accelerations: Tuple of lists with timestamps and accelerations/decelerations
        max_decleration: Max deceleration of the vehicle (m/s^2)

    Returns:
        is_successful: Boolean - True if average acceleration over a deceleration period is less than the max
    """
    if not time_enter_geofence:
        print(f"FWZ-24 Failed: Vehicle never entered geofence, can not evaluate deceleration upon entering geofence")
        return False

    find_decelerations = True
    time_begin_deceleration_in_geofence, time_end_deceleration_in_geofence, decelerations = find_accel_period(accelerations, time_enter_geofence, find_decelerations)

    # Handle case where deceleration period is never met
    if not decelerations:
        print(f"FWZ-24 Failed: Deceleration period never began upon entering geofence")
        return False

    is_successful = False
    print(f"Deceleration timeframe upon entering geofence found. Start: {time_begin_deceleration_in_geofence} End: {time_end_deceleration_in_geofence}")

    average_deceleration = sum(decelerations) / len(decelerations)
    print(f"Average Deceleration: {average_deceleration} m/s^2")

    if(abs(average_deceleration) > abs(max_deceleration)):
        print(f"FWZ-24 Failed: Average deceleration upon entering the geofence is {average_deceleration} m/s^2. This is greater than the maximum of {max_deceleration} m/s^2")
    else:
        print(f"FWZ-24 Succeeded: Average deceleration upon entering the geofence is {average_deceleration} m/s^2. This is within the maximum of {max_deceleration} m/s^2")
        is_successful = True

    return is_successful

def check_time_to_begin_acceleration(speed_limit_changes, response_times, response_threshold, save_stats_dir, save_data_dir):
    """
    Verifies that all speed up speed limit changes are responded to within a threshold

    Args:
        speed_limit_changes: List of tuple containing (time of speed limit change, old speed limit, new speed limit)
        response_times: List of speed limit change response times
        response_threshold: Max value vehicle can take to respond to speed limit change (sec)

    Returns:
        is_successful: Boolean - True if all speed up speed limit change responses are within the threshold
    """
    if not speed_limit_changes:
        print(f"FWZ-25 Failed: No speed limit changes recorded. Can not evaluate response time to acceleration command.")
        return False

    is_successful = True
    acceleration_responses = []
    for i, (speed_change, response_time) in enumerate(zip(speed_limit_changes, response_times)):
        if speed_change[1] < speed_change[2]:
            acceleration_responses.append(response_time)
            if response_time > response_threshold:
                is_successful = False
                index = i

    if is_successful:
        print(f"FWZ-25 Succeeded: All acceleration commands were issued less than {response_threshold} sec after exiting the geofenced area.")
    else:
        print(f"FWZ-25 Failed: Speed limit change {index} acceleration command was issued late. Expected {response_threshold} sec after exiting the geofenced area, was {response_time} sec.")

    if acceleration_responses:
        acceleration_responses = np.array(acceleration_responses)
        stats = calculate_error_statistics(acceleration_responses)
        print_stats(stats, 'Acceleration Command Response Time')

        if save_stats_dir:
            save_stats_dir = Path(save_stats_dir)
            save_stats_dir.mkdir(parents=True, exist_ok=True)
            stats_full_path = save_stats_dir / "acceleration_response_analysis.json"
            with open(stats_full_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Stats saved to: {save_stats_dir}")

        if save_data_dir:
            save_data_dir = Path(save_data_dir)
            save_data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_data_dir / "acceleration_response_data.npz",
                speed_limit_changes=speed_limit_changes,
                acceleration_responses=acceleration_responses,
                stats=stats,
            )
            print(f"\nData saved to: {save_data_dir}")

    return is_successful

def check_acceleration_after_geofence(time_exit_geofence, accelerations, min_average_acceleration, section_accelerations, max_section_acceleration):
    """
    Verifies that the average acceleration over an acceleration period is not less than the average min
    and that the average over any 1-second section is not greater than the section max

    Args:
        time_exit_geofence: time the vehicle exited the geofence
        accelerations: Tuple of lists with timestamps and accelerations/decelerations
        min_average_acceleration: smallest average acceleration allowed over the entire acceleration period (m/s^2)
        section_accelerations: Tuple of lists with timestamps and average accelerations over any given 1 second section
        max_section_acceleration: Max acceleration of the vehilce allowed over any 1-second section(m/s^2)

    Returns:
        is_successful: Boolean - True if average acceleration over a deceleration period is less than the max
    """
    if not time_exit_geofence:
        print(f"FWZ-26 Failed: Vehicle never entered geofence, can not evaluate acceleration after leaving geofence")
        return False

    find_decelerations = False
    time_begin_acceleration_after_geofence, time_end_acceleration_after_geofence, exit_accelerations = find_accel_period(accelerations, time_exit_geofence, find_decelerations)

    # Handle case where acceleration period is never met
    if not accelerations:
        print(f"FWZ-26 Failed: Acceleration period never began upon exiting geofence")
        return False, None

    is_successful = True
    print(f"Acceleration timeframe upon exiting geofence found. Start: {time_begin_acceleration_after_geofence} End: {time_end_acceleration_after_geofence}")

    average_acceleration = sum(exit_accelerations) / len(exit_accelerations)
    print(f"Average Acceleration: {average_acceleration} m/s^2")

    if(abs(average_acceleration) < abs(min_average_acceleration)):
        print(f"FWZ-26 Failed: Average acceleration upon exiting the geofence is {average_acceleration} m/s^2. This is less than the minimum of {min_average_acceleration} m/s^2")
        return False

    # Only get accelerations in the acceleration time from
    filtered_section_accelerations = [entry for entry in section_accelerations if ((entry[0] >= time_begin_acceleration_after_geofence) and (entry[0] < time_end_acceleration_after_geofence))]
    for timestamp, accel in filtered_section_accelerations:
        if accel > max_section_acceleration:
            print(f"FWZ-26 Failed: Average acceleration at the {timestamp} 1-second interval is {accel} m/s^2. This is greater than the maximum of {max_section_acceleration} m/s^2")
            return False

    print(f"FWZ-26 Succeeded: Average acceleration upon exiting the geofence is {average_acceleration} m/s^2. This is greater than the minimum of {min_average_acceleration} m/s^2")
    print(f"FWZ-26 Succeeded: All 1-second averages are below the maximum of {max_section_acceleration} m/s^2.")

    return is_successful


# More guidance specific analysis scripts to come ....

def main():
    """
    Main function to run the analysis scripts.
    """
    # Example usage of the functions
    mcap_path = "/path/to/your/mcap_file.mcap"
    run_speed_limit_change_response_analysis(mcap_path)

if __name__ == "__main__":
    main()
