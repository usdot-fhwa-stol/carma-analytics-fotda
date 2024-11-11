from parse_ros2_bags import open_bagfile, extract_mcap_data
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.spatial import KDTree
import json
from utils import calculate_error_statistics, print_stats, align_time_series

STD_DEV_LABEL_STRING = "±1 Std Dev"
TIME_SECONDS_LABEL_STRING = "Time (seconds)"
# ROS Topics Constants
GUIDANCE_STATE_TOPIC = "/guidance/state"
GUIDANCE_ROUTE_STATE_TOPIC = "/guidance/route_state"
GUIDANCE_PLAN_TRAJECTORY_TOPIC = "/guidance/plan_trajectory"
GUIDANCE_CONTROL_CMD_TOPIC = "/guidance/ctrl_cmd"

LOCALIZATION_POSE_TOPIC = "/localization/current_pose"

HARDWARE_VEHICLE_STATUS_TOPIC = "/hardware_interface/vehicle_status"
HARDWARE_VEHICLE_TWIST_TOPIC = "/hardware_interface/vehicle/twist"
HARDWARE_PACMOD_STEER_REPORT_TOPIC = "/hardware_interface/as/pacmod/parsed_tx/steer_rpt"
HARDWARE_PACMOD_STEER_CMD_TOPIC = "/hardware_interface/as/pacmod/as_rx/steer_cmd"


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
    is_passed = float(stats["median"]) < error_threshold_to_pass_meter

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
        stats_full_path = save_stats_dir / "cross_track_stats_result.json"
        with open(stats_full_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {save_stats_dir}")

    if save_data_dir:
        np.savez(
            save_data_dir / "cross_track_extracted_numpy_data.npz",
            timestamps=timestamps,
            cross_tracks=cross_tracks,
            stats=stats,
        )
        print(f"\nData saved to: {save_data_dir}")

    if save_plot_dir:
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
    Analyzes steering performance by comparing commanded vs actual steering values at PACMod level.

    Args:
        mcap_path: Path to MCAP file
        error_threshold_to_pass: Maximum allowed steering error
        start_time: Time to start the analysis
        end_time: Time to end the analysis
        save_stats_dir: Directory to save analysis stats
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/hardware_interface/as/pacmod/parsed_tx/steer_rpt,
                /hardware_interface/as/pacmod/as_rx/steer_cmd]
    """
    topics = [HARDWARE_PACMOD_STEER_REPORT_TOPIC, HARDWARE_PACMOD_STEER_CMD_TOPIC]

    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        start_time=start_time,
        end_time=end_time,
        field_extractors={
            topics[0]: lambda msg: msg.output,
            topics[1]: lambda msg: msg.command,
        },
    )

    actual_timestamps, actual_values = extracted_data[topics[0]]
    cmd_timestamps, cmd_values = extracted_data[topics[1]]

    # Convert to numpy arrays
    actual_timestamps = np.array(actual_timestamps)
    actual_values = np.array(actual_values)
    cmd_timestamps = np.array(cmd_timestamps)
    cmd_values = np.array(cmd_values)

    # Align the time series
    common_timestamps, aligned_cmd_values, aligned_actual_values = align_time_series(
        cmd_timestamps, cmd_values, actual_timestamps, actual_values
    )

    # Calculate differences between commanded and actual steering wheel values
    error_values = np.abs(aligned_cmd_values - aligned_actual_values)

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
        cmd_timestamps,
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

    plt.title("Steering Wheel Value Comparison")
    plt.xlabel(TIME_SECONDS_LABEL_STRING)
    plt.ylabel("Steering Wheel Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot error over time
    plt.subplot(2, 1, 2)
    plt.plot(
        common_timestamps,
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
        common_timestamps,
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
            common_timestamps=common_timestamps,
            cmd_timestamps=cmd_timestamps,
            cmd_values=cmd_values,
            actual_timestamps=actual_timestamps,
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

    return (is_passed, stats, plt.gcf(), error_values, common_timestamps)


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


# More guidance specific analysis scripts to come ....
