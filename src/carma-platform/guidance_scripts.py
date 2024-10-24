from parse_ros2_bags import open_bagfile, extract_mcap_data
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.spatial import KDTree


def run_crosstrack_analysis(mcap_path, save_data_dir=None, save_plot_dir=None):
    topics = ["/guidance/route_state"]
    extracted_data = extract_mcap_data(
        mcap_path, topics, {"/guidance/route_state": lambda msg: msg.cross_track}
    )
    timestamps, cross_tracks = extracted_data[topics[0]]

    # Calculate statistics
    stats = {
        "minimum": np.min(cross_tracks),
        "maximum": np.max(cross_tracks),
        "median": np.median(cross_tracks),
        "std_dev": np.std(cross_tracks),
        "mean": np.mean(cross_tracks),
        "sample_count": len(cross_tracks),
        "rms": np.sqrt(np.mean(np.square(cross_tracks))),
    }

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cross_tracks, "b-", label="Cross Track Error", linewidth=1)
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.fill_between(
        timestamps,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label="±1 Std Dev",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Cross Track Error (m)")
    plt.title("Route State Cross Track Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Print
    print("\nCross Track Error Statistics:")
    print(f"Minimum: {stats['minimum']:.4f} m")
    print(f"Maximum: {stats['maximum']:.4f} m")
    print(f"Median:  {stats['median']:.4f} m")
    print(f"Mean:    {stats['mean']:.4f} m")
    print(f"RMS:     {stats['rms']:.4f} m")
    print(f"Std Dev: {stats['std_dev']:.4f} m")
    print(f"Sample Count: {stats['sample_count']}")

    if save_data_dir:
        np.savez(
            save_data_dir / "extracted_numpy_data.npz",
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

    return (stats, plt.gcf(), cross_tracks, timestamps)


def run_turn_accuracy_analysis(mcap_path, save_data_dir=None, save_plot_dir=None):
    """
    Analyzes turn accuracy by comparing actual path to planned trajectory using spline fitting.

    Args:
        mcap_path: Path to MCAP file
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    """
    # Extract actual and planned paths
    actual_path = []
    planned_path = []
    last_planned_point = None

    # Extract messages from MCAP
    topics = ["/localization/current_pose", "/guidance/plan_trajectory"]
    extracted_data = extract_mcap_data(
        mcap_path,
        topics,
        {
            "/localization/current_pose": lambda msg: (
                msg.pose.position.x,
                msg.pose.position.y,
            ),
            "/guidance/plan_trajectory": lambda msg: [
                (p.x, p.y) for p in msg.trajectory_points[1:]
            ],  # Skip first point
        },
    )

    # print(extracted_data[topics[1]])
    # Process actual path
    timestamps, odom = extracted_data[topics[0]]

    for point in odom:
        actual_path.append([point[0], point[1]])
    actual_path = np.array(actual_path)

    # Process planned path with duplicate removal
    timestamps, traj_plans = extracted_data[topics[1]]

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

    planned_path = np.array(planned_path)

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
    stats = {
        "minimum": np.min(distances),
        "maximum": np.max(distances),
        "median": np.median(distances),
        "mean": np.mean(distances),
        "std_dev": np.std(distances),
        "rms": np.sqrt(np.mean(np.square(distances))),
        "sample_count": len(distances),
    }

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot paths
    plt.subplot(2, 1, 1)
    plt.plot(
        planned_path[:, 0], planned_path[:, 1], "b-", label="Planned Path", linewidth=1
    )
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

    # Plot error over distance
    plt.subplot(2, 1, 2)
    path_distance = np.cumsum(
        np.sqrt(np.sum(np.diff(actual_path, axis=0) ** 2, axis=1))
    )
    path_distance = np.insert(path_distance, 0, 0)

    plt.plot(path_distance, distances, "b-", label="Distance Error", linewidth=1)
    plt.axhline(y=stats["median"], color="r", linestyle="--", label="Median")
    plt.fill_between(
        path_distance,
        stats["median"] - stats["std_dev"],
        stats["median"] + stats["std_dev"],
        alpha=0.2,
        color="r",
        label="±1 Std Dev",
    )

    plt.title("Turn Accuracy Error Over Distance")
    plt.xlabel("Distance Traveled (m)")
    plt.ylabel("Error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Print statistics
    print("\nTurn Accuracy Statistics:")
    print(f"Minimum Error: {stats['minimum']:.4f} m")
    print(f"Maximum Error: {stats['maximum']:.4f} m")
    print(f"Median Error: {stats['median']:.4f} m")
    print(f"Mean Error:   {stats['mean']:.4f} m")
    print(f"RMS Error:    {stats['rms']:.4f} m")
    print(f"Std Dev:      {stats['std_dev']:.4f} m")
    print(f"Sample Count: {stats['sample_count']}")

    # Save data if requested
    if save_data_dir:
        save_path = Path(save_data_dir)
        np.savez(
            save_path / "turn_accuracy_data.npz",
            actual_path=actual_path,
            planned_path=planned_path,
            spline_points=spline_points,
            distances=distances,
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

    return (stats, plt.gcf(), distances, path_distance)
