from parse_ros2_bags import open_bagfile, extract_mcap_data
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.spatial import KDTree
import json


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
        if isinstance(value, (int, bool)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.{decimal_places}f}")


def run_crosstrack_analysis(
    mcap_path,
    error_threshold_to_pass_meter=2.0,
    save_stats_dir=None,
    save_data_dir=None,
    save_plot_dir=None,
):
    """
    Analyzes cross trask error from CARMA Platform's internal route logic.

    Args:
        mcap_path: Path to MCAP file
        save_data_dir: Directory to save extracted data
        save_plot_dir: Directory to save generated plots
    Deps:
        Topics: [/localization/current_pose]
        Msgs: carma_planning_msgs
    """

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

    # Pass or no pass
    is_passed = float(stats["median"]) < error_threshold_to_pass_meter

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

    return (is_passed, plt.gcf(), cross_tracks, timestamps)


# More guidance specific analysis scripts to come ....
