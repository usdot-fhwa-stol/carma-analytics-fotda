from parse_ros2_bags import open_bagfile, extract_mcap_data
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.spatial import KDTree


def run_crosstrack_analysis(mcap_path, save_data_dir=None, save_plot_dir=None):
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


# More guidance specific analysis scripts to come ....
