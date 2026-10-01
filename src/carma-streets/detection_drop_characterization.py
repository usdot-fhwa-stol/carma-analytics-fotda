import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from dateutil import tz
from matplotlib import pyplot as plt

from parse_kafka_logs import KafkaLogMessageType, parse_kafka_logs_as_type

DEFAULT_DETECTION_RATE_HZ = 10.0
DEFAULT_MAX_FIRST_DETECTION_DELAY_SEC = 10.0
DEFAULT_TIMEZONE = "America/New_York"
PLOT_NAME = "detection_drop_characterization.png"
CSV_NAME = "detection_drop_characterization.csv"


def to_epoch_sec(entry_time, tz_name: str = DEFAULT_TIMEZONE) -> float:
    """Convert a recorded entry time to epoch seconds.

    Args:
        entry_time: Epoch seconds, or a datetime string (e.g. "2026-09-03 14:03:21"). Naive
            datetime strings are read in tz_name.
        tz_name (str): Timezone for naive datetime strings. Default is America/New_York, matching
            parse_kafka_logs.py.

    Returns:
        float: Epoch seconds
    """
    try:
        return float(entry_time)
    except ValueError:
        entry_datetime = datetime.fromisoformat(entry_time)
    if entry_datetime.tzinfo is None:
        entry_datetime = entry_datetime.replace(tzinfo=tz.gettz(tz_name))
    return entry_datetime.timestamp()


def load_detection_frame_times(detection_log_path: Path) -> np.ndarray:
    """Read a detected object Kafka log into sorted, unique detection frame times.

    A frame detecting several objects logs one message per object, so timestamps are
    de-duplicated to count each camera frame once.

    Args:
        detection_log_path (Path): Path to the detected object Kafka topic log

    Returns:
        np.ndarray: Sorted unique frame times in epoch seconds
    """
    if not detection_log_path.is_file():
        raise FileNotFoundError(f"Detection log {detection_log_path} does not exist")
    msgs = parse_kafka_logs_as_type(detection_log_path, KafkaLogMessageType.DetectedObject)
    return np.unique([msg.json_message["timestamp"] for msg in msgs]) / 1e3


def find_dropped_spans(frame_offsets_sec: np.ndarray, duration_sec: float, frame_interval_sec: float) -> list:
    """Find spans within a run's window where one or more frames are missing.

    Args:
        frame_offsets_sec (np.ndarray): Received frame times, in seconds since the run's first detection
        duration_sec (float): Length of the run's window in seconds
        frame_interval_sec (float): Expected seconds between frames

    Returns:
        list: (start, end) spans, in seconds since the first detection, longer than 1.5 frame intervals
    """
    if len(frame_offsets_sec) == 0:
        return [(0.0, duration_sec)]
    edges = np.append(frame_offsets_sec, duration_sec)
    gaps = np.flatnonzero(np.diff(edges) > 1.5 * frame_interval_sec)
    return [(edges[i], edges[i + 1]) for i in gaps]


def characterize_detection_drops(
    detection_log_path: Path,
    run_entry_times: list,
    run_durations_sec: list,
    detection_rate_hz: float = DEFAULT_DETECTION_RATE_HZ,
    max_first_detection_delay_sec: float = DEFAULT_MAX_FIRST_DETECTION_DELAY_SEC,
    tz_name: str = DEFAULT_TIMEZONE,
    plots_dir: Path = None,
) -> dict:
    """Characterize dropped detection frames over runs where a pedestrian stood in the detection
    zone for a known duration.

    Recorded entry times are only approximate, so each run's window starts at the first detection
    at or after its entry time and spans that run's duration. Frames received in the window are
    compared against duration * detection_rate_hz expected frames.

    Args:
        detection_log_path (Path): Path to the detected object (v2xhub_sim_sensor_detected_object) Kafka log
        run_entry_times (list): Recorded pedestrian entry time for each run (see to_epoch_sec)
        run_durations_sec (list): Seconds the pedestrian stayed in the detection zone, one per run
        detection_rate_hz (float): Expected detection frame rate. Default is 10 Hz.
        max_first_detection_delay_sec (float): A run whose first detection comes later than this after
            its entry time is treated as never detected (all frames dropped), rather than borrowing the
            next run's detections. Default is 10 s.
        tz_name (str): Timezone for naive entry time strings. Default is America/New_York.
        plots_dir (Path): Directory to save the plot and per-run csv to. If not given, the plot is shown.

    Returns:
        dict: Totals across all runs and a per-run list of results
    """
    if len(run_entry_times) != len(run_durations_sec):
        raise ValueError(
            f"Got {len(run_entry_times)} entry times but {len(run_durations_sec)} durations; need one duration per run"
        )

    frame_times = load_detection_frame_times(Path(detection_log_path))
    frame_interval_sec = 1.0 / detection_rate_hz

    runs = []
    run_frame_offsets = []
    for run_number, (entry_time, duration_sec) in enumerate(zip(run_entry_times, run_durations_sec), start=1):
        entry_time_sec = to_epoch_sec(entry_time, tz_name)
        expected_frames = round(duration_sec * detection_rate_hz)

        first_idx = np.searchsorted(frame_times, entry_time_sec)
        if first_idx == len(frame_times) or frame_times[first_idx] - entry_time_sec > max_first_detection_delay_sec:
            print(
                f"WARNING: Run {run_number} has no detection within {max_first_detection_delay_sec} s of entry "
                f"time {entry_time}. Counting all {expected_frames} frames as dropped."
            )
            first_detection_sec = None
            frame_offsets_sec = np.array([])
        else:
            first_detection_sec = frame_times[first_idx]
            # End half a frame early so a last frame jittering around the window edge is counted
            # exactly once (ideal frames land at 0, 1, ..., expected_frames - 1 intervals).
            window_end_sec = first_detection_sec + duration_sec - frame_interval_sec / 2
            window_frames = frame_times[(frame_times >= first_detection_sec) & (frame_times < window_end_sec)]
            frame_offsets_sec = window_frames - first_detection_sec

        received_frames = len(frame_offsets_sec)
        dropped_frames = max(expected_frames - received_frames, 0)
        runs.append({
            "Run": run_number,
            "Entry Time": str(entry_time),
            "First Detection Time": (
                datetime.fromtimestamp(first_detection_sec, tz.gettz(tz_name)).isoformat(sep=" ", timespec="milliseconds")
                if first_detection_sec is not None else ""
            ),
            "First Detection Delay (s)": (
                round(first_detection_sec - entry_time_sec, 3) if first_detection_sec is not None else ""
            ),
            "Duration (s)": duration_sec,
            "Expected Frames": expected_frames,
            "Received Frames": received_frames,
            "Dropped Frames": dropped_frames,
            "Drop (%)": round(dropped_frames / expected_frames * 100, 2),
        })
        run_frame_offsets.append(frame_offsets_sec)

    total_expected = sum(run["Expected Frames"] for run in runs)
    total_received = sum(run["Received Frames"] for run in runs)
    total_dropped = sum(run["Dropped Frames"] for run in runs)
    total_drop_pct = total_dropped / total_expected * 100

    print("\n=== Detection Drop Characterization ===")
    print(f"{'Run':>4} {'Entry Time':>26} {'1st Det. Delay (s)':>19} {'Duration (s)':>13} "
          f"{'Expected':>9} {'Received':>9} {'Dropped':>8} {'Drop (%)':>9}")
    for run in runs:
        print(f"{run['Run']:>4} {run['Entry Time']:>26} {str(run['First Detection Delay (s)']):>19} "
              f"{run['Duration (s)']:>13g} {run['Expected Frames']:>9} {run['Received Frames']:>9} "
              f"{run['Dropped Frames']:>8} {run['Drop (%)']:>9.2f}")
    print(f"Total: {total_dropped}/{total_expected} frames dropped ({total_drop_pct:.2f}%) "
          f"across {len(runs)} runs at {detection_rate_hz:g} Hz")

    fig, (bar_ax, raster_ax) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 2]}
    )

    run_numbers = np.array([run["Run"] for run in runs])
    run_durations = np.array([run["Duration (s)"] for run in runs])
    drop_pcts = np.array([run["Drop (%)"] for run in runs])
    dropped_counts = np.array([run["Dropped Frames"] for run in runs])
    for color_idx, duration_sec in enumerate(sorted(set(run_durations))):
        in_group = run_durations == duration_sec
        bars = bar_ax.bar(
            run_numbers[in_group], drop_pcts[in_group], color=plt.cm.tab10(color_idx % 10),
            label=f"{duration_sec:g} s runs",
        )
        bar_ax.bar_label(bars, labels=[str(n) for n in dropped_counts[in_group]], fontsize=7)
    bar_ax.axhline(total_drop_pct, color="black", linestyle="--", label=f"Overall ({total_drop_pct:.2f}%)")
    bar_ax.set_title(
        f"Detection Drop Characterization: {total_dropped}/{total_expected} frames dropped "
        f"({total_drop_pct:.2f}%) at {detection_rate_hz:g} Hz"
    )
    bar_ax.set_xlabel("Run (bar labels: dropped frames)")
    bar_ax.set_ylabel("Frames Dropped (%)")
    bar_ax.set_xticks(run_numbers)
    bar_ax.tick_params(axis="x", labelsize=7)
    bar_ax.grid(True, axis="y", alpha=0.3)
    bar_ax.legend(fontsize=8)

    for run, frame_offsets_sec in zip(runs, run_frame_offsets):
        y = run["Run"]
        raster_ax.hlines(y, 0, run["Duration (s)"], color="lightgray", linewidth=6)
        raster_ax.plot(frame_offsets_sec, np.full(len(frame_offsets_sec), y), "|", color="green", markersize=6)
        for span_start, span_end in find_dropped_spans(frame_offsets_sec, run["Duration (s)"], frame_interval_sec):
            raster_ax.hlines(y, span_start, span_end, color="red", linewidth=6)
    raster_ax.plot([], [], color="lightgray", linewidth=6, label="Expected window")
    raster_ax.plot([], [], "|", color="green", markersize=8, label="Received frame")
    raster_ax.plot([], [], color="red", linewidth=6, label="Dropped frame(s)")
    raster_ax.set_yticks(run_numbers)
    raster_ax.set_yticklabels([f"{run['Run']} ({run['Duration (s)']:g} s)" for run in runs], fontsize=7)
    raster_ax.invert_yaxis()
    raster_ax.set_xlabel("Seconds Since First Detection")
    raster_ax.set_ylabel("Run (duration)")
    raster_ax.grid(True, axis="x", alpha=0.3)
    raster_ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / PLOT_NAME, dpi=300)
        with open(plots_dir / CSV_NAME, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(runs[0].keys()))
            csv_writer.writeheader()
            csv_writer.writerows(runs)
        print(f"Plot and per-run csv saved to: {plots_dir}")
        plt.close(fig)
    else:
        plt.show()

    return {
        "total_expected_frames": total_expected,
        "total_received_frames": total_received,
        "total_dropped_frames": total_dropped,
        "total_drop_pct": total_drop_pct,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Script to characterize dropped detection frames from a CARMA Streets detected object "
        "Kafka log, over runs where a pedestrian stood in the detection zone for a known duration."
    )
    parser.add_argument(
        "--kafka-log-dir", help="Directory containing Kafka Log files.", type=Path, required=True
    )
    parser.add_argument(
        "--entry-times",
        help='Recorded time the pedestrian entered the detection zone for each run, as epoch seconds or '
        'datetime strings (e.g. "2026-09-03 14:03:21").',
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--durations",
        help="Seconds the pedestrian stayed in the detection zone, one per entry time.",
        nargs="+",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--plots-dir", help="Directory to save generated plot and per-run csv.", type=Path, required=True
    )
    parser.add_argument(
        "--rate-hz", help="Expected detection frame rate.", type=float, default=DEFAULT_DETECTION_RATE_HZ
    )
    parser.add_argument(
        "--timezone", help="Timezone of naive entry time strings.", default=DEFAULT_TIMEZONE
    )
    parser.add_argument(
        "--max-first-detection-delay",
        help="Seconds after an entry time to look for the run's first detection before treating the run as undetected.",
        type=float,
        default=DEFAULT_MAX_FIRST_DETECTION_DELAY_SEC,
    )
    args = parser.parse_args()

    detection_logs = list(args.kafka_log_dir.glob(f"*{KafkaLogMessageType.DetectedObject.value}*.log"))
    if len(detection_logs) != 1:
        print(f"ERROR: Expected one detected object log in {args.kafka_log_dir}, found {detection_logs}")
        return

    characterize_detection_drops(
        detection_logs[0],
        args.entry_times,
        args.durations,
        detection_rate_hz=args.rate_hz,
        max_first_detection_delay_sec=args.max_first_detection_delay,
        tz_name=args.timezone,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
