from pathlib import Path
import argparse
from functools import lru_cache
import json
import re
import sys
import textwrap

import argcomplete
import numpy as np
from matplotlib import pyplot as plt

from run_all_analysis import run_all_analysis
from message_scripts import (
    INCOMING_SDSM_TOPIC,
    check_message_broadcast_rate,
    plot_message_time_intervals,
    run_obu_bsm_transmission_drop_analysis,
)
from parse_ros2_bags import open_bagfile
from utils import parse_kafka_log_timestamps
from carma_cooperative_perception_scripts import (
    DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_IN_S,
    run_sdsm_latency_analysis,
    run_sdsm_detection_drop_rate_analysis,
    run_rsu_sdsm_transmission_drop_rate_analysis,
    run_detection_to_kafka_latency_analysis,
)
from guidance_scripts import get_engage_time

# carma-streets is a hyphenated directory, not an importable package
sys.path.append(str(Path(__file__).resolve().parent.parent / "carma-streets"))
from detection_drop_characterization import characterize_detection_drops
from sdsm_location_spoofing_verification import verify_location_spoofing

# Raw v2xhub object detection Kafka log. A single log is a running record across every MCAP
# recording in a dt_wz (pedestrian detection) test session, so the same path is used for every
# MCAP file - each analysis below windows the log down to that MCAP's own recording span.
DETECTION_LOG_PATH = Path(
    "/workspaces/carma_ws/src/data-verification-initial/dth-flir-camera-laptop-kafka-logs/v2xhub_sim_sensor_detected_object_kafka.log"
)

# CP-03: RSU captures of the session's SDSM broadcasts. Every pcap is checked against every MCAP - each
# MCAP only uses the broadcasts that fall within its own analysis window.
RSU_PCAP_DIR = Path("/workspaces/carma_ws/src/data-verification-initial/rsu-files")

# CP-01: manually recorded pedestrian entry times (one per run, America/New_York) and how long
# the pedestrian stayed in the detection zone for each run (not exit - entry, which includes walking
# in and out). These are the valid runs recorded during the 2026-09-10 data-verification-initial session.
CP01_RUN_ENTRY_TIMES = [
    "2026-09-10 13:43:22", "2026-09-10 13:46:42", "2026-09-10 13:49:51", "2026-09-10 13:53:39", "2026-09-10 13:57:56",
]
CP01_RUN_DURATIONS_SEC = [5, 5, 10, 20, 20]
CP01_OUTPUT_SUBDIR = "cp01_detection_drop_characterization"

# CS-03: remote reference location configured in FLIRCameraDriver (CameraLatitude/CameraLongitude) for the
# 2026-09-10 session. Its heading (CameraRotation, 180 deg) is already applied to the detections by the driver.
# CS-03 only needs one run where the pedestrian is detected: run 5 (entry 13:57:56, 20 s), recorded in this MCAP.
CS03_REF_LAT = 38.955027
CS03_REF_LON = -77.1484523
CS03_MCAP_PATH = Path(
    "/workspaces/carma_ws/src/data-verification-initial/mcap/rosbag2_2026-09-10_135602_0.mcap"
)
CS03_OUTPUT_SUBDIR = "cs03_sdsm_location_spoofing"

# PL-01: Message communication regression testing. Expected average rates of the messages CARMA Platform
# receives/broadcasts, each within +/-20% (+/-2 Hz at 10 Hz, +/-0.2 Hz at 1 Hz). SDSMs are only expected while an
# object is detected, so their rate is averaged over the detection log's detection periods only.
PL01_EXPECTED_RATES_HZ = {
    "/message/incoming_map": ("MAP", "received", 1.0),
    "/message/incoming_spat": ("SPAT", "received", 10.0),
    INCOMING_SDSM_TOPIC: ("SDSM", "received", 10.0),
    "/message/incoming_mobility_operation": ("MOM", "received", 1.0),
    "/message/bsm_outbound": ("BSM", "broadcast", 10.0),
}
PL01_RATE_TOLERANCE_PCT = 0.2
# Detections further apart than this split the detection log into separate detection periods
PL01_DETECTION_GAP_SEC = 0.5
# OBU rmnet_data1 (radio side) captures, checked for the BSMs CARMA Platform sent the OBU to broadcast. Only the
# pcaps overlapping each MCAP's analysis window are used.
PL01_OBU_PCAP_DIR = Path("/workspaces/carma_ws/src/data-verification-initial/obu-pcap")
PL01_OUTPUT_SUBDIR = "pl01_message_communication"
PL01_PLOT_NAME = "pl01_message_intervals_and_obu_bsm_drops.png"

# CP-helper (SDSM message intervals), CP-02 (detection match status), CP-03 (RSU broadcast receipt status) and
# CP-04 (detection to Kafka latency) are drawn as panels of one figure sharing a time axis, so SDSM interval gaps
# line up exactly with dropped detections, dropped RSU transmissions and late detections.
CP_COMBINED_PLOT_NAME = "cp_helper_cp02_cp03_cp04_combined.png"


def _mark_panel_unavailable(ax, metric: str, error: Exception) -> None:
    """Label a combined-figure panel whose analysis couldn't run, rather than leaving an empty axis"""
    # File names are enough to explain the error here; full paths are in the console output
    message = textwrap.fill(re.sub(r"/(?:[^/\s',\]]+/)+", "", str(error)), width=110)
    ax.set_title(f"{metric}: not available")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="dimgray")
    ax.set_yticks([])


def analyze_mcap_file_for_dt_wz_analysis(
    mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path
) -> list:
    """Extract single MCAP file and run all dt_wz (pedestrian detection to SDSM) analysis on it"""
    try:
        engage_time, disengage_time = get_engage_time(mcap_path)
    except Exception as e:
        print(f"Error getting engage time for mcap {mcap_path}: {e}")
        return None

    analysis_stats = {}
    fig, (interval_ax, drop_rate_ax, transmission_ax, kafka_latency_ax) = plt.subplots(
        4, 1, sharex=True, figsize=(12, 15), gridspec_kw={"height_ratios": [5, 3, 3, 3]}
    )

    # CP-helper: SDSM message interval plot, shaded with raw-detection gaps for context
    try:
        plot_message_time_intervals(
            mcap_path=mcap_path,
            topic_name=INCOMING_SDSM_TOPIC,
            detection_log_path=DETECTION_LOG_PATH,
            start_time=engage_time,
            end_time=disengage_time,
            ax=interval_ax,
        )
        analysis_stats["CP_helper_sdsm_message_intervals"] = True
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP_helper_sdsm_message_intervals: {e}")
        analysis_stats["CP_helper_sdsm_message_intervals"] = None
        _mark_panel_unavailable(interval_ax, "CP_helper_sdsm_message_intervals", e)

    # CP-02: Raw detection to SDSM drop rate should be less than 2%
    try:
        is_passed, _, _, _ = run_sdsm_detection_drop_rate_analysis(
            mcap_path=mcap_path,
            detection_log_path=DETECTION_LOG_PATH,
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            save_data_dir=data_dir,
            ax=drop_rate_ax,
        )
        analysis_stats["CP02_sdsm_detection_drop_rate"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP02_sdsm_detection_drop_rate: {e}")
        analysis_stats["CP02_sdsm_detection_drop_rate"] = None
        _mark_panel_unavailable(drop_rate_ax, "CP02_sdsm_detection_drop_rate", e)

    # CP-03: SDSMs broadcast by the RSU but never received by CARMA Platform should be less than 2%
    try:
        is_passed, _, _ = run_rsu_sdsm_transmission_drop_rate_analysis(
            mcap_path=mcap_path,
            rsu_pcap_paths=sorted(RSU_PCAP_DIR.glob("*.pcap")),
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            ax=transmission_ax,
        )
        analysis_stats["CP03_rsu_sdsm_transmission_drop_rate"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP03_rsu_sdsm_transmission_drop_rate: {e}")
        analysis_stats["CP03_rsu_sdsm_transmission_drop_rate"] = None
        _mark_panel_unavailable(transmission_ax, "CP03_rsu_sdsm_transmission_drop_rate", e)

    # CP-04: Detection to Kafka message creation latency should average < 0.5 s with < 2% of detections late (> 0.1 s)
    try:
        is_passed, _, _ = run_detection_to_kafka_latency_analysis(
            mcap_path,
            DETECTION_LOG_PATH,
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            save_data_dir=data_dir,
            ax=kafka_latency_ax,
        )
        analysis_stats["CP04_detection_to_kafka_latency"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP04_detection_to_kafka_latency: {e}")
        analysis_stats["CP04_detection_to_kafka_latency"] = None
        _mark_panel_unavailable(kafka_latency_ax, "CP04_detection_to_kafka_latency", e)

    # DT-05: Median latency from object detection until CARMA Platform receives it in an SDSM should be < 0.3 s
    try:
        is_passed, _, _, _ = run_sdsm_latency_analysis(
            mcap_path,
            error_threshold_to_pass_seconds=DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_IN_S,
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            save_data_dir=data_dir,
            save_plot_dir=plots_dir,
        )
        analysis_stats["DT05_detection_to_sdsm_receipt_latency"] = bool(is_passed)
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric DT05_detection_to_sdsm_receipt_latency: {e}")
        analysis_stats["DT05_detection_to_sdsm_receipt_latency"] = None

    # plot_message_time_intervals is generic across topics, so name what this panel shows here
    if analysis_stats["CP_helper_sdsm_message_intervals"] is not None:
        interval_ax.set_title(
            "CP-helper: SDSM Receive Intervals on CARMA Platform (/message/incoming_sdsm)\n"
            "Shaded green: no FLIR detection in the v2xhub_sim_sensor_detected_object Kafka log",
            pad=20,
        )
        interval_ax.set_ylabel("Time Since Previous SDSM Received (s)")
    # The shared x-axis is labeled once, under the bottom panel
    interval_ax.set_xlabel("")
    drop_rate_ax.set_xlabel("")
    transmission_ax.set_xlabel("")
    kafka_latency_ax.set_xlabel("Time Since Start of Recording (seconds)")
    fig.tight_layout()
    fig.savefig(plots_dir / CP_COMBINED_PLOT_NAME, dpi=300)
    plt.close(fig)
    print(f"Combined CP-helper/CP-02/CP-03/CP-04 plot saved to: {plots_dir / CP_COMBINED_PLOT_NAME}")

    analysis_stats.update(analyze_pl01_message_communication(
        mcap_path, engage_time, disengage_time, stats_dir / PL01_OUTPUT_SUBDIR, plots_dir / PL01_OUTPUT_SUBDIR
    ))

    return [analysis_stats]


@lru_cache(maxsize=None)
def _detection_times_sec(detection_log_path: Path) -> np.ndarray:
    """Sorted epoch-second timestamps of every detection in the (session-wide) detection log"""
    return parse_kafka_log_timestamps(detection_log_path) / 1e3


def detection_intervals(mcap_path: Path, start_time: float, end_time: float) -> list:
    """
    Periods within [start_time, end_time] (seconds since the start of the MCAP recording) during which
    DETECTION_LOG_PATH has detections, split wherever consecutive detections are more than
    PL01_DETECTION_GAP_SEC apart. Periods with a single detection have no duration and are left out.
    """
    _, _, global_start_time_ns = open_bagfile(str(mcap_path))
    times = _detection_times_sec(DETECTION_LOG_PATH) - global_start_time_ns / 1e9
    times = np.unique(times[(times >= start_time) & (times <= end_time)])
    if len(times) < 2:
        return []
    breaks = np.flatnonzero(np.diff(times) > PL01_DETECTION_GAP_SEC)
    starts = np.r_[times[0], times[breaks + 1]]
    ends = np.r_[times[breaks], times[-1]]
    return [(float(start), float(end)) for start, end in zip(starts, ends) if end > start]


def analyze_pl01_message_communication(
    mcap_path: Path, engage_time: float, disengage_time: float, stats_dir: Path, plots_dir: Path
) -> dict:
    """
    PL-01: Message communication regression testing over the engaged window of one MCAP.
    - Freq check (ROS side, covers the OBU side too): each PL01_EXPECTED_RATES_HZ topic's average rate should be
      within PL01_RATE_TOLERANCE_PCT of its expected rate
    - Msg drop check (ROS side): message interval plots per topic, the SDSM one shaded with detection log gaps
    - Msg drop check (OBU side): BSMs CARMA Platform sent that the OBU never transmitted (characterization only)
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_stats = {}

    for topic, (label, direction, expected_rate_hz) in PL01_EXPECTED_RATES_HZ.items():
        metric = f"PL01_{label}_{direction}_rate"
        try:
            is_passed, rate_stats, rate_fig, _, _ = check_message_broadcast_rate(
                mcap_path,
                topic,
                expected_rate_hz,
                rate_tolerance_pct=PL01_RATE_TOLERANCE_PCT,
                start_time=engage_time,
                end_time=disengage_time,
                save_stats_dir=stats_dir,
                save_plot_dir=plots_dir,
                pass_on_average_rate=True,
                active_intervals=detection_intervals(mcap_path, engage_time, disengage_time) if label == "SDSM" else None,
            )
            if rate_fig is not None:
                plt.close(rate_fig)
            analysis_stats[metric] = is_passed if rate_stats else None
        except Exception as e:
            print(f"Error analyzing {mcap_path} for metric {metric}: {e}")
            analysis_stats[metric] = None

    fig, axes = plt.subplots(
        len(PL01_EXPECTED_RATES_HZ) + 1, 1, sharex=True, figsize=(12, 22),
        gridspec_kw={"height_ratios": [4] * len(PL01_EXPECTED_RATES_HZ) + [2]},
    )
    for ax, (topic, (label, direction, expected_rate_hz)) in zip(axes, PL01_EXPECTED_RATES_HZ.items()):
        expected_interval_sec = 1.0 / expected_rate_hz
        try:
            plot_message_time_intervals(
                mcap_path=mcap_path,
                topic_name=topic,
                expected_interval_sec=expected_interval_sec,
                interval_tolerance_pct=PL01_RATE_TOLERANCE_PCT,
                detection_log_path=DETECTION_LOG_PATH if label == "SDSM" else None,
                start_time=engage_time,
                end_time=disengage_time,
                ax=ax,
                max_view_sec=5 * expected_interval_sec if expected_rate_hz >= 10 else 2 * expected_interval_sec,
            )
        except Exception as e:
            print(f"Error plotting {topic} message intervals for {mcap_path}: {e}")
        ax.set_title(
            f"PL-01: {label} {direction.capitalize()} Intervals on CARMA Platform ({topic}), "
            f"expected {expected_rate_hz:g} Hz" + ("\nShaded green: no FLIR detection in the detection Kafka log" if label == "SDSM" else ""),
            pad=20,
        )
        ax.set_ylabel(f"Time Since Previous {label} (s)")
        ax.set_xlabel("")

    metric = "PL01_obu_bsm_transmission_drop"
    try:
        run_obu_bsm_transmission_drop_analysis(
            mcap_path,
            sorted(PL01_OBU_PCAP_DIR.glob("*rmnet*.pcap")),
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            ax=axes[-1],
        )
        # Characterization only (no pass/fail threshold): True means the drop rate was measured
        analysis_stats[metric] = True
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric {metric}: {e}")
        analysis_stats[metric] = None

    axes[-1].set_xlabel("Time Since Start of Recording (seconds)")
    fig.tight_layout()
    fig.savefig(plots_dir / PL01_PLOT_NAME, dpi=200)
    plt.close(fig)
    print(f"PL-01 message interval and OBU BSM drop plot saved to: {plots_dir / PL01_PLOT_NAME}")

    return analysis_stats


def run_session_analysis(output_dir: Path) -> None:
    """
    Run the metrics evaluated once per test session rather than per MCAP, and add their results to
    the session's analysis_summary.json:
    - CP-01: Detection drop characterization over the recorded runs, which span several MCAP files
      (characterization only)
    - CS-03: SDSMs received in one run should place the spoofed pedestrian at the reference
      (mean position error < 0.2 m, mean heading error < 1 deg)
    - CP-03 total: the per-MCAP RSU SDSM transmission drop counts pooled into one session drop rate
    - PL-01 OBU total: the per-MCAP OBU BSM transmission drop counts pooled into one session drop rate
    """
    session_metrics = {}

    try:
        cp01_result = characterize_detection_drops(
            DETECTION_LOG_PATH,
            CP01_RUN_ENTRY_TIMES,
            CP01_RUN_DURATIONS_SEC,
            plots_dir=output_dir / CP01_OUTPUT_SUBDIR,
        )
        session_metrics["CP01_detection_drop_characterization"] = {
            "output_dir": str(output_dir / CP01_OUTPUT_SUBDIR),
            "runs": len(cp01_result["runs"]),
            "expected_frames": cp01_result["total_expected_frames"],
            "received_frames": cp01_result["total_received_frames"],
            "dropped_frames": cp01_result["total_dropped_frames"],
            "drop_rate": f"{cp01_result['total_drop_pct']:.2f}%",
        }
    except Exception as e:
        print(f"Error analyzing metric CP01_detection_drop_characterization: {e}")
        session_metrics["CP01_detection_drop_characterization"] = None

    try:
        cs03_result = verify_location_spoofing(
            DETECTION_LOG_PATH,
            CS03_REF_LAT,
            CS03_REF_LON,
            mcap_path=CS03_MCAP_PATH,
            plots_dir=output_dir / CS03_OUTPUT_SUBDIR,
        )
        mcap_result = cs03_result["sources"]["mcap"]
        session_metrics["CS03_sdsm_location_spoofing"] = {
            "output_dir": str(output_dir / CS03_OUTPUT_SUBDIR),
            "mcap": str(CS03_MCAP_PATH),
            "passed": cs03_result["pass"],
            "verified_objects": mcap_result["verified_objects"],
            "mean_position_error_m": round(mcap_result["mean_position_error_m"], 4),
            "mean_heading_error_deg": round(mcap_result["mean_heading_error_deg"], 4),
        }
    except Exception as e:
        print(f"Error analyzing metric CS03_sdsm_location_spoofing: {e}")
        session_metrics["CS03_sdsm_location_spoofing"] = None

    # CP-03 pooled over every MCAP's RSU SDSM broadcasts, for the session's overall transmission drop rate
    cp03_stats = [
        json.loads(stats_path.read_text())
        for stats_path in sorted(output_dir.glob("*/stats/rsu_sdsm_transmission_drop_rate.json"))
    ]
    if cp03_stats:
        checked = sum(stats["total_broadcasts_checked"] for stats in cp03_stats)
        dropped = sum(stats["total_dropped"] for stats in cp03_stats)
        session_metrics["CP03_rsu_sdsm_transmission_drop_rate"] = {
            "mcaps": len(cp03_stats),
            "broadcasts_checked": checked,
            "received": sum(stats["total_received"] for stats in cp03_stats),
            "received_late": sum(stats["total_received_late"] for stats in cp03_stats),
            "dropped": dropped,
            "drop_rate": f"{dropped / checked * 100:.2f}%",
        }
        print(f"CP-03 session drop rate: {dropped}/{checked} RSU SDSM broadcasts not received "
              f"({dropped / checked * 100:.2f}%) across {len(cp03_stats)} MCAPs")

    # PL-01 OBU side pooled over every MCAP's sent BSMs, for the session's overall BSM transmission drop rate
    pl01_stats = [
        json.loads(stats_path.read_text())
        for stats_path in sorted(output_dir.glob(f"*/stats/{PL01_OUTPUT_SUBDIR}/obu_bsm_transmission_drop_rate.json"))
    ]
    if pl01_stats:
        checked = sum(stats["total_bsms_checked"] for stats in pl01_stats)
        dropped = sum(stats["total_dropped"] for stats in pl01_stats)
        session_metrics["PL01_obu_bsm_transmission_drop"] = {
            "mcaps": len(pl01_stats),
            "bsms_checked": checked,
            "transmitted": sum(stats["total_transmitted"] for stats in pl01_stats),
            "transmitted_late": sum(stats["total_transmitted_late"] for stats in pl01_stats),
            "dropped": dropped,
            "drop_rate": f"{dropped / checked * 100:.2f}%",
        }
        print(f"PL-01 session OBU BSM drop rate: {dropped}/{checked} sent BSMs not transmitted "
              f"({dropped / checked * 100:.2f}%) across {len(pl01_stats)} MCAPs")

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["session_metrics"] = session_metrics
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Session metrics (CP-01, CS-03, CP-03 total, PL-01 OBU total) added to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all DT-WZ (pedestrian detection to SDSM) analysis on multiple MCAP files in a given directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MCAP files to analyze",
        default=Path("/workspaces/carma_ws/src/data-verification-initial"),
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
        output_dir = run_all_analysis(
            args.input_dir,
            analyze_mcap_file_for_dt_wz_analysis,
            args.output_dir,
            analysis_name="dt_wz_analysis",
        )
        run_session_analysis(output_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
