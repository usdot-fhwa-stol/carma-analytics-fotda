from pathlib import Path
import argparse
import json
import sys

import argcomplete
from matplotlib import pyplot as plt

from run_all_analysis import run_all_analysis
from message_scripts import plot_message_time_intervals, INCOMING_SDSM_TOPIC
from carma_cooperative_perception_scripts import (
    DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_IN_S,
    run_sdsm_latency_analysis,
    run_sdsm_detection_drop_rate_analysis,
    run_rsu_sdsm_transmission_drop_rate_analysis,
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

# CP-helper (SDSM message intervals), CP-02 (detection match status) and CP-03 (RSU broadcast receipt status)
# are drawn as panels of one figure sharing a time axis, so SDSM interval gaps line up exactly with dropped
# detections and dropped RSU transmissions.
CP_HELPER_CP02_CP03_PLOT_NAME = "cp_helper_cp02_cp03_combined.png"


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
    fig, (interval_ax, drop_rate_ax, transmission_ax) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 12), gridspec_kw={"height_ratios": [5, 3, 3]}
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

    # DT-05: Median latency from object detection until CARMA Platform receives it in an SDSM should be < 0.2 s
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
    interval_ax.set_title(
        "CP-helper: SDSM Receive Intervals on CARMA Platform (/message/incoming_sdsm)\n"
        "Shaded green: no FLIR detection in the v2xhub_sim_sensor_detected_object Kafka log",
        pad=20,
    )
    interval_ax.set_ylabel("Time Since Previous SDSM Received (s)")
    # The shared x-axis is labeled once, under the bottom panel
    interval_ax.set_xlabel("")
    drop_rate_ax.set_xlabel("")
    transmission_ax.set_xlabel("Time Since Start of Recording (seconds)")
    fig.tight_layout()
    fig.savefig(plots_dir / CP_HELPER_CP02_CP03_PLOT_NAME, dpi=300)
    plt.close(fig)
    print(f"Combined CP-helper/CP-02/CP-03 plot saved to: {plots_dir / CP_HELPER_CP02_CP03_PLOT_NAME}")

    return [analysis_stats]


def run_session_analysis(output_dir: Path) -> None:
    """
    Run the metrics evaluated once per test session rather than per MCAP, and add their results to
    the session's analysis_summary.json:
    - CP-01: Detection drop characterization over the recorded runs, which span several MCAP files
      (characterization only)
    - CS-03: SDSMs received in one run should place the spoofed pedestrian at the reference
      (mean position error < 0.2 m, mean heading error < 1 deg)
    - CP-03 total: the per-MCAP RSU SDSM transmission drop counts pooled into one session drop rate
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

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["session_metrics"] = session_metrics
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Session metrics (CP-01, CS-03, CP-03 total) added to: {summary_path}")


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
