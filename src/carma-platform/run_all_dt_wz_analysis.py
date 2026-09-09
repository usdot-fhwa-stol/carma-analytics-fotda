from pathlib import Path
import argparse
import argcomplete

from run_all_analysis import run_all_analysis
from message_scripts import plot_message_time_intervals, INCOMING_SDSM_TOPIC
from carma_cooperative_perception_scripts import run_sdsm_detection_drop_rate_analysis
from guidance_scripts import get_engage_time

# Raw v2xhub object detection Kafka log. A single log is a running record across every MCAP
# recording in a dt_wz (pedestrian detection) test session, so the same path is used for every
# MCAP file - each analysis below windows the log down to that MCAP's own recording span.
DETECTION_LOG_PATH = Path(
    "/workspaces/carma_ws/src/data/kafka-logs/v2xhub_sim_sensor_detected_object_kafka.log"
)


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

    # CP-01: SDSM message interval plot, shaded with raw-detection gaps for context
    try:
        plot_message_time_intervals(
            mcap_path=mcap_path,
            topic_name=INCOMING_SDSM_TOPIC,
            detection_log_path=DETECTION_LOG_PATH,
            start_time=engage_time,
            end_time=disengage_time,
            save_plot_dir=plots_dir,
        )
        analysis_stats["CP01_sdsm_message_intervals"] = True
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP01_sdsm_message_intervals: {e}")
        analysis_stats["CP01_sdsm_message_intervals"] = None

    # CP-02: Raw detection to SDSM drop rate should be less than 2%
    try:
        is_passed, _, _, _ = run_sdsm_detection_drop_rate_analysis(
            mcap_path=mcap_path,
            detection_log_path=DETECTION_LOG_PATH,
            start_time=engage_time,
            end_time=disengage_time,
            save_stats_dir=stats_dir,
            save_data_dir=data_dir,
            save_plot_dir=plots_dir,
        )
        analysis_stats["CP02_sdsm_detection_drop_rate"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric CP02_sdsm_detection_drop_rate: {e}")
        analysis_stats["CP02_sdsm_detection_drop_rate"] = None

    return [analysis_stats]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all DT-WZ (pedestrian detection to SDSM) analysis on multiple MCAP files in a given directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MCAP files to analyze",
        default=Path("/workspaces/carma_ws/src/data"),
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
            args.input_dir,
            analyze_mcap_file_for_dt_wz_analysis,
            args.output_dir,
            analysis_name="dt_wz_analysis",
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
