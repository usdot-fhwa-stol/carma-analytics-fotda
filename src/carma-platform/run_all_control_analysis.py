from pathlib import Path
from typing import Dict
from guidance_scripts import (
    get_engage_time,
    run_crosstrack_analysis,
    run_turn_accuracy_analysis,
    run_acceleration_comfort_analysis,
)
from run_all_analysis import run_all_analysis
import argparse
import argcomplete

# VARIOUS THRESHOLDS FOR THE METRICS
# 1. Cross_track analysis
CROSS_TRACK_ERROR_THRESHOLD_METER = 2.0
# 2. Turn accuracy analysis
TURN_ACCURACY_ERROR_THRESHOLD_METER = 2.0
# 3. Acceleration comfort analysis
COMFORT_ACCELERATION_THRESHOLD_MS2 = 3.0


def analyze_mcap_file_for_control_analysis(
    mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path
) -> Dict:
    """Extract single MCAP file and run all control analysis on it"""
    # 0. General steps needed for all
    try:
        engage_time, disengage_time = get_engage_time(mcap_path)
    except Exception as e:
        print(f"Error analyzing {mcap_path}: {e}")
        return None

    analysis_stats = {}
    # 1. Cross_track analysis
    try:
        is_passed, _, _, _, _ = run_crosstrack_analysis(
            mcap_path,
            CROSS_TRACK_ERROR_THRESHOLD_METER,
            engage_time,
            disengage_time,
            stats_dir,
            data_dir,
            plots_dir,
        )
        analysis_stats["run_crosstrack_analysis"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric run_crosstrack_analysis: {e}")
        analysis_stats["run_crosstrack_analysis"] = None

    # 2. Turn accuracy analysis by spline fitting
    try:
        is_passed, _, _, _, _ = run_turn_accuracy_analysis(
            mcap_path,
            TURN_ACCURACY_ERROR_THRESHOLD_METER,
            engage_time,
            disengage_time,
            stats_dir,
            data_dir,
            plots_dir,
        )
        analysis_stats["run_turn_accuracy_analysis"] = is_passed
    except Exception as e:
        print(f"Error analyzing {mcap_path} for metric run_turn_accuracy_analysis: {e}")
        analysis_stats["run_turn_accuracy_analysis"] = None

    # 3.
    try:
        is_passed, _, _, _, _, _, _, _ = run_acceleration_comfort_analysis(
            mcap_path,
            COMFORT_ACCELERATION_THRESHOLD_MS2,
            engage_time,
            disengage_time,
            stats_dir,
            data_dir,
            plots_dir,
        )
        analysis_stats["run_acceleration_comfort_analysis"] = is_passed
    except Exception as e:
        print(
            f"Error analyzing {mcap_path} for metric run_acceleration_comfort_analysis: {e}"
        )
        analysis_stats["run_acceleration_comfort_analysis"] = None

    return analysis_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all control analysis on multiple MCAP files in a given directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MCAP files to analyze",
        required=True,
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
            args.input_dir, analyze_mcap_file_for_control_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
