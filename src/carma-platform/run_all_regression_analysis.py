from pathlib import Path
from typing import Dict
from guidance_scripts import (
    get_engage_time,
    run_crosstrack_analysis,
    run_acceleration_comfort_analysis,
    run_guidance_speed_analysis,
    run_turn_speed_analysis,
    run_turn_acceleration_analysis,
    run_speed_limit_change_response_analysis
)
from run_all_analysis import run_all_analysis
import argparse
import argcomplete


# VARIOUS THRESHOLDS FOR THE METRICS
# 1. Cross_track analysis
CROSS_TRACK_ERROR_THRESHOLD_METER = 0.2
CROSS_TRACK_ERROR_THRESHOLD_PERCENTILE = None
# 2. Acceleration comfort analysis
COMFORT_ACCELERATION_THRESHOLD_MS2 = 3.0
# 3. Guidance Speed analysis
SPEED_LIMIT_ERROR_THRESHOLD_TO_PASS_MPH = 2.0
# 4. Turn Speed and acceleration analysis
TURN_THRESHOLD = 0.2 #rad
WHEELBASE = 2.75 # meters
LATERAL_ACCELERATION_LIMIT = 2.5 # m/s^2
EXCESS_TURN_SPEED_THRESHOLD = 0.1 # m/s
# 6. Speed limit change response analysis
RESPONSE_TIME_THRESHOLD = 0.2 # seconds
STEADY_STATE_DURATION = 3.0 # seconds
SPEED_TOLERANCE_PCT = 0.2 # 20% tolerance


def analyze_mcap_file_for_regression_analysis(
    mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path
) -> list:
    """Extract single MCAP file and run all control analysis on it"""
    # 0. General steps needed for all
    try:
        engage_time, disengage_time = get_engage_time(mcap_path)
    except Exception as e:
        print(f"Error getting engage time for mcap {mcap_path}: {e}")
        return None

    # Run all the tests for engage to disengage
    # and specifically for lane change durations:
    intervals = [(engage_time, disengage_time)]

    all_analysis_stats = []
    for start_time, end_time in intervals:
        analysis_stats = {}

        # 1. Cross_track analysis
        try:
            is_passed, _, _, _, _ = run_crosstrack_analysis(
                mcap_path,
                CROSS_TRACK_ERROR_THRESHOLD_METER,
                CROSS_TRACK_ERROR_THRESHOLD_PERCENTILE,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_crosstrack_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_crosstrack_analysis: {e}"
            )
            analysis_stats["run_crosstrack_analysis"] = None


        # 2. Longitudinal acceleration analysis
        try:
            is_passed, _, _, _, _, _, _, _ = run_acceleration_comfort_analysis(
                mcap_path,
                COMFORT_ACCELERATION_THRESHOLD_MS2,
                start_time,
                end_time,
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

        # 3. Guidance speed analysis
        try:
            is_passed, _, _, _, _  = run_guidance_speed_analysis(
                mcap_path,
                SPEED_LIMIT_ERROR_THRESHOLD_TO_PASS_MPH,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_guidance_speed_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_guidance_speed_analysis: {e}"
            )
            analysis_stats["run_guidance_speed_analysis"] = None

        # 4. Turn speed analysis
        try:
            is_passed, _, _, _, _ = run_turn_speed_analysis(
                mcap_path,
                TURN_THRESHOLD,
                WHEELBASE,
                LATERAL_ACCELERATION_LIMIT,
                EXCESS_TURN_SPEED_THRESHOLD,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_turn_speed_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_turn_speed_analysis: {e}"
            )
            analysis_stats["run_turn_speed_analysis"] = None

        # 5. Turn acceleration analysis
        try:
            is_passed, _, _, _, _ = run_turn_acceleration_analysis(
                mcap_path,
                LATERAL_ACCELERATION_LIMIT,
                TURN_THRESHOLD,
                WHEELBASE,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_turn_acceleration_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_turn_acceleration_analysis: {e}"
            )
            analysis_stats["run_turn_acceleration_analysis"] = None

        # 6. Speed limit change response analysis
        try:
            is_passed, _, _, _, _ = run_speed_limit_change_response_analysis(
                mcap_path,
                RESPONSE_TIME_THRESHOLD,
                STEADY_STATE_DURATION,
                SPEED_TOLERANCE_PCT,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
            analysis_stats["run_speed_limit_change_response_analysis"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_speed_limit_change_response_analysis: {e}"
            )
            analysis_stats["run_speed_limit_change_response_analysis"] = None

        all_analysis_stats.append(analysis_stats)

    return all_analysis_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all regression analysis on multiple MCAP files in a given directory"
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
            args.input_dir, analyze_mcap_file_for_regression_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
