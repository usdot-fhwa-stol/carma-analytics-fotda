from pathlib import Path
from typing import Dict
from guidance_scripts import (
    get_engage_time,
    run_crosstrack_analysis,
    run_turn_accuracy_analysis,
    run_acceleration_comfort_analysis,
    run_lateral_analysis,
    run_guidance_steering_analysis,
    run_steering_wheel_analysis,
    get_planner_trajectory_intervals,
    run_guidance_speed_analysis,
    run_turn_speed_analysis,
    run_turn_acceleration_analysis,
)
from run_all_analysis import run_all_analysis
import argparse
import argcomplete
from environment_scripts import (
    extract_lanelet2_map_from_mcap,
    filter_map_points_for_trajectory,
    plot_2d_map_and_pose,
)

# VARIOUS THRESHOLDS FOR THE METRICS
# 1. Cross_track analysis
CROSS_TRACK_ERROR_THRESHOLD_METER = 2.0
# 2. Turn accuracy analysis
TURN_ACCURACY_ERROR_THRESHOLD_METER = 2.0
# 3. Acceleration comfort analysis
COMFORT_ACCELERATION_THRESHOLD_MS2 = 3.0
# 4. Lateral acceleration jerk analysis
ACC_THRESHOLD_TO_PASS_MS2 = 2.0
JERK_THRESHOLD_TO_PASS_MS3 = 3.0
# 5. Steering angle and steering wheel analysis
STEERING_ANGLE_ERROR_THREHOLD_RADIAN = 0.1
STEERING_WHEEL_ANGLE_ERROR_THRESHOLD_RADIAN = 0.1
# 6. Speed analysis
SPEED_THRESHOLD_TO_PASS_MPH = 2.0
# 7. Turn Threshold for Steering Angle
TURN_THRESHOLD = 0.2 #rad
# 8. Wheelbase
WHEELBASE = 2.75
# 9. Lateral Acceleration Limit
LATERAL_ACCELERATION_LIMIT = 2.5
# 10. Excess Speed Threshold on Turns
EXCESS_TURN_SPEED_THRESHOLD = 0.1



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
        try:
            guidance_speed_analysis_results = run_guidance_speed_analysis(
                mcap_path,
                SPEED_THRESHOLD_TO_PASS_MPH,
                start_time,
                end_time,
                stats_dir,
                data_dir,
                plots_dir,
            )
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_guidance_speed_analysis: {e}"
            )
            analysis_stats["run_guidance_speed_analysis"] = None
        try:
            turn_speed_analysis_results = run_turn_speed_analysis(
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
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_turn_speed_analysis: {e}"
            )
            analysis_stats["run_turn_speed_analysis"] = None

        try:
            turn_acc_analysis_results = run_turn_acceleration_analysis(
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
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for metric run_turn_speed_analysis: {e}"
            )
            analysis_stats["run_turn_speed_analysis"] = None

        all_analysis_stats.append(analysis_stats)

    return all_analysis_stats


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
            args.input_dir, analyze_mcap_file_for_regression_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
