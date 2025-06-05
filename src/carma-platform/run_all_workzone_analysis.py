import argparse
import argcomplete
from pathlib import Path
from run_all_analysis import run_all_analysis
from message_scripts import (
    process_cc_logs_for_tcr_tcm_data,
    check_cc_response_delay,
    check_tcm_acknowledgement_delay,
    check_tcm_broadcast_count,
    check_tcm_broadcast_rate,
    check_tcm_response_time
)
from guidance_scripts import (
    get_engage_time,
    get_geofence_entrance_and_exit_times,
    get_route_original_speed,
    check_speed_limits_in_geofence,
    check_geofence_in_reroute,
    check_reroute_duration,
    check_lanechange_lateral_velocity,
    check_lanechange_duration,
    check_time_to_begin_deceleration,
    run_speed_limit_change_response_analysis,
    check_speed_before_workzone,
    create_geofence_acceleration_plot,
    check_deceleration_for_geofence,
    run_acceleration_comfort_analysis,
    check_time_to_begin_acceleration,
    check_acceleration_after_geofence,
)


############################################################################
# Constant Values - adjust these as needed for your metrics/environment
############################################################################
CC_LOG_SOURCE_FOLDER = "/workspaces/carma-analytics-fotda/.devcontainer/cc_log"
DATE_CC_LOGS_TAKEN = "2025-05-20" # Assumes all logs were taken the same day
CLOSED_LANELETS = [] # List of closed lanelets used in this test
CC_TCR_TO_TCM_SEC = 0.1
CC_MAX_BROADCAST_COUNT = 10
CC_MAX_BROADCAST_RATE_HZ = 10

FREIGHT_STEADY_STATE_TIME_SEC = 5
FREIGHT_MAINTAIN_SPEED_RANGE_MS = 0.89408 # 0.89408 m/s is 2 mph
FREIGHT_TCM_ACKNOWLEDGEMENT_DELAY_SEC = 1
FREIGHT_UPDATE_ACTIVE_ROUTE_SEC = 3
FREIGHT_MIN_LAT_VELOCITY_MS = 0.5
FREIGHT_MAX_LAT_VELOCITY_MS = 1.25
FREIGHT_ADVISORY_SPEED_RESPONSE_SEC = 1.3
FREIGHT_MAX_DECELERATION_MS2 = -2
FREIGHT_MIN_AVERAGE_ACCELERATION_MS2 = 1
FREIGHT_MAX_SECTION_ACCELERATION_MS2 = 2
FREIGHT_TCR_SEND_TO_TCM_RECEIVE_DELAY_SEC = 1
FREIGHT_ADVISORY_SPEED_LIMIT_MS = 4.5


def analyze_mcap_file_for_workzone_analysis(mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path) -> list:
    """Extract single MCAP file and run all workzone analysis on it"""
    # 0. General Steps needed for all

    # Get the engage/disengage times
    print("\nGetting engage/disengage times")
    try:
        engage_time, disengage_time = get_engage_time(mcap_path)
    except Exception as e:
        print(f"Error getting engage time for mcap {mcap_path}: {e}\n")
        return None

    # Get the geofence entrance and exit times
    print("\nGetting the times the vehicle entered/exited the geofence")
    try:
        time_enter_geofence, time_exit_geofence, found_geofence_times = get_geofence_entrance_and_exit_times(mcap_path)
        print(f"Enter time: {time_enter_geofence} Exit time: {time_exit_geofence} Found: {found_geofence_times}\n")
    except Exception as e:
        print(
            f"Error analyzing {mcap_path} for getting geofence times: {e}\n"
        )
        return None

    # Get the original speed limit
    print("\nGetting the original route speed limit")
    original_speed_limit_ms = get_route_original_speed(mcap_path, engage_time)
    print(f"Original Route Speed Limit: {original_speed_limit_ms} m/s\n")

    print("\nProcessing Carma Cloud Logs")
    # Process the CC Tomcat Logs
    all_cc_data, _ = process_cc_logs_for_tcr_tcm_data(CC_LOG_SOURCE_FOLDER, DATE_CC_LOGS_TAKEN, CC_TCR_TO_TCM_SEC, CC_MAX_BROADCAST_RATE_HZ, stats_dir, data_dir, plots_dir)

    print("\nRunning Speed Limit Analysis")
    # Run the speed limit analysis for the entire period of time
    _, _, _, speed_limit_changes, response_times = run_speed_limit_change_response_analysis(mcap_path, steady_state_indication_time=FREIGHT_ADVISORY_SPEED_RESPONSE_SEC, save_stats_dir=stats_dir, save_data_dir=data_dir, save_plot_dir=plots_dir)

    print("\nRunning Acceleration Analysis")
    # Run the acceleration analysis for the entire period of time
    _, _, _, _, accelerations, avg_accelerations, timepoints, avg_timepoints = run_acceleration_comfort_analysis(mcap_path, FREIGHT_MAX_DECELERATION_MS2, save_stats_dir=stats_dir, save_data_dir=data_dir, save_plot_dir=plots_dir)
    accelerations_times = []
    for (timepoint, acceleration) in zip(timepoints, accelerations):
        accelerations_times.append((timepoint, acceleration))
    avg_accelerations_times = []
    for (timepoint, avg_acceleration) in zip(avg_timepoints, avg_accelerations):
        avg_accelerations_times.append((timepoint, avg_acceleration))

    print("\nCreating Geofence Plots")
    create_geofence_acceleration_plot(accelerations_times, avg_accelerations_times, time_enter_geofence, time_exit_geofence, plots_dir)

    # Run all the tests from engage to disengage
    intervals = [(engage_time, disengage_time)]
    all_analysis_stats = []

    print(f"-----------------------------------------------------")
    print(f"-------------Beginning workzone analysis-------------")
    print(f"-----------------------------------------------------")
    for start_time, end_time in intervals:
        analysis_stats = {}

        ##########################################################################################################
        # FWZ-1: The geofenced area is a part of the initial route plan.
        #
        # FWZ-8: The vehicle receives a message from CC that includes the closed lane ahead. The Vehicle
        #           procceses this closed lane information.
        ##########################################################################################################
        print(f"Starting analysis for FWZ-1, FWZ-8")
        try:
            fwz1_is_passed, fwz8_is_passed = check_geofence_in_reroute(mcap_path, CLOSED_LANELETS, data_dir)
            analysis_stats["check_geofence_in_original_route"] = fwz1_is_passed
            analysis_stats["check_geofence_in_reroute"] = fwz8_is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for processing the closed lane message from CC: {e}"
            )
            analysis_stats["check_geofence_in_original_route"] = None
            analysis_stats["check_geofence_in_reroute"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-4: Once CC receives a TCR from the vehicle, it sends a TCM to the vehicle in 0.1 seconds
        ##########################################################################################################
        print(f"Starting analysis for FWZ-4")
        try:
            is_passed = check_cc_response_delay(all_cc_data, CC_TCR_TO_TCM_SEC, stats_dir, data_dir)
            analysis_stats["check_cc_response_delay"] = is_passed
        except Exception as e:
            print(
                f"Error analying CC Logs for delay in sending TCM to vehicle after receiving TCR: {e}"
            )
            analysis_stats["check_cc_response_delay"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-7: The vehicle receives a message from CC that includes the new speed limit for the geofence area.
        #        The vehicle successfully processes this new speed limit.
        ##########################################################################################################
        print(f"Starting analysis for FWZ-7")
        try:
            is_passed = check_speed_limits_in_geofence(mcap_path, time_enter_geofence, time_exit_geofence, FREIGHT_ADVISORY_SPEED_LIMIT_MS, data_dir)
            analysis_stats["check_in_geofence_speed_limits"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for receiving new speed limit in geofence from CC: {e}"
            )
            analysis_stats["check_in_geofence_speed_limits"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-9: The vehicle returns an Acknowledgement message within 1 second after successfully processing
        #           a TCM from CC
        ##########################################################################################################
        print(f"Starting analysis for FWZ-9")
        try:
            is_passed, tcm_acknowledgements, _, _ = check_tcm_acknowledgement_delay(mcap_path, FREIGHT_TCM_ACKNOWLEDGEMENT_DELAY_SEC, stats_dir, data_dir, plots_dir)
            analysis_stats["check_tcm_acknowledgement"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for sending acknowledgement after TCM: {e}"
            )
            analysis_stats["check_tcm_acknowledgement"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-11: After the vehicle has received a message from CC with the Work Zone information, the vehicle
        #           shall successfully update its active route in less than 3 seconds to avoid to Work Zone.
        ##########################################################################################################
        print(f"Starting analysis for FWZ-11")
        try:
            is_passed = check_reroute_duration(mcap_path, FREIGHT_UPDATE_ACTIVE_ROUTE_SEC, data_dir)
            analysis_stats["check_reroute_duration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking reroute duration: {e}"
            )
            analysis_stats["check_reroute_duration"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-12: The vehicle changes lanes to follow the lane restrictions defined in the TCM while
        #           in the work zone. Handled observationally during live testing
        ##########################################################################################################
        print(f"-----------------------------------------------------")
        print(f"Analysis for FWZ-12 is handled observationally")
        print(f"-----------------------------------------------------\n")
        ##########################################################################################################
        # FWZ-13: The vehicle lateral velocity during a lane change remains between 0.5 m/s and 1.25 m/s
        ##########################################################################################################
        print(f"Starting analysis for FWZ-13")
        try:
            is_passed, _, _ = check_lanechange_lateral_velocity(mcap_path, FREIGHT_MIN_LAT_VELOCITY_MS, FREIGHT_MAX_LAT_VELOCITY_MS, stats_dir, data_dir, plots_dir)
            analysis_stats["check_lanechange_lateral_velocity"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking lane change velocity: {e}"
            )
            analysis_stats["check_lanechange_lateral_velocity"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-14: After changing lanes, the vehicle will achieve steady-state
        #           (i.e. truck is driving within the lane) within 5 seconds
        ##########################################################################################################
        print(f"Starting analysis for FWZ-14")
        try:
            is_passed, _ = check_lanechange_duration(mcap_path, start_time, FREIGHT_STEADY_STATE_TIME_SEC, stats_dir, data_dir)
            analysis_stats["check_lanechange_duration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking lane change duration: {e}"
            )
            analysis_stats["check_lanechange_duration"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-18: CC will broadcast the TCM up to 10 times or until receipt of acknowledgement
        #           message from the CMV
        ##########################################################################################################
        print(f"Starting analysis for FWZ-18")
        try:
            is_passed = check_tcm_broadcast_count(all_cc_data, tcm_acknowledgements, CC_MAX_BROADCAST_COUNT, stats_dir, data_dir)
            analysis_stats["check_tcm_broadcast_count"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing Carma Cloud and TCM Acknowledgement data for TCM broadcast count: {e}"
            )
            analysis_stats["check_tcm_broadcast_count"] = None
        print(f"-----------------------------------------------------\n")
        ##########################################################################################################
        # FWZ-19: CC will broadcast the TCM at 10 Hz until receipt of acknowledgement message from the CMV
        ##########################################################################################################
        print(f"Starting analysis for FWZ-19")
        try:
            is_passed = check_tcm_broadcast_rate(all_cc_data, tcm_acknowledgements, CC_MAX_BROADCAST_RATE_HZ, stats_dir, data_dir)
            analysis_stats["check_tcm_broadcast_rate"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing Carma Cloud data for TCM Broadcast rate: {e}"
            )
            analysis_stats["check_tcm_broadcast_rate"] = None
        print(f"-----------------------------------------------------\n")
        ##########################################################################################################
        # FWZ-22: Upon entering the geofenced area with an advisory speed limit, the vehicle will initiate the
        #           deceleration command to the advisory speed limit within less than 1.3 seconds
        ##########################################################################################################
        print(f"Starting analysis for FWZ-22")
        try:
            is_passed = check_time_to_begin_deceleration(speed_limit_changes, response_times, FREIGHT_ADVISORY_SPEED_RESPONSE_SEC, stats_dir, data_dir)
            analysis_stats["check_time_to_begin_deceleration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking beginning of deceleration: {e}"
            )
            analysis_stats["check_time_to_begin_deceleration"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-23: The vehicle will decelerate and achieve the advisory speed limit prior to reaching the work zone.
        ##########################################################################################################
        print(f"Starting analysis for FWZ-23")
        try:
            workzone_lanelet_id=0
            is_passed = check_speed_before_workzone(mcap_path, start_time, end_time, workzone_lanelet_id, FREIGHT_ADVISORY_SPEED_LIMIT_MS, FREIGHT_MAINTAIN_SPEED_RANGE_MS)
            analysis_stats["check_speed_before_workzone"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking that advisory speed is reached prior to workzone: {e}"
            )
            analysis_stats["check_speed_before_workzone"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-24: Upon entering the geofenced area with an advisory speed limit, the actual trajectory
        #           to the reduced speed operations will include an acceleration section. The average
        #           deceleration over the entire section shall be no greater than 2 m/s^2.
        ##########################################################################################################
        print(f"Starting analysis for FWZ-24")
        try:
            is_passed = check_deceleration_for_geofence(time_enter_geofence, accelerations_times, FREIGHT_MAX_DECELERATION_MS2)
            analysis_stats["check_deceleration_for_geofence"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking the deceleration in geofence: {e}"
            )
            analysis_stats["check_deceleration_for_geofence"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-25: After exiting the geofenced area with an advisory speed limit, the vehicle will begin
        #           accelerating back to the origianl speed limit within less than 1.3 seconds
        ##########################################################################################################
        print(f"Starting analysis for FWZ-25")
        try:
            is_passed = check_time_to_begin_acceleration(speed_limit_changes, response_times, FREIGHT_ADVISORY_SPEED_RESPONSE_SEC, stats_dir, data_dir)
            analysis_stats["check_time_to_begin_acceleration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for start time of acceleration in geofence: {e}"
            )
            analysis_stats["check_time_to_begin_acceleration"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-26: After exiting the geofenced area with an advisory speed limit, the actual trajectory back to
        #           normal operations will include an acceleration section. The average acceleration over the
        #           entire section shall be no less than 1 m/s^2, and the average acceleration over any 1-second
        #           portion of the section shall be no greater than 2.0 m/s^2
        ##########################################################################################################
        print(f"Starting analysis for FWZ-26")
        try:
            is_passed = check_acceleration_after_geofence(time_exit_geofence, accelerations_times, FREIGHT_MIN_AVERAGE_ACCELERATION_MS2, avg_accelerations_times, FREIGHT_MAX_SECTION_ACCELERATION_MS2)
            analysis_stats["check_acceleration_after_geofence"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for acceleration after geofence: {e}"
            )
            analysis_stats["check_acceleration_after_geofence"] = None
        print(f"-----------------------------------------------------\n")

        ##########################################################################################################
        # FWZ-31: The time taken between when a TCR is sent from the vehicle to a TCM is received by11
        #           the vehicle shall be less than 1 second
        ##########################################################################################################
        print(f"Starting analysis for FWZ-31")
        try:
            is_passed = check_tcm_response_time(mcap_path, FREIGHT_TCR_SEND_TO_TCM_RECEIVE_DELAY_SEC, stats_dir, data_dir, plots_dir)
            analysis_stats["check_tcm_response_time"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for tcr sent and tcm received: {e}"
            )
            analysis_stats["check_tcm_response_time"] = None
        print(f"-----------------------------------------------------\n")

        all_analysis_stats.append(analysis_stats)

    return all_analysis_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all workzone analysis on multiple MCAP files in a given directory"

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
            args.input_dir, analyze_mcap_file_for_workzone_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
