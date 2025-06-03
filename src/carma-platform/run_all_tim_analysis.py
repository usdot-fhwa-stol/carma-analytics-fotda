from pathlib import Path
from guidance_scripts import (
    get_engage_time,
    get_geofence_entrance_and_exit_times,
    get_route_original_speed,
    run_speed_limit_change_response_analysis,
    create_geofence_acceleration_plot,
    run_acceleration_comfort_analysis,
    check_in_geofence_speed_limits,
    check_geofence_in_reroute,
    check_reroute_duration,
    check_lanechange_lateral_velocity,
    check_lanechange_duration,
    check_time_to_begin_deceleration,
    check_speed_before_workzone,
    check_deceleration_for_geofence,
    check_acceleration_after_geofence,
    check_steady_state_after_geofence,
    get_planner_trajectory_intervals,
    run_crosstrack_analysis
    )
from message_scripts import (
    check_message_broadcast_rate,
)
from run_all_analysis import run_all_analysis
import argparse
import argcomplete

############################################################################
# Constant Values - adjust these as needed for your metrics/environment
############################################################################
DEG_TO_RAD = 0.0174533
MPH_TO_MPS = 0.44704
INCOMING_MOBILITY_OPERATION_TOPIC = '/message/incoming_mobility_operation'
TIM_LANECHANGE_DURATION = 5
TIM_MAINTAIN_SPEED_RANGE_MS = 0.89408 # 0.89408 m/s is 2 mph
TIM_TCM_ACKNOWLEDGEMENT_DELAY = 1
TIM_UPDATE_ACTIVE_ROUTE = 3
TIM_MIN_LAT_VELOCITY = 0.5
TIM_MAX_LAT_VELOCITY = 1.25
TIM_ADVISORY_SPEED_RESPONSE = 1.3
TIM_SPEED_LIMIT_THRESHOLD = 0.05
TIM_MAX_DECELERATION = -2
TIM_MIN_AVERAGE_ACCELERATION = 1
TIM_MAX_SECTION_ACCELERATION = 2
TIM_TCR_SEND_TO_TCM_RECEIVE_DELAY = 1
TIM_ADVISORY_SPEED_LIMIT_MS = 6.7 # 6.7 m/s is 15 mph
TIM_MIN_STEADY_STATE_DURATION_AFTER_GEOFENCE_S = 5 # 5 seconds of steady state after geofence exit
CROSS_TRACK_ERROR_THRESHOLD_METER = 0.1
CROSS_TRACK_ERROR_THRESHOLD_PERCENTILE = None
CLOSED_LANELETS = [1245999] # Lanelet ID for the closed lanelets
SPEED_ADVISORY_LANELET_ID = 1302198  # Lanelet ID for the speed advisory lanelet


def analyze_mcap_file_for_tim_analysis(
    mcap_path: Path, output_dir: Path, stats_dir: Path, data_dir: Path, plots_dir: Path) -> list:
    """
    Extract single MCAP file and run all TIM analysis (Traffic Incident Management - Move Over)
    NOTE: TIM use case uses Mobility Operation Message (MOM), which generates a
    TrafficControlMessage (TCM) message. So TCM related function can be used to verify.
    """
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
        time_enter_geofence, time_exit_geofence, found_geofence_times = (
            get_geofence_entrance_and_exit_times(mcap_path)
        )
        print(f"Enter time: {time_enter_geofence} Exit time: {time_exit_geofence}"
              f"Found: {found_geofence_times}\n")

    except Exception as e:
        print(
            f"Error analyzing {mcap_path} for getting geofence times: {e}\n"
        )
        return None

    # Get the original speed limit
    print("\nGetting the original route speed limit")
    original_speed_limit_ms = get_route_original_speed(mcap_path, engage_time)
    print(f"Original Route Speed Limit: {original_speed_limit_ms} m/s\n")

    print("\nRunning Speed Limit Analysis")
    # Run the speed limit analysis for the entire period of time
    _, _, _, speed_limit_changes, response_times = (
        run_speed_limit_change_response_analysis(
            mcap_path, steady_state_indication_time=TIM_ADVISORY_SPEED_RESPONSE,
            save_stats_dir=stats_dir, save_data_dir=data_dir, save_plot_dir=plots_dir)
    )

    print("\nRunning Acceleration Analysis")
    # Run the acceleration analysis for the entire period of time
    _, _, _, _, accelerations, avg_accelerations, timepoints, avg_timepoints = (
        run_acceleration_comfort_analysis(mcap_path, TIM_MAX_DECELERATION,
        save_stats_dir=stats_dir, save_data_dir=data_dir, save_plot_dir=plots_dir)
    )
    accelerations_times = []
    for (timepoint, acceleration) in zip(timepoints, accelerations):
        accelerations_times.append((timepoint, acceleration))
    avg_accelerations_times = []
    for (timepoint, avg_acceleration) in zip(avg_timepoints, avg_accelerations):
        avg_accelerations_times.append((timepoint, avg_acceleration))

    create_geofence_acceleration_plot(accelerations_times,
        avg_accelerations_times, time_enter_geofence, time_exit_geofence, plots_dir)

    # Run all the tests from engage to disengage
    intervals = [(engage_time, disengage_time)]
    all_analysis_stats = []

    print(f"-----------------------------------------------------")
    print(f"-------------Beginning TIM analysis for {mcap_path}-------------")
    print(f"-----------------------------------------------------")
    for start_time, end_time in intervals:
        analysis_stats = {}
        ############################################################################################
        # FWZ-7: The vehicle receives a message from CARMA Messenger that includes the new speed
        # limit for the geofence area. The vehicle successfully processes this new speed limit.
        ############################################################################################
        print(f"Starting analysis for FWZ-7")
        try:
            is_passed = check_in_geofence_speed_limits(
                mcap_path, time_enter_geofence,
                time_exit_geofence, TIM_ADVISORY_SPEED_LIMIT_MS, data_dir)
            analysis_stats["fwz-7-check_in_geofence_speed_limits"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for receiving new speed limit from ERV: {e}"
            )
            analysis_stats["fwz-7-check_in_geofence_speed_limits"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-8: The vehicle receives a message from CM that includes info about the closed
        # lane ahead. The vehicle successfully processes this closed lane information.
        ############################################################################################
        print(f"Starting analysis for FWZ-8")
        try:
            _, fwz8_is_passed = check_geofence_in_reroute(mcap_path, CLOSED_LANELETS, data_dir)
            analysis_stats["fwz-8-check_geofence_in_reroute"] = fwz8_is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for processing the closed lane message from CC: {e}"
            )
            analysis_stats["fwz-8-check_geofence_in_reroute"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-11: After the vehicle has received a message from CM with the Emergency Vehicle Zone
        # information, the vehicle shall successfully update its active route in less than 3 seconds
        # to avoid the Work Zone.
        ############################################################################################
        print(f"Starting analysis for FWZ-11")
        try:
            is_passed = check_reroute_duration(mcap_path, TIM_UPDATE_ACTIVE_ROUTE, data_dir)
            analysis_stats["fwz-11-check_reroute_duration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking reroute duration: {e}"
            )
            analysis_stats["fwz-11-check_reroute_duration"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-12:The vehicle changes lanes to follow the lane restrictions defined in the
        # TCM while in the work zone.
        ############################################################################################
        print(f"-----------------------------------------------------")
        print(f"Analysis for FWZ-12 is handled observationally")
        print(f"-----------------------------------------------------\n")
        ############################################################################################
        # FWZ-13: The vehicle lateral velocity during a lane change remains between 0.5 m/s
        # and 1.25 m/s
        ############################################################################################
        print(f"Starting analysis for FWZ-13")
        try:
            is_passed, _, _ = check_lanechange_lateral_velocity(
                mcap_path, TIM_MIN_LAT_VELOCITY, TIM_MAX_LAT_VELOCITY,
                stats_dir, data_dir, plots_dir)
            analysis_stats["fwz-13-check_lanechange_lateral_velocity"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking lane change velocity: {e}"
            )
            analysis_stats["fwz-13-check_lanechange_lateral_velocity"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-14: Lane change happens within 5 seconds
        ############################################################################################
        print(f"Starting analysis for FWZ-14")
        try:
            is_passed, _ = check_lanechange_duration(
                mcap_path, start_time, TIM_LANECHANGE_DURATION, stats_dir, data_dir)
            analysis_stats["fwz-14-check_lanechange_duration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking lane change duration: {e}"
            )
            analysis_stats["fwz-14-check_lanechange_duration"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-19: CM will broadcast the MOM at 10 Hz
        ############################################################################################
        print(f"Starting analysis for FWZ-19")
        try:
            is_passed, _, _, _, _ = check_message_broadcast_rate(
                mcap_path, INCOMING_MOBILITY_OPERATION_TOPIC, 10,
                start_time=engage_time,end_time=disengage_time, save_plot_dir=plots_dir,
            )

            analysis_stats["fwz-19-check_message_broadcast_rate"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing CMV data for MOM broadcast rate of ERV: {e}"
            )
            analysis_stats["fwz-19-check_message_broadcast_rate"] = None
        print(f"-----------------------------------------------------\n")


        ############################################################################################
        # FWZ-22: Upon entering the geofenced area with an advisory speed limit, the vehicle will
        # initiate the deceleration command to the advisory speed limit within less than 1.3 seconds
        ############################################################################################
        print(f"Starting analysis for FWZ-22")
        try:
            is_passed = check_time_to_begin_deceleration(
                speed_limit_changes, response_times,
                TIM_ADVISORY_SPEED_RESPONSE, stats_dir, data_dir)
            analysis_stats["fwz-22-check_time_to_begin_deceleration"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking beginning of deceleration: {e}"
            )
            analysis_stats["fwz-22-check_time_to_begin_deceleration"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-23: The vehicle will decelerate and achieve the advisory speed limit prior to
        # reaching the work zone.
        ############################################################################################
        print(f"Starting analysis for FWZ-23")
        try:
            is_passed = check_speed_before_workzone(
                mcap_path, start_time, end_time, SPEED_ADVISORY_LANELET_ID,
                TIM_ADVISORY_SPEED_LIMIT_MS, TIM_MAINTAIN_SPEED_RANGE_MS)
            analysis_stats["fwz-23-check_speed_before_workzone"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking that advisory"
                f"speed is reached prior to workzone: {e}"
            )
            analysis_stats["fwz-23-check_speed_before_workzone"] = None
        print(f"-----------------------------------------------------\n")


        ############################################################################################
        # FWZ-24: Upon entering the geofenced area with an advisory speed limit, the actual
        # trajectory to the reduced speed operations will include an acceleration section.
        # The average deceleration over the entire section shall be no greater than 2 m/s^2.
        ############################################################################################
        print(f"Starting analysis for FWZ-24")
        try:
            is_passed = check_deceleration_for_geofence(
                time_enter_geofence, accelerations_times, TIM_MAX_DECELERATION)
            analysis_stats["fwz-24-check_deceleration_for_geofence"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for checking the deceleration in geofence: {e}"
            )
            analysis_stats["fwz-24-check_deceleration_for_geofence"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-26: After exiting the geofenced area with an advisory speed limit, the trajectory
        # back to normal operations will include an accel section. The average acceleration over the
        # entire section shall be no less than 1 m/s^2, and the average acceleration over any 1-sec
        # portion of the section shall be no greater than 2.0 m/s^2
        ############################################################################################
        print(f"Starting analysis for FWZ-26")
        try:
            is_passed = check_acceleration_after_geofence(
                time_exit_geofence, accelerations_times, TIM_MIN_AVERAGE_ACCELERATION,
                avg_accelerations_times, TIM_MAX_SECTION_ACCELERATION)
            analysis_stats["fwz-26-check_acceleration_after_geofence"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for acceleration after geofence: {e}"
            )
            analysis_stats["fwz-26-check_acceleration_after_geofence"] = None
        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-27: When not changing lanes, the vehicle never crosses the left or right lane line
        # associated with its current lane by more than 0.1 meters while the CARMA system is engaged
        ############################################################################################
        planner_plugin = "cooperative_lanechange"
        lanechange_intervals = get_planner_trajectory_intervals(mcap_path, planner_plugin)

        if (not lanechange_intervals or
            len(lanechange_intervals) < 2):
            print(
                f"Error analyzing {mcap_path} for checking lane change intervals: "
                f"Expected at least 2 intervals, found {len(lanechange_intervals)}"
            )
            analysis_stats["fwz-27-check_lanechange_intervals"] = None
        else:
            straight_intervals = [
                (engage_time, lanechange_intervals[0][0]),
                (lanechange_intervals[0][1], lanechange_intervals[1][0]),
                (lanechange_intervals[1][1], disengage_time)
            ]

            for straight_interval in straight_intervals:
                print(f"Analyzing straight interval: {straight_interval}")
                start_time, end_time = straight_interval
                try:
                    is_passed, _, _, _, _ = run_crosstrack_analysis(
                        mcap_path,
                        CROSS_TRACK_ERROR_THRESHOLD_METER,
                        CROSS_TRACK_ERROR_THRESHOLD_PERCENTILE,
                        start_time,
                        end_time,
                        stats_dir / f"crosstrack-error-during-straightaway-{start_time}-{end_time}",
                        data_dir,
                        plots_dir,
                    )
                    analysis_stats["fwz-27-run_crosstrack_analysis"] = is_passed
                    if not is_passed:
                        print(
                            f"Cross-track error exceeded threshold in interval {straight_interval}"
                            f" during lanechanges, not checking other straight sections."
                        )
                        break
                except Exception as e:
                    print(
                        f"Error analyzing {mcap_path} for fwz-27-run_crosstrack_analysis: {e}"
                    )
                    analysis_stats["fwz-27-run_crosstrack_analysis"] = None

        print(f"-----------------------------------------------------\n")

        ############################################################################################
        # FWZ-28: After exiting the geofenced area, the planned route must end with the vehicle
        # having been at steady state for at least 5 seconds. 
        ############################################################################################
        print(f"Starting analysis for FWZ-28")
        try:
            is_passed = check_steady_state_after_geofence(
                mcap_path,
                time_exit_geofence, disengage_time, TIM_SPEED_LIMIT_THRESHOLD,
                TIM_MIN_STEADY_STATE_DURATION_AFTER_GEOFENCE_S,
                TIM_MAINTAIN_SPEED_RANGE_MS)
            analysis_stats["fwz-28-check_steady_state_after_geofence"] = is_passed
        except Exception as e:
            print(
                f"Error analyzing {mcap_path} for acceleration after geofence: {e}"
            )
            analysis_stats["fwz-28-check_steady_state_after_geofence"] = None
        print(f"-----------------------------------------------------\n")

        all_analysis_stats.append(analysis_stats)



    return all_analysis_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all tim analysis on multiple MCAP files in a given directory"

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
            args.input_dir, analyze_mcap_file_for_tim_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
