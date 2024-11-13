#!/usr/bin/python3

#  Copyright (C) 2023 LEIDOS.
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not
#  use this file except in compliance with the License. You may obtain a copy of
#  the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations under
#  the License.

import sys
sys.path.insert(0, '../../../ros2_tracing/tracetools_read/')
sys.path.insert(0, '../../../tracetools_analysis/tracetools_analysis/')

import datetime as dt
import numpy as np
import pandas as pd

from tracetools_analysis.loading import load_file
from tracetools_analysis.processor.ros2 import Ros2Handler
from tracetools_analysis.utils.ros2 import Ros2DataModelUtil

import csv # Meaningful statistics are outputted to a csv file
import matplotlib.pyplot as plt
import os

from pathlib import Path
import scipy.stats as stats
import argparse

# TODO for user: These are callbacks to ignore; callbacks containing these strings typically do not affect
#      CARMA Platform planning and controls, and can be edited as needed
CALLBACKS_TO_IGNORE = ["parameter_events",
                        "list_parameters", #ros2 service
                        "georeference",
                        "load_node", #ros2 service
                        "system_alert",
                        "ChangeState",
                        "carma_wm",
                        "ComponentManager"]

def get_timestamp_carma_engaged(data_util, callback_symbols, verbose=False):
    '''
    Obtain the timestamp that CARMA Platform was engaged for a given trace session.

    :param data_util: Ros2DataModelUtil utility class containing the trace session event data
    :param callback_symbols: Mappings between a callback object and its resolved symbol.
    :param verbose: Flag indicating whether debug information should be printed to terminal.

    :return: Timestamp that CARMA Platform was engaged.
             NOTE: This is roughly the timestamp of the first service call to the Guidance node's 'SetGuidanceActive' service.
    '''

    # Initialize variables
    timestamp_carma_engaged = 0.0
    first = True

    # Find the earliest "SetGuidanceActive" service callback from the Trace Session. This timestamp will be used to
    #      approximate the timestamp that CARMA Platform was engaged.
    for obj, symbol in callback_symbols.items():
        owner_info = data_util.get_callback_owner_info(obj)
        if owner_info is None:
            owner_info = '[unknown]'

        # Create dataframe of durations for this callback
        if ("SetGuidanceActive" in owner_info) or ("SetGuidanceActive" in symbol):
            duration_df = data_util.get_callback_durations(obj)

            if first:
                timestamp_carma_engaged = duration_df['timestamp'].iloc[0]
                first = False
            else:
                if duration_df['timestamp'].iloc[0] < timestamp_carma_engaged:
                    timestamp_carma_engaged = duration_df['timestamp'].iloc[0]

    if(verbose):
        print("For this trace session, CARMA Platform engaged at : " + str(timestamp_carma_engaged))

    return timestamp_carma_engaged

def get_timestamp_carma_started(data_util, callback_symbols, verbose=False):
    '''
    Obtain the timestamp that CARMA Platform was started for a given trace session.

    :param data_util: Ros2DataModelUtil utility class containing the trace session event data
    :param callback_symbols: Mappings between a callback object and its resolved symbol.
    :param verbose: Flag indicating whether debug information should be printed to terminal.

    :return: Timestamp that CARMA Platform was started.
             NOTE: This is the timestamp of the earliest logged event from the Trace Session; this will be consider
                   the start time for CARMA Platform
    '''

    timestamp_carma_started = 0.0
    first = True
    for obj, symbol in callback_symbols.items():

        # Create dataframe of durations for this callback
        duration_df = data_util.get_callback_durations(obj)

        # Update 'carma_start_timestamp' if an earlier timestamp has been found
        if first:
            timestamp_carma_started = duration_df['timestamp'].iloc[0]
            first = False
        else:
            if duration_df['timestamp'].iloc[0] < timestamp_carma_started:
                timestamp_carma_started = duration_df['timestamp'].iloc[0]

    if(verbose):
        print("For this trace session, CARMA Platform started at : " + str(timestamp_carma_started))

    return timestamp_carma_started

def plot_callback_durations_scatter_plot(duration_df, callback_description, results_directory, show_plots):
    '''
    Plot callback durations vs. time for a given callback

    :param duration_df: Pandas dataframe for a specific a callback. Contains duration for each time the callback was processed.
    :param callback_description: String describing the specific callback.
    :param results_directory: String containing the directory for which the generated plot will be stored.
    :param show_plots: Flag indicating whether the generated plot should be immediately displayed to the user before being saved.

    :return: None
    '''

    ax = duration_df.plot(x='timestamp', y='duration', linestyle='', marker='o')
    plt.rc('axes', labelsize=12)  # Set font size of the axes labels
    plt.rc('legend', fontsize=10)  # Set font size of the legend text
    ax.get_legend().remove()
    ax.set_title(callback_description)
    ax.set_ylabel("Callback Duration (ms)")
    ax.set_xlabel("Seconds since CARMA was started (sec)")

    filename = str(results_directory) + "/" + str(callback_description.replace("/","-")) + "-scatter_plot.png"
    plt.savefig(filename, bbox_inches='tight')
    if(show_plots):
        plt.show()
    plt.close()

    return

def plot_callback_durations_histogram(duration_df, callback_description, results_directory, show_plots):
    '''
    Plot histogram of callback durations for a given callback

    :param duration_df: Pandas dataframe for a specific a callback. Contains duration for each time the callback was processed.
    :param callback_description: String describing the specific callback.
    :param results_directory: String containing the directory for which the generated plot will be stored.
    :param show_plots: Flag indicating whether the generated plot should be immediately displayed to the user before being saved.

    :return: None
    '''

    ax_hist = duration_df['duration'].hist()
    plt.rc('axes', labelsize=12)  # fontsize of the axes labels
    plt.rc('legend', fontsize=10)  # fontsize of the legend text
    ax_hist.set_title(callback_description)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_xlabel("Callback Duration (ms)")

    filename = str(results_directory) + "/" + str(callback_description.replace("/","-")) + "-histogram.png"
    plt.savefig(filename, bbox_inches='tight')
    if(show_plots):
        plt.show()
    plt.close()

    return

def analyze_callback_durations(data_util, callback_symbols, results_directory,
                               timestamp_start_analysis, trace_session_filename,
                               components_to_analyze, show_plots=False, verbose=False, 
                               callbacks_to_ignore = CALLBACKS_TO_IGNORE):
    '''
    Main function for analyzing callback durations. This function calls other functions as needed to
    generate statistics for a given callback and generate informative plots.

    :param data_util: Ros2DataModelUtil utility class containing the trace session event data
    :param callback_symbols: Mappings between a callback object and its resolved symbol.
    :param results_directory: String containing the directory for which the generated plot will be stored.
    :param timestamp_start_analysis: Timestamp from which to start analysis; all callbacks occurring before
                                     this timestamp will be discarded.
    :param trace_session_filename: String containing the filename of the trace session being analyzed.
    :param components_to_analyze: List of strings describing the nodes/components that the user wishes to have analyzed
    :param callbacks_to_ignore: List of strings with keywords of callbacks that the user wishes to ignore. These can be discarded.
    :param show_plots: Flag indicating whether the generated plot should be immediately displayed to the user before being saved.
    :param verbose: Flag indicating whether debug information should be printed to terminal.

    :return: None
    '''

    # Create .csv file in which results for each callback will be stored
    csv_results_filename = str(results_directory) + "/all_results_" + str(trace_session_filename) + ".csv"
    f = open(csv_results_filename, 'w')
    csv_results_writer = csv.writer(f)
    csv_results_writer.writerow(["Node/Component", "Callback Description", "Mean (ms)",
                                "Min (ms)", "Median (ms)", "Max (ms)", "Std Dev",
                                "Count"])

    for component in components_to_analyze:
        # For each component, log statisics and generate plots for callbacks. If a callback contains content
        #     that matches a string in "callbacks_to_ignore", the callback will be skipped (no results or
        #     plots will be generated).

        if(verbose):
            print("*******************************************************************")
            print("Analyzing " + str(component))
            print("*******************************************************************")

        for obj, symbol in callback_symbols.items():
            owner_info = data_util.get_callback_owner_info(obj)
            if owner_info is None:
                owner_info = "[unknown]"

            # Skip callback if it is not related to the current 'component' being analyzed
            if (component not in owner_info) and (component not in symbol):
                continue

            # Skip callback if it includes content that user wants to ignore
            if any((callback in owner_info or callback in symbol) for callback in callbacks_to_ignore):
                continue

            # Generate descriptive information for this callback
            callback_description = ""
            if "Timer" in owner_info:
                callback_description = component + " Timer Callback" + owner_info.split(",")[-1]
            if "Subscription" in owner_info:
                callback_description = component + " Subscription Callback" + owner_info.split(",")[-1]
            if "plan_trajectory" in owner_info:
                callback_description = component + " PlanTrajectory Callback" + owner_info.split(",")[-1]
            if "plan_maneuvers" in owner_info:
                callback_description = component + " PlanManeuvers Callback" + owner_info.split(",")[-1]

            # Create dataframe of durations for this callback
            duration_df = data_util.get_callback_durations(obj)

            # Remove all entries that occurred before the given 'timestamp_start_analysis'
            duration_df = duration_df[duration_df['timestamp'] > timestamp_start_analysis]

            # Update all timestamps to be "seconds since timestamp_start_analysis"
            duration_df['timestamp'] = duration_df['timestamp'] - timestamp_start_analysis

            # Change timestamp from np.datetime64 to seconds for easier statistical analysis
            duration_df['timestamp'] = duration_df['timestamp'] / np.timedelta64(1, 's')

            # If dataframe is empty, skip
            if(duration_df.empty):
                if(verbose):
                    print("Skipping empty dataframe: " + str(callback_description))
                continue
            else:
                if(verbose):
                    print(callback_description)

            # Extract statistics on the callback
            mean_duration_ms =    duration_df['duration'].mean()
            minimum_duration_ms = duration_df['duration'].min()
            median_duration_ms =  duration_df['duration'].median()
            maximum_duration_ms = duration_df['duration'].max()
            std_dev_duration_ms = duration_df['duration'].std()
            total_count =         duration_df['duration'].count()

            # Store statistics in .csv
            csv_results_writer.writerow([component, callback_description, mean_duration_ms,
                                        minimum_duration_ms, median_duration_ms, maximum_duration_ms,
                                        std_dev_duration_ms, total_count])

            # Generate plots for callback duration
            plot_callback_durations_scatter_plot(duration_df, callback_description, results_directory, show_plots)
            plot_callback_durations_histogram(duration_df, callback_description, results_directory, show_plots)

            if(verbose):
                print("Mean: " + str(mean_duration_ms) + " ms")
                print("Minimum: " + str(minimum_duration_ms) + " ms")
                print("Median: " + str(median_duration_ms) + " ms")
                print("Maximum: " + str(maximum_duration_ms) + " ms")
                print("Standard Deviation: " + str(std_dev_duration_ms) + " ms")
                print("Count: " + str(total_count))
                print("-------------------------")

    # Close .csv file
    f.close()
    return

def get_trace_path(trace_session_directory, trace_session, session_num, trace_sessions):
    # Analyze each trace session in 'tracing_sessions'
    print("**************************************************************")
    trace_path = trace_session_directory + "/" + trace_session + "/ust"
    print("Analyzing trace session: " + str(trace_path) + " (" + str(session_num) + " of " + str(len(trace_sessions)) + ")")
    return trace_path

def initialize_ros2_tracing(trace_path):
    # Process data in tracing session
    # References data loading steps from tracetools_analysis 'callback_durations.ipny' example
    #       Jupyter Notebook: https://github.com/ros-tracing/tracetools_analysis/blob/foxy/tracetools_analysis/analysis/callback_duration.ipynb
    events = load_file(trace_path)
    handler = Ros2Handler.process(events)
    data_util = Ros2DataModelUtil(handler.data)
    callback_symbols = data_util.get_callback_symbols() # Mappings between a callback object and its resolved symbol.

    return data_util, callback_symbols

def extract_all_traces(input_dir, output_dir=None, show_plots=False, verbose=False):
    """
    Process all trace sessions in the input directory, extracting callback statistics.
    
    Args:
        input_dir (str or Path): Directory containing trace sessions
        output_dir (str or Path, optional): Directory for output. If None, uses input_dir/analysis_results
        show_plots (bool): Whether to display plots
        verbose (bool): Whether to print detailed progress
    """
    input_path = Path(input_dir)
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path / 'analysis_results'
    
    output_path.mkdir(parents=True, exist_ok=True)

    # Find trace sessions
    trace_sessions = [
        d.name for d in input_path.iterdir() 
        if d.is_dir() and d.name.startswith('my-tracing-session')
    ]

    if not trace_sessions:
        print(f"No trace sessions found in {input_dir}")
        return

    # Define component groups
    planning_nodes = ["arbitrator", "plan_delegator"]
    strategic_plugin_nodes = [
        "route_following_plugin",
        "approaching_emergency_vehicle_plugin",
        "lci_strategic_plugin",
        "sci_strategic_plugin",
        "platoon_strategic_ihp"
    ]
    tactical_plugin_nodes = [
        "inlanecruising_plugin",
        "cooperative_lanechange",
        "stop_and_wait_plugin",
        "yield_plugin",
        "intersection_transit_maneuvering",
        "light_controlled_intersection_tactical_plugin",
        "platooning_tactical_plugin",
        "stop_controlled_intersection_tactical_plugin"
    ]
    control_nodes = [
        "trajectory_executor",
        "trajectory_follower",
        "twist_filter",
        "twist_gate"
    ]
    v2x_nodes = ["cpp_message", "j2735_convertor", "bsm_generator"]

    components_to_analyze = (
        planning_nodes + 
        strategic_plugin_nodes + 
        tactical_plugin_nodes + 
        control_nodes + 
        v2x_nodes
    )

    for session_num, trace_session in enumerate(sorted(trace_sessions), 1):
        session_output_dir = output_path / trace_session
        session_output_dir.mkdir(exist_ok=True)

        if verbose:
            print(f"\nProcessing session {session_num}/{len(trace_sessions)}: {trace_session}")

        try:
            # Initialize tracing
            trace_path = str(input_path / trace_session / "ust")
            events = load_file(trace_path)
            handler = Ros2Handler.process(events)
            data_util = Ros2DataModelUtil(handler.data)
            callback_symbols = data_util.get_callback_symbols()

            # Get timestamps
            timestamp_started = get_timestamp_carma_started(data_util, callback_symbols, verbose)

            # Analyze callbacks
            analyze_callback_durations(
                data_util, 
                callback_symbols, 
                session_output_dir,
                timestamp_started,
                trace_session,
                components_to_analyze,
                show_plots,
                verbose
            )

        except Exception as e:
            print(f"Error processing {trace_session}: {e}")
            continue

        if verbose:
            print(f"Completed trace extraction for {trace_session}")

    print(f"\nTrace extraction complete. Results saved in {output_path}")

def main():
    """
    Main function to extract trace data.
    
    Usage: python carma_analyze_callback_durations.py [-v] [-sp] [-o OUTPUT_DIR] input_dir
    """
    parser = argparse.ArgumentParser(description='Extract ROS2 trace data statistics')
    parser.add_argument('input_dir', help='Directory containing trace sessions')
    parser.add_argument('-o', '--output-dir', help='Directory to save results (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-sp', '--show-plots', action='store_true', help='Show plots during analysis')
    args = parser.parse_args()

    extract_all_traces(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()