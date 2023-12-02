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

# Instructions for running script:

# ----------------------------------------
# WORKSPACE SETUP INSTRUCTIONS
# ----------------------------------------
# This script requires a workspace setup as follows:
#   <workspace-directory>/carma-analytics-fotda
#   <workspace-directory>/tracetools_analysis/    NOTE: 'foxy' branch required; instructions included below
#   <workspace-directory>/ros2_tracing/           NOTE: 'foxy' branch required; instructions included below
#   The 'tracetools_analysis' and 'ros2_tracing' repos can be cloned out via git and set to their 'foxy' branch
#         git clone -b foxy https://github.com/ros-tracing/tracetools_analysis
#         git clone -b foxy https://github.com/ros2/ros2_tracing


# ----------------------------------------
# DEPENDENCIES
# ----------------------------------------
# Python 3.8
# Numpy: sudo apt-get install python3-numpy
# Pandas: sudo apt-get install python3-pandas
# Babeltrace and lttng with Python Bindings: sudo apt-get install python3-babeltrace python3-lttng


# ----------------------------------------
# SCRIPT USAGE INSTRUCTIONS
# ----------------------------------------
# From terminal, run 'python3 analyze_callback_durations.py"
# Additional arguments supported:
#       -v  --verbose    | Print out debug information to the terminal when analyzing a trace session
#       -sp --show-plots | Display plots immediately when they are generated. Regardless of this flag,
#                        |      plots will still be saved in an output directory when generated.
# NOTE: Search for all 'TODO for user' statements in this script to find parameters that can be customized
#       by the user prior to running this analysis script.

# ----------------------------------------
# SCRIPT OUTPUTS
# ----------------------------------------
# For each trace session analyzed by this script, a new folder will be created (in the same directory as
# this script) containing the analysis results for that trace session. Within that results folder, there will
# be one .csv file containing a summary of the callback duration statistics for each analyzed callback.
# Additionally, two plots (each stored as a separate .png file) will be generated for each callback: one containing
# a line chart of callback durations vs. time, and one containing a histogram of the callback durations. 

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

def plot_callback_durations_line_chart(duration_df, callback_description, results_directory, show_plots):
    '''
    Plot callback durations vs. time for a given callback

    :param duration_df: Pandas dataframe for a specific a callback. Contains duration for each time the callback was processed. 
    :param callback_description: String describing the specific callback.
    :param results_directory: String containing the directory for which the generated plot will be stored.
    :param show_plots: Flag indicating whether the generated plot should be immediately displayed to the user before being saved.

    :return: None
    '''

    ax = duration_df.plot(x='timestamp', y='duration')      
    plt.rc('axes', labelsize=12)  # Set font size of the axes labels
    plt.rc('legend', fontsize=10)  # Set font size of the legend text
    ax.get_legend().remove()
    ax.set_title(callback_description)
    ax.set_ylabel("Callback Duration (ms)")
    ax.set_xlabel("Seconds since CARMA was started (sec)")
    
    filename = str(results_directory) + "/" + str(callback_description.replace("/","-")) + "-linechart.png"
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
                               components_to_analyze, callbacks_to_ignore, 
                               show_plots=False, verbose=False):
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
            if "PlanTrajectory" in symbol:
                # NOTE: Currently can't relate a PlanTrajectory Service Callback to the source node. 
                #       As a workaround, it shouldn't be too difficult to relate the data to a specific 
                #       Tactical Plugin for a given test.
                callback_description = "PlanTrajectory Service Callback"
            if "PlanManeuvers" in symbol:
                # NOTE: Currently can't relate a PlanManeuvers Service Callback to the source node. 
                #       As a workaround, it shouldn't be too difficult to relate the data to a specific 
                #       Strategic Plugin for a given test.
                callback_description = "PlanManeuvers Service Callback"
            
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
            plot_callback_durations_line_chart(duration_df, callback_description, results_directory, show_plots)
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

def main():  

    # Parse command line arguments to set 'verbose_flag' and 'show_plots_flag'
    verbose_flag = False # True if user wants debug information outputted directly to terminal
    show_plots_flag = False # True if user wants generated plots to be displayed (regardless of this setting, plots will always be saved as a .png)
    if len(sys.argv) > 0:
        for arg in sys.argv[1:]:
            if(arg == "-v" or arg == "--verbose"):
                verbose_flag = True
            elif(arg == "-sp" or arg == "--show-plots"):
                show_plots_flag = True
            else:
                print("Unrecognized argument: " + str(arg))

    # TODO for user: All trace sessions should be stored in a central directory. Add one or more trace sessions to
    #      the 'trace_sessions' list, and set 'trace_sessions_directory' to the central directory that they're stored in.
    trace_sessions = ["example-trace-directory-1",
                      "example-trace-directory-2",
                      "example-trace-directory-3"]
    trace_session_directory = "/example-directory-containing-trace-sessions" 

    session_num = 1
    for trace_session in trace_sessions:
        # Analyze each trace session in 'tracing_sessions'

        print("**************************************************************")
        trace_path = trace_session_directory + trace_session + "/ust"
        print("Analyzing trace session: " + str(trace_path) + " (" + str(session_num) + " of " + str(len(trace_sessions)) + ")")

        # Create a folder that results and plots will be saved in
        results_directory = str(trace_session) + "-results"
        os.makedirs(results_directory, exist_ok=True) 
        current_directory = os.path.dirname(os.path.realpath(__file__))
        print("All generated statistics and plots will be stored in directory: " + str(current_directory) + "/" + str(results_directory))

        # Process data in tracing session
        # References data loading steps from tracetools_analysis 'callback_durations.ipny' example 
        #       Jupyter Notebook: https://github.com/ros-tracing/tracetools_analysis/blob/foxy/tracetools_analysis/analysis/callback_duration.ipynb
        events = load_file(trace_path)
        handler = Ros2Handler.process(events)
        data_util = Ros2DataModelUtil(handler.data) 
        callback_symbols = data_util.get_callback_symbols() # Mappings between a callback object and its resolved symbol.

        # Obtain informative timestamps from trace session
        timestamp_carma_started = get_timestamp_carma_started(data_util, callback_symbols, verbose_flag)
        timestamp_carma_engaged = get_timestamp_carma_engaged(data_util, callback_symbols, verbose_flag)

        # TODO for user: Update the logical groups (add nodes, remove nodes, etc.) as needed to 
        #      analyze nodes/components that you're interested in. The below logical groups are just an example,
        #      but include all functional ROS 2 strategic and tactical plugins at the time this
        #      script was created.

        # Organize ROS 2 nodes and services into logical groups
        planning_nodes = ["arbitrator", 
                          "plan_delegator"]

        strategic_plugin_nodes = ["route_following_plugin", 
                                  "approaching_emergency_vehicle_plugin",
                                  "lci_strategic_plugin", 
                                  "sci_strategic_plugin",
                                  "platoon_strategic_ihp"] 

        tactical_plugin_nodes = ["inlanecruising_plugin", 
                                 "cooperative_lanechange", 
                                 "stop_and_wait_plugin", 
                                 "yield_plugin",
                                 "intersection_transit_maneuvering",
                                 "light_controlled_intersection_tactical_plugin",
                                 "platooning_tactical_plugin",
                                 "stop_controlled_intersection_tactical_plugin"] 
        
        control_nodes = ["trajectory_executor", 
                         "pure_pursuit", 
                         "twist_filter", 
                         "twist_gate"]

        # NOTE: Currently, service callback trace logs only contain the source node's base class (rather than the node's descriptive name).
        #       An example of this is that trace logs will output several separate "PlanManeuvers Service Callback" analyses (1 for each activated Strategic Plugin), all
        #       attributed to PluginBaseNode. This can make it challenging to analyze service callbacks, but typically a user can detect which node is responsible
        #       for each service callback based on context from the analysis (for example, stop_and_wait_plugin will typically be called more frequently near
        #       the end of a trace log).
        planning_service_callbacks = ["PlanManeuvers", 
                                      "PlanTrajectory"]

        v2x_nodes = ["cpp_message", 
                     "j2735_convertor",
                     "bsm_generator"]

        # Example of combining logical groups for planning and control ROS 2 stack
        components_to_analyze = planning_nodes + strategic_plugin_nodes + tactical_plugin_nodes + \
                                    control_nodes + planning_service_callbacks + v2x_nodes
        
        # TODO for user: These are callbacks to ignore; callbacks containing these strings typically do not affect
        #      CARMA Platform planning and controls, and can be edited as needed
        callbacks_to_ignore = ["parameter_events", 
                               "georeference", 
                               "system_alert", 
                               "ChangeState", 
                               "PluginBaseNode", 
                               "carma_wm",
                               "ComponentManager"]
        
        # Perform statistical analysis on callbacks included in the trace session. Output results to 
        #       a .csv file and save informative plots for each callback
        analyze_callback_durations(data_util, callback_symbols, results_directory,
                               timestamp_carma_started, trace_session,
                               components_to_analyze, callbacks_to_ignore, 
                               show_plots_flag, verbose_flag)

        session_num += 1


if __name__ == "__main__":
    main()
