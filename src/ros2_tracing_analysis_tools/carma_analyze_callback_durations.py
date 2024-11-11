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

# Instructions for running script:

# ----------------------------------------
# WORKSPACE SETUP INSTRUCTIONS
# ----------------------------------------
# This script requires a workspace setup as follows:
#   <workspace-directory>/carma-analytics-fotda
#   <workspace-directory>/tracetools_analysis/    NOTE: 'foxy' branch required; instructions included below
#   <workspace-directory>/ros2_tracing/           NOTE: 'foxy' branch required; instructions included below
#   The 'tracetools_analysis' and 'ros2_tracing' repos can be cloned out via git and set to their 'foxy' branch
#         git clone -b foxy https://github.com/usdot-fhwa-stol/tracetools_analysis
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
# a scatter plot of callback durations vs. time, and one containing a histogram of the callback durations.

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

def aggregate_session_statistics(results_dir):
    """
    Aggregate statistics across all analyzed sessions, properly accounting for 
    sequential pipeline behavior.
    
    Args:
        results_dir (str or Path): Directory containing analysis results
    """
    results_path = Path(results_dir)
    session_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and d.name.startswith('my-tracing-session')]
    
    if not session_dirs:
        raise ValueError(f"No session directories found in {results_dir}")
    
    all_component_stats = []
    
    # Collect stats from all sessions
    for session_dir in session_dirs:
        component_files = list(session_dir.glob('component_stats_*.csv'))
        if component_files:
            latest_component = max(component_files, key=lambda x: x.stat().st_mtime)
            comp_df = pd.read_csv(latest_component)
            comp_df['Session'] = session_dir.name
            all_component_stats.append(comp_df)
    
    if not all_component_stats:
        raise ValueError("No statistics files found")
        
    # Combine all sessions' data
    combined_df = pd.concat(all_component_stats, ignore_index=True)
    
    # Calculate per-component statistics across sessions
    component_summary = combined_df.groupby('Node/Component').agg({
        'Mean (ms)': ['mean', 'std', 'min', 'max'],
        'Min (ms)': 'min',
        'Max (ms)': 'max',
        'Median (ms)': 'mean',  # Using mean of medians
        'Std Dev': ['mean', 'std'],
        'Count': 'sum'
    }).round(6)
    
    component_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                               for col in component_summary.columns]
    component_summary = component_summary.reset_index()
    
    # Calculate pipeline statistics (sequential addition)
    session_summaries = []
    for session in combined_df['Session'].unique():
        session_data = combined_df[combined_df['Session'] == session]
        
        # For sequential pipeline:
        total_mean = session_data['Mean (ms)'].sum()  # Means add directly
        total_variance = (session_data['Std Dev'] ** 2).sum()  # Variances add for independent components
        total_std = np.sqrt(total_variance)
        total_min = session_data['Min (ms)'].sum()  # Best case: all components at their minimum
        total_max = session_data['Max (ms)'].sum()  # Worst case: all components at their maximum
        
        session_summaries.append({
            'Session': session,
            'Total Mean (ms)': total_mean,
            'Total Std Dev (ms)': total_std,
            'Total Min (ms)': total_min,
            'Total Max (ms)': total_max
        })
    
    pipeline_summary = pd.DataFrame(session_summaries)
    
    # Calculate aggregate pipeline statistics across sessions
    pipeline_stats = {
        'Total Pipeline Statistics (ms)': [
            'Mean Latency',
            'Std Dev',
            'Min Latency',
            'Max Latency',
            'Sessions Analyzed'
        ],
        'Value': [
            f"{pipeline_summary['Total Mean (ms)'].mean():.4f} ± {pipeline_summary['Total Mean (ms)'].std():.4f}",
            f"{pipeline_summary['Total Std Dev (ms)'].mean():.4f} ± {pipeline_summary['Total Std Dev (ms)'].std():.4f}",
            f"{pipeline_summary['Total Min (ms)'].mean():.4f} ± {pipeline_summary['Total Min (ms)'].std():.4f}",
            f"{pipeline_summary['Total Max (ms)'].mean():.4f} ± {pipeline_summary['Total Max (ms)'].std():.4f}",
            len(session_dirs)
        ]
    }
    
    pipeline_stats_df = pd.DataFrame(pipeline_stats)
    
    # Save results
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save component summary
    component_summary_path = results_path / f"aggregated_component_stats_{timestamp}.csv"
    component_summary.to_csv(component_summary_path, index=False)
    
    # Save pipeline summary
    pipeline_summary_path = results_path / f"aggregated_pipeline_stats_{timestamp}.csv"
    pipeline_stats_df.to_csv(pipeline_summary_path, index=False)
    
    # Create human-readable summary
    summary_path = results_path / f"aggregated_analysis_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("Aggregated Sequential Pipeline Statistics\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Sessions Analyzed: {len(session_dirs)}\n\n")
        
        f.write("Component-wise Statistics:\n")
        f.write(component_summary.to_string(index=False))
        
        f.write("\n\nPipeline Statistics:\n")
        f.write(pipeline_stats_df.to_string(index=False))
        
        f.write("\n\nPer-Session Pipeline Totals:\n")
        f.write(pipeline_summary.to_string(index=False))
        
        f.write("\n\nNotes:")
        f.write("\n- Statistics represent end-to-end sequential processing time")
        f.write("\n- Mean ± Std shows variation across sessions")
        f.write("\n- Min/Max represent best/worst case scenarios")
        f.write("\n- Total Std Dev accounts for component independence")
    
    print(f"\nAggregated analysis results saved in {results_path}:")
    print(f"- Component summary: {component_summary_path.name}")
    print(f"- Pipeline summary: {pipeline_summary_path.name}")
    print(f"- Analysis summary: {summary_path.name}")
    
    return component_summary, pipeline_stats_df, pipeline_summary

# Example usage in main():
# After processing all sessions, add:
try:
    component_summary, pipeline_summary = aggregate_session_statistics(output_dir)
    if verbose_flag:
        print("\nComponent-wise Summary Statistics:")
        print(component_summary)
        print("\nPipeline Summary Statistics:")
        print(pipeline_summary)
except Exception as e:
    print(f"Error aggregating statistics: {e}")

def initialize_directory(trace_session_directory, trace_session, session_num, trace_sessions):
    # Analyze each trace session in 'tracing_sessions'
    print("**************************************************************")
    trace_path = trace_session_directory + "/" + trace_session + "/ust"
    print("Analyzing trace session: " + str(trace_path) + " (" + str(session_num) + " of " + str(len(trace_sessions)) + ")")

    # Create a folder that results and plots will be saved in
    results_directory = str(trace_session) + "-results"
    os.makedirs(results_directory, exist_ok=True)
    current_directory = os.path.dirname(os.path.realpath(__file__))
    print("All generated statistics and plots will be stored in directory: " + str(current_directory) + "/" + str(results_directory))
    
    return results_directory, trace_path

def initialize_ros2_tracing(trace_path):
    # Process data in tracing session
    # References data loading steps from tracetools_analysis 'callback_durations.ipny' example
    #       Jupyter Notebook: https://github.com/ros-tracing/tracetools_analysis/blob/foxy/tracetools_analysis/analysis/callback_duration.ipynb
    events = load_file(trace_path)
    handler = Ros2Handler.process(events)
    data_util = Ros2DataModelUtil(handler.data)
    callback_symbols = data_util.get_callback_symbols() # Mappings between a callback object and its resolved symbol.

    return data_util, callback_symbols

def analyze_sequential_latencies(input_dir):
    """
    Analyze sequential component latencies and save results in the input directory.
    
    Args:
        input_dir (str or Path): Path to directory containing CSV files
        
    Saves:
        - component_stats.csv: Component-wise statistics
        - pipeline_stats.csv: Overall pipeline statistics
        - analysis_summary.txt: Human-readable summary
    """
    input_path = Path(input_dir)
    all_components = []
    
    # Process each CSV file
    for csv_file in input_path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            component_stats = df.groupby('Node/Component').agg({
                'Mean (ms)': 'first',
                'Min (ms)': 'first',
                'Median (ms)': 'first',
                'Max (ms)': 'first',
                'Std Dev': 'first',
                'Count': 'first'
            }).reset_index()
            all_components.append(component_stats)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_components:
        raise ValueError(f"No valid CSV files found in {input_dir}")
    
    # Combine and process component statistics
    combined_stats = pd.concat(all_components, ignore_index=True)
    final_component_stats = combined_stats.groupby('Node/Component').agg({
        'Mean (ms)': 'mean',
        'Min (ms)': 'min',
        'Median (ms)': 'median',
        'Max (ms)': 'max',
        'Std Dev': 'mean',
        'Count': 'sum'
    }).reset_index()
    
    # Calculate pipeline statistics
    confidence_level = 0.95
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    total_variance_independent = np.sum(final_component_stats['Std Dev']**2)
    total_variance_correlated = np.sum(final_component_stats['Std Dev'])**2
    
    std_dev_independent = np.sqrt(total_variance_independent)
    std_dev_correlated = np.sqrt(total_variance_correlated)
    
    total_samples = int(final_component_stats['Count'].min())
    total_mean = float(final_component_stats['Mean (ms)'].sum())
    
    margin_error_independent = z_value * (std_dev_independent / np.sqrt(total_samples))
    margin_error_correlated = z_value * (std_dev_correlated / np.sqrt(total_samples))
    
    # Create pipeline stats DataFrame
    pipeline_stats = pd.DataFrame({
        'Metric': [
            'Total Mean (ms)',
            'Total Min (ms)',
            'Total Max (ms)',
            'Independent Std Dev',
            'Correlated Std Dev',
            'Independent CI Lower',
            'Independent CI Upper',
            'Correlated CI Lower',
            'Correlated CI Upper',
            'Total Samples'
        ],
        'Value': [
            total_mean,
            float(final_component_stats['Min (ms)'].sum()),
            float(final_component_stats['Max (ms)'].sum()),
            float(std_dev_independent),
            float(std_dev_correlated),
            float(total_mean - margin_error_independent),
            float(total_mean + margin_error_independent),
            float(total_mean - margin_error_correlated),
            float(total_mean + margin_error_correlated),
            total_samples
        ]
    })
    
    # Create timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save component statistics to CSV
    csv_path = input_path / f"component_stats_{timestamp}.csv"
    final_component_stats.to_csv(csv_path, index=False)
    
    # Save pipeline statistics to CSV
    pipeline_csv_path = input_path / f"pipeline_stats_{timestamp}.csv"
    pipeline_stats.to_csv(pipeline_csv_path, index=False)
    
    # Create human-readable summary
    summary_path = input_path / f"analysis_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("Sequential Component Latency Analysis\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Component-wise Statistics:\n")
        f.write(final_component_stats.to_string())
        
        f.write("\n\nPipeline Statistics:\n")
        f.write(pipeline_stats.to_string())
        
    print(f"\nAnalysis results saved in {input_path}:")
    print(f"- Component statistics: {csv_path.name}")
    print(f"- Pipeline statistics: {pipeline_csv_path.name}")
    print(f"- Analysis summary: {summary_path.name}")
    
    return final_component_stats, pipeline_stats


def main():
    """
    Main function to analyze ROS2 tracing data.
    Automatically detects tracing sessions in input directory.
    If output directory is not specified, creates 'analysis_results' in input directory.
    
    Usage: python script.py [-v|--verbose] [-sp|--show-plots] [-o|--output-dir output_dir] -i|--input-dir input_dir
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze ROS2 tracing data')
    parser.add_argument('-i', '--input-dir', help='Directory containing tracing sessions')
    parser.add_argument('-o', '--output-dir', help='Directory to save analysis results (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-sp', '--show-plots', action='store_true', help='Show plots during analysis')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    verbose_flag = args.verbose
    show_plots_flag = args.show_plots

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create 'analysis_results' directory inside input directory
        output_dir = input_dir / 'analysis_results'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all tracing sessions
    trace_sessions = [
        d.name for d in input_dir.iterdir() 
        if d.is_dir() and d.name.startswith('my-tracing-session')
    ]

    if not trace_sessions:
        print(f"No tracing sessions found in {input_dir}")
        return

    if verbose_flag:
        print(f"Found {len(trace_sessions)} tracing sessions:")
        for session in trace_sessions:
            print(f"  - {session}")
        print(f"\nResults will be saved in: {output_dir}")

    # Process each tracing session
    for session_num, trace_session in enumerate(sorted(trace_sessions), 1):
        session_output_dir = output_dir / trace_session
        session_output_dir.mkdir(exist_ok=True)

        if verbose_flag:
            print(f"\nProcessing session {session_num}/{len(trace_sessions)}: {trace_session}")

        # Initialize directory and get trace path
        results_directory, trace_path = initialize_directory(
            str(input_dir), trace_session, session_num, trace_sessions
        )

        # # Initialize ROS2 tracing
        # data_util, callback_symbols = initialize_ros2_tracing(trace_path)

        # # Get timestamps
        # timestamp_carma_started = get_timestamp_carma_started(
        #     data_util, callback_symbols, verbose_flag
        # )
        # timestamp_carma_engaged = get_timestamp_carma_engaged(
        #     data_util, callback_symbols, verbose_flag
        # )

        # # Define component groups
        # planning_nodes = [
        #     "arbitrator",
        #     "plan_delegator"
        # ]

        # strategic_plugin_nodes = [
        #     "route_following_plugin",
        #     "approaching_emergency_vehicle_plugin",
        #     "lci_strategic_plugin",
        #     "sci_strategic_plugin",
        #     "platoon_strategic_ihp"
        # ]

        # tactical_plugin_nodes = [
        #     "inlanecruising_plugin",
        #     "cooperative_lanechange",
        #     "stop_and_wait_plugin",
        #     "yield_plugin",
        #     "intersection_transit_maneuvering",
        #     "light_controlled_intersection_tactical_plugin",
        #     "platooning_tactical_plugin",
        #     "stop_controlled_intersection_tactical_plugin"
        # ]

        # control_nodes = [
        #     "trajectory_executor",
        #     "pure_pursuit",
        #     "twist_filter",
        #     "twist_gate"
        # ]

        # v2x_nodes = [
        #     "cpp_message",
        #     "j2735_convertor",
        #     "bsm_generator"
        # ]

        # # Combine all components
        # components_to_analyze = (
        #     planning_nodes + 
        #     strategic_plugin_nodes + 
        #     tactical_plugin_nodes + 
        #     control_nodes + 
        #     v2x_nodes
        # )

        # # Analyze callback durations
        # analyze_callback_durations(
        #     data_util, 
        #     callback_symbols, 
        #     session_output_dir,
        #     timestamp_carma_started, 
        #     trace_session,
        #     components_to_analyze,
        #     show_plots_flag, 
        #     verbose_flag
        # )
        
        # # Analyze sequential latencies
        # analyze_sequential_latencies(session_output_dir)

        

        if verbose_flag:
            print(f"Completed analysis for {trace_session}")

    aggregate_session_statistics(output_dir)

    print(f"\nAnalysis complete. Results saved in {output_dir}")


if __name__ == "__main__":
    main()
