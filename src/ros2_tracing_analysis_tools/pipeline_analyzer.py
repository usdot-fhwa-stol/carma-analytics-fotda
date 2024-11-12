import argparse
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats

# Pipeline Latency Analyzer for CARMA Platform

# This script analyzes sequential component latencies from trace data, focusing on
# end-to-end pipeline performance across multiple sessions.

# ----------------------------------------
# DEPENDENCIES
# ----------------------------------------
# - Python 3.8+
# - Required packages:
#     numpy
#     pandas
#     scipy

# ----------------------------------------
# USAGE INSTRUCTIONS
# ----------------------------------------
# 1. Basic usage (after running trace_analyzer.py):
#     python3 pipeline_analyzer.py /path/to/trace/results

# 2. All options:
#     python3 pipeline_analyzer.py [-h] [-o OUTPUT_DIR] [-p PLUGINS [PLUGINS ...]] [-v] input_dir

# Arguments:
#     input_dir             Directory containing trace analysis results
#     -o, --output-dir     Optional output directory (default: input_dir/analysis_results)
#     -p, --plugins        Space-separated list of specific plugins to analyze
#     -v, --verbose        Enable detailed progress output

# Default analyzed plugins if none specified:
#     - arbitrator
#     - plan_delegator
#     - trajectory_executor
#     - trajectory_follower
#     - twist_filter
#     - twist_gate

# ----------------------------------------
# OUTPUTS
# ----------------------------------------
# For each session:
# 1. Component statistics (CSV)
# 2. Pipeline statistics (CSV)
# 3. Analysis summary (TXT)

# Additionally creates aggregated statistics across all sessions:
# 1. aggregated_component_stats_{timestamp}.csv
# 2. aggregated_pipeline_stats_{timestamp}.csv
# 3. aggregated_analysis_summary_{timestamp}.txt

# Analysis includes:
# - Individual component statistics
# - End-to-end pipeline performance
# - Cross-session variations
# - Combined standard deviations (within + between session)

# Notes:
# - Processes trace analysis results from trace_analyzer.py
# - Calculates sequential pipeline metrics
# - Handles both single-session and multi-session analysis


def write_analysis_summary_to_text_file(file_path, title, timestamp, components_df, pipeline_df, 
                         num_sessions=None, plugin_names=None):
    """Write analysis summary to a text file with consistent formatting."""
    with open(file_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"Generated: {timestamp}\n")
        
        if num_sessions is not None:
            f.write(f"Number of Sessions Analyzed: {num_sessions}\n")
            
        if plugin_names:
            f.write(f"Analyzed plugins: {', '.join(plugin_names)}\n")
            
        f.write("\n")
        f.write("Component-wise Statistics:\n")
        f.write(components_df.to_string(index=False))
        f.write("\n\nPipeline Statistics:\n")
        f.write(pipeline_df.to_string(index=False))

def analyze_pipeline_latencies(input_dir, plugin_names=None):
    """
    Analyze latencies of sequential pipeline components from a single trace session.
    
    Args:
        input_dir (Path): Directory containing component latency CSVs
        plugin_names (list): Optional list of specific components to analyze
    
    Returns:
        tuple: (component_stats DataFrame, pipeline_stats DataFrame)
    """
    input_path = Path(input_dir)
    all_components = []
    
    # Process each CSV file
    for csv_file in input_path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if plugin_names is not None:
                df = df[df['Node/Component'].isin(plugin_names)]
                if df.empty:
                    print(f"Warning: No matching plugins found in {csv_file}")
                    continue
            
            component_stats = df.groupby('Node/Component').agg({
                'Mean (ms)': 'first',
                'Min (ms)': 'first',
                'Max (ms)': 'first',
                'Median (ms)': 'first',
                'Std Dev': 'first',
                'Count': 'first'
            }).reset_index()
            all_components.append(component_stats)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_components:
        error_msg = f"No valid CSV files found in {input_dir}"
        if plugin_names:
            error_msg += f" for plugins: {', '.join(plugin_names)}"
        raise ValueError(error_msg)
    
    # Calculate component and pipeline statistics
    combined_stats = pd.concat(all_components, ignore_index=True)
    final_component_stats = combined_stats.groupby('Node/Component').agg({
        'Mean (ms)': 'mean',
        'Min (ms)': 'min',
        'Max (ms)': 'max',
        'Median (ms)': 'median',
        'Std Dev': 'mean',
        'Count': 'sum'
    }).reset_index()

    # Calculate pipeline totals
    total_mean = float(final_component_stats['Mean (ms)'].sum())
    total_min = float(final_component_stats['Min (ms)'].sum())
    total_max = float(final_component_stats['Max (ms)'].sum())
    total_median = float(final_component_stats['Median (ms)'].sum())
    total_variance = np.sum(final_component_stats['Std Dev']**2)
    total_std_dev = np.sqrt(total_variance)
    total_samples = int(final_component_stats['Count'].min())
    
    # Calculate confidence intervals
    confidence_level = 0.95
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    margin_error = z_value * (total_std_dev / np.sqrt(total_samples))
    
    pipeline_stats = pd.DataFrame({
        'Node/Component': ['Pipeline Total'],
        'Mean (ms)': [total_mean],
        'Min (ms)': [total_min],
        'Max (ms)': [total_max],
        'Median (ms)': [total_median],
        'Std Dev': [total_std_dev],
        'Count': [total_samples],
        'CI Lower': [total_mean - margin_error],
        'CI Upper': [total_mean + margin_error],
        'Skewness': [abs(total_mean - total_median) / total_std_dev if total_std_dev > 0 else 0]
    })
    
    # Save results
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_readable = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plugin_suffix = "_filtered" if plugin_names else ""
    
    csv_path = input_path / f"component_stats{plugin_suffix}_{timestamp}.csv"
    final_component_stats.to_csv(csv_path, index=False)
    
    pipeline_csv_path = input_path / f"pipeline_stats{plugin_suffix}_{timestamp}.csv"
    pipeline_stats.to_csv(pipeline_csv_path, index=False)
    
    summary_path = input_path / f"analysis_summary{plugin_suffix}_{timestamp}.txt"
    write_analysis_summary_to_text_file(
        file_path=summary_path,
        title="Sequential Component Latency Analysis",
        timestamp=timestamp_readable,
        components_df=final_component_stats,
        pipeline_df=pipeline_stats,
        plugin_names=plugin_names
    )
    
    print(f"\nAnalysis results saved in {input_path}:")
    print(f"- Component statistics: {csv_path.name}")
    print(f"- Pipeline statistics: {pipeline_csv_path.name}")
    print(f"- Analysis summary: {summary_path.name}")
    
    return final_component_stats, pipeline_stats

def aggregate_session_statistics(results_dir):
    """
    Aggregate statistics across multiple trace sessions, combining within-session
    and between-session variations.
    
    Args:
        results_dir (Path): Directory containing session folders with statistics
    
    Returns:
        tuple: (component_summary DataFrame, pipeline_summary DataFrame)
    """
    results_path = Path(results_dir)
    session_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and d.name.startswith('my-tracing-session')]
    
    if not session_dirs:
        raise ValueError(f"No session directories found in {results_dir}")
    
    # Collect all session stats
    all_component_stats = []
    all_pipeline_stats = []
    
    for session_dir in session_dirs:
        print(f"Looking session_dir at {session_dir}")
        # Get latest component stats
        component_files = list(session_dir.glob('component_stats_*.csv'))
        if component_files:
            latest_component = max(component_files, key=lambda x: x.stat().st_mtime)
            comp_df = pd.read_csv(latest_component)
            comp_df['Session'] = session_dir.name
            all_component_stats.append(comp_df)
        
        # Get latest pipeline stats
        pipeline_files = list(session_dir.glob('pipeline_stats_*.csv'))
        if pipeline_files:
            latest_pipeline = max(pipeline_files, key=lambda x: x.stat().st_mtime)
            pipe_df = pd.read_csv(latest_pipeline)
            pipe_df['Session'] = session_dir.name
            all_pipeline_stats.append(pipe_df)
    
    if not all_component_stats or not all_pipeline_stats:
        raise ValueError("No statistics files found")
    
    # Process component statistics
    combined_comp_df = pd.concat(all_component_stats, ignore_index=True)
    base_stats = combined_comp_df.groupby('Node/Component').agg({
        'Mean (ms)': 'mean',
        'Min (ms)': 'min',
        'Max (ms)': 'max',
        'Median (ms)': 'mean',
        'Count': 'sum'
    })
    
    # Calculate combined std dev (within + between session variation)
    def calc_combined_std(group):
        within_var = (group['Std Dev'] ** 2).mean()
        between_var = group['Mean (ms)'].var()
        return np.sqrt(within_var + between_var)
    
    # Add combined std dev to results
    component_summary = base_stats.copy()
    component_summary['Std Dev'] = combined_comp_df.groupby('Node/Component').apply(calc_combined_std)
    component_summary = component_summary.reset_index()
    
    # Process pipeline statistics similarly
    combined_pipe_df = pd.concat(all_pipeline_stats, ignore_index=True)
    base_pipe_stats = combined_pipe_df.groupby('Node/Component').agg({
        'Mean (ms)': 'mean',
        'Min (ms)': 'min',
        'Max (ms)': 'max',
        'Median (ms)': 'mean',
        'Count': 'sum'
    })
    
    pipeline_summary = base_pipe_stats.copy()
    pipeline_summary['Std Dev'] = combined_pipe_df.groupby('Node/Component').apply(calc_combined_std)
    pipeline_summary = pipeline_summary.reset_index()
    
    # Save results
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_readable = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    component_summary_path = results_path / f"aggregated_component_stats_{timestamp}.csv"
    component_summary.to_csv(component_summary_path, index=False)
    
    pipeline_summary_path = results_path / f"aggregated_pipeline_stats_{timestamp}.csv"
    pipeline_summary.to_csv(pipeline_summary_path, index=False)
    
    summary_path = results_path / f"aggregated_analysis_summary_{timestamp}.txt"
    write_analysis_summary_to_text_file(
        file_path=summary_path,
        title="Aggregated Sequential Pipeline Statistics",
        timestamp=timestamp_readable,
        components_df=component_summary,
        pipeline_df=pipeline_summary,
        num_sessions=len(session_dirs)
    )
    
    return component_summary, pipeline_summary

def analyze_all_sessions(input_dir, output_dir=None, plugin_names=None, verbose=False):
    """
    Process all trace sessions in the input directory, analyzing pipeline latencies
    and aggregating statistics.
    
    Args:
        input_dir (str or Path): Directory containing trace session folders
        output_dir (str or Path, optional): Directory for output. If None, uses input_dir
        plugin_names (list, optional): List of components to analyze. If None, analyzes all
        verbose (bool): Whether to print detailed progress
    """
    input_path = Path(input_dir)
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path
    
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all trace session directories
    trace_sessions = [
        d for d in input_path.iterdir() 
        if d.is_dir() and d.name.startswith('my-tracing-session')
    ]

    if not trace_sessions:
        print(f"No trace sessions found in {input_dir}")
        return

    if verbose:
        print(f"Found {len(trace_sessions)} trace sessions:")
        for session in trace_sessions:
            print(f"  - {session.name}")
        print(f"\nResults will be saved in: {output_path}")

    # Process each session
    for session_num, session_dir in enumerate(sorted(trace_sessions), 1):
        session_output_dir = output_path / session_dir.name
        session_output_dir.mkdir(exist_ok=True)

        if verbose:
            print(f"\nProcessing session {session_num}/{len(trace_sessions)}: {session_dir.name}")

        try:
            analyze_pipeline_latencies(session_dir, plugin_names)
        except Exception as e:
            print(f"Error processing {session_dir.name}: {e}")
            continue

        if verbose:
            print(f"Completed analysis for {session_dir.name}")

    # Aggregate results across all sessions
    try:
        component_summary, pipeline_summary = aggregate_session_statistics(output_path)
        if verbose:
            print("\nComponent-wise Summary Statistics:")
            print(component_summary)
            print("\nPipeline Summary Statistics:")
            print(pipeline_summary)
    except Exception as e:
        print(f"Error aggregating statistics: {e}")

    print(f"\nAnalysis complete. Results saved in {output_path}")

def main():
    """
    Main function to run pipeline analysis on trace data.
    
    Usage: python pipeline_analyzer.py [-v] [-o OUTPUT_DIR] [-p PLUGINS [PLUGINS ...]] input_dir
    """
    parser = argparse.ArgumentParser(description='Analyze pipeline latencies from trace data')
    parser.add_argument('input_dir', help='Directory containing trace session data')
    parser.add_argument('-o', '--output-dir', help='Directory to save analysis results (optional)')
    parser.add_argument('-p', '--plugins', nargs='+', help='Specific plugins to analyze (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    control_pipeline_plugins = [
        "arbitrator",
        "plan_delegator",
        "trajectory_executor",
        "trajectory_follower",
        "twist_filter",
        "twist_gate",
    ]

    # Use specified plugins or default control pipeline plugins
    plugins_to_analyze = args.plugins if args.plugins else control_pipeline_plugins
    print(plugins_to_analyze)
    analyze_all_sessions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        plugin_names=plugins_to_analyze,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()