# CARMA Analytics Tools

This repository contains two main analysis scripts for the ROS2 Tracing Analysis for CARMA Platform: 
- Callback Duration Analyzer 
- Pipeline Latency Analyzer.

#### Notes
- The Pipeline Latency Analyzer processes trace analysis results from the Callback Duration Analyzer
- The Pipeline Latency Analyzer calculates sequential pipeline metrics
- The Pipeline Latency Analyzer supports both single-session and multi-session analysis


## Workspace Setup

Your workspace directory should contain the following on the same level:
- `carma-analytics-fotda/`
- `tracetools_analysis/` (foxy branch)
- `ros2_tracing/` (foxy branch)

Clone the required repositories:
```bash
git clone -b foxy https://github.com/usdot-fhwa-stol/tracetools_analysis
git clone -b foxy https://github.com/ros2/ros2_tracing
```

## Dependencies

### System Requirements
- Python 3.8+

### Required Packages
Install the following dependencies:
```bash
sudo apt-get install python3-numpy
sudo apt-get install python3-pandas
sudo apt-get install python3-babeltrace python3-lttng
```

Additional Python packages needed:
- scipy

## Script 1: Callback Duration Analyzer

### Usage

#### Basic Usage
```bash
python3 carma_analyze_callback_durations.py /path/to/trace/sessions
```

#### Advanced Usage
```bash
python3 carma_analyze_callback_durations.py [-h] [-o OUTPUT_DIR] [-v] [-sp] input_dir
```

#### Arguments
- `input_dir`: Directory containing trace sessions
- `-o, --output-dir`: Optional output directory (default: input_dir/trace_results)
- `-v, --verbose`: Enable detailed progress output
- `-sp, --show-plots`: Display plots during analysis (also saves to files)

### Outputs
For each analyzed trace session, the script creates a new results folder containing:
- CSV file with callback duration statistics for each specified componenets
- Two plots per callback:
    - Scatter plot (callback durations vs. time)
    - Histogram of callback durations

#### Example csv file
![Example Extracted Trace Data](example_extracted_trace_data.png)

#### Example callback duration plot vs time
![Example Callback Dueation Scatter Plot](example_callback_duration_scatter_plot.png)

#### Example callback duration histogram
![Example Callback Duration Histogram](example_callback_duration_histogram.png)

## Script 2: Pipeline Latency Analyzer

### Usage

#### Basic Usage
```bash
python3 carma_analyze_sequential_node_pipeline.py /path/to/trace/results
```

#### Advanced Usage
```bash
python3 carma_analyze_sequential_node_pipeline.py [-h] [-o OUTPUT_DIR] [-p PLUGINS [PLUGINS ...]] [-v] input_dir
```

#### Arguments
- `input_dir`: Directory containing trace analysis results
- `-o, --output-dir`: Optional output directory (default: input_dir/analysis_results)
- `-p, --plugins`: Space-separated list of specific plugins to analyze
- `-v, --verbose`: Enable detailed progress output

### Default Analyzed Plugins
If no plugins are specified, the following are analyzed:
- arbitrator
- plan_delegator
- trajectory_executor
- trajectory_follower
- twist_filter
- twist_gate

### Outputs

#### Per Session Outputs
1. Component statistics (CSV)
2. Pipeline statistics (CSV)
3. Analysis summary (TXT)

### Example Per Session Analysis Results
```
Sequential Component Latency Analysis
Generated: 2024-11-12 01:57:31
Analyzed plugins: arbitrator, plan_delegator, trajectory_executor, trajectory_follower, twist_filter, twist_gate

Component-wise Statistics:
      Node/Component  Mean (ms)  Min (ms)   Max (ms)  Median (ms)   Std Dev  Count
          arbitrator   0.253442  0.000441  11.056656     0.002429  0.768611   1706
      plan_delegator   0.006873  0.000734   7.285678     0.003034  0.108669   4935
 trajectory_executor   0.008941  0.000646   7.973493     0.002239  0.193791   1728
 trajectory_follower   0.031122  0.008851   6.259083     0.014623  0.170346   5456
        twist_filter   0.069140  0.009860  17.676080     0.017865  0.422927   5382
          twist_gate   0.006322  0.000694   2.780104     0.002875  0.054423   8343

Pipeline Statistics:
 Node/Component  Mean (ms)  Min (ms)   Max (ms)  Median (ms)   Std Dev  Count  CI Lower  CI Upper  Skewness
 Pipeline Total    0.37584  0.021226  53.031094     0.043064  0.922482   1706  0.332066  0.419614   0.36074
```

#### Aggregated Outputs
1. `aggregated_component_stats_{timestamp}.csv`
2. `aggregated_pipeline_stats_{timestamp}.csv`
3. `aggregated_analysis_summary_{timestamp}.txt`

#### Example Aggregated Analysis Results
```
Aggregated Sequential Pipeline Statistics
Generated: 2024-11-12 01:57:31
Number of Sessions Analyzed: 2

Component-wise Statistics:
      Node/Component  Mean (ms)  Min (ms)   Max (ms)  Median (ms)  Count   Std Dev
          arbitrator   0.128259  0.000441  11.056656     0.002317   2747  0.571648
      plan_delegator   0.007244  0.000734   7.285678     0.003052   7951  0.122905
 trajectory_executor   0.010607  0.000646   7.973493     0.002255   2791  0.221760
 trajectory_follower   0.029674  0.008851   6.259083     0.014745   8748  0.144739
        twist_filter   0.055677  0.009860  17.676080     0.017247   8604  0.343571
          twist_gate   0.006658  0.000694   2.780104     0.002900  13429  0.058989

Pipeline Statistics:
 Node/Component  Mean (ms)  Min (ms)   Max (ms)  Median (ms)  Count  Std Dev
 Pipeline Total   0.238119  0.021226  53.031094     0.042516   2747  0.73468
```

### Analysis Features
- Individual component statistics
- End-to-end pipeline performance
- Cross-session variations
- Combined standard deviations (within + between session)