# Guidance Scripts

This file contains various analysis functions for CARMA Platform data, focusing on guidance and control aspects.

## Functions

### get_engage_time

Get the (engage, disengage_time) as a tuple from /guidance/state. Returns last recorded time if no disengaged.
NOTE: If there are multiple engage operations, it will only take the first engage time as the start_time.

#### Parameters

- `mcap_path`: Path to MCAP file

#### Output

- Returns a tuple: (start_time, end_time)

### run_crosstrack_analysis

Analyzes cross-track error from CARMA Platform's internal route logic using topic /guidance/route_state

#### Parameters

- `mcap_path`: Path to MCAP file
- `error_threshold_to_pass_meter`: Threshold for passing the analysis (default: 2.0 meters)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

- Returns a tuple: (is_passed, stats, plot_figure, cross_tracks, timestamps)
- Saves statistics as JSON, data as NPZ, and plot as PNG (if directories are provided)

#### Example Plot

![Cross Track Error Over Time](cross_track_error_over_time.png)

This plot shows the cross-track error over time, including the median and standard deviation.

### run_turn_accuracy_analysis

Analyzes turn accuracy by comparing actual path to planned trajectory using spline fitting over time.

#### Parameters

- `mcap_path`: Path to MCAP file
- `error_threshold_to_pass_meter`: Threshold for passing the analysis (default: 2.0 meters)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

- Returns a tuple: (is_passed, stats, plot_figure, distances, timestamps)
- Saves statistics as JSON, data as NPZ, and plot as PNG (if directories are provided)

#### Example Plot

![Turn Accuracy Analysis](turn_accuracy_analysis.png)

### run_acceleration_comfort_analysis

Analyzes acceleration comfort based on vehicle status data.

#### Parameters

- `mcap_path`: Path to MCAP file
- `comfort_deceleration_threshold_to_pass`: Threshold for comfort analysis (default: 3.0 m/s²)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

- Returns a tuple: (is_passed, instant_stats, avg_stats, plot_figure, accelerations, avg_accelerations, time_points, avg_time_points)
- Saves instantaneous and 1-second average statistics as JSON files
- Saves data as NPZ and plots as PNG (if directories are provided)

#### Example Plot

![Acceleration Comfort Analysis](acceleration_comfort_analysis.png)

## Adding New Analysis Functions

To add a new analysis function:

1. Define the function in this file.
2. Ensure it follows a similar structure to existing functions (e.g., `run_crosstrack_analysis`).
3. Update this README to include documentation for the new function.
4. Integrate the new function into the main analysis pipeline for specific use case(`run_all_control_analysis.py` or others).
5. If the function is going to be reused a lot, please add unit test to the test folder. One can run it by `python3 -m pytest test` in carma-platform folder.
