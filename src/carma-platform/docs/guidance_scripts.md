# Guidance Scripts

This file contains various analysis functions for CARMA Platform data, focusing on guidance and control aspects.

## Functions

### get_engage_time

Get the (engage, disengage_time) as a tuple from `/guidance/state`. Returns last recorded time if no disengaged.
NOTE: If there are multiple engage operations, it will only take the first engage time as the start_time.

#### Parameters

- `mcap_path`: Path to MCAP file

#### Output

- Returns a tuple: (start_time, end_time)

### run_crosstrack_analysis

Analyzes cross-track error from CARMA Platform's internal route logic using topic `/guidance/route_state`

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
-`/localization/current_pose`: Source of actual traveled trajectory
-`/guidance/plan_trajectory`: Source of planned trajectory

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

Analyzes acceleration comfort based on vehicle status data using topic `/hardware_interface/vehicle_status` for the vehicle's speed

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

### run_lateral_analysis

Analyzes lateral acceleration and jerk from vehicle state data using both instantaneous and 1-second window averages. Uses twist data from vehicle interface (topic `/hardware_interface/vehicle/twist`) to calculate lateral dynamics.

#### Parameters

- `mcap_path`: Path to MCAP file
- `acc_threshold_to_pass`: Maximum acceptable lateral acceleration (default: 2.0 m/s²)
- `jerk_threshold_to_pass`: Maximum acceptable lateral jerk (default: 2.0 m/s³)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

Returns a tuple containing:

- `is_passed`: Boolean indicating if all comfort thresholds were met
- `acc_inst_stats`: Statistics for instantaneous acceleration
- `acc_avg_stats`: Statistics for 1-second average acceleration
- `jerk_inst_stats`: Statistics for instantaneous jerk
- `jerk_avg_stats`: Statistics for 1-second average jerk
- `figures`: Tuple of (acceleration_figure, jerk_figure)
- `lateral_acc_inst`: Array of instantaneous lateral acceleration values
- `lateral_acc_avg`: Array of averaged lateral acceleration values
- `lateral_jerk_inst`: Array of instantaneous lateral jerk values
- `lateral_jerk_avg`: Array of averaged lateral jerk values
- `timestamps`: Array of corresponding timestamps

#### Saved Outputs

If directories are provided:

- Statistics saved as JSON in "lateral_analysis_stats.json"
- Data saved as NPZ in "lateral_analysis_data.npz"
- Plots saved as:
  - "lateral_acceleration_analysis.png"
  - "lateral_jerk_analysis.png"

#### Example Plots

![Lateral Acceleration Analysis](lateral_acceleration_analysis.png)
Shows instantaneous and 1-second average lateral acceleration over time.

![Lateral Jerk Analysis](lateral_jerk_analysis.png)
Shows instantaneous and 1-second average lateral jerk over time.

#### Notes

- Uses vehicle's longitudinal velocity and angular velocity to calculate lateral dynamics
- Comfort thresholds are applied to both instantaneous and averaged values
- Analysis passes only if no comfort threshold violations occur in any metric
- Time-weighted averaging is used for the 1-second window calculations


### run_guidance_steering_analysis

Analyzes steering performance by comparing commanded vs actual steering angles at guidance level.
Time series alignment is performed to match commanded and actual values
Both instantaneous and statistical measures are considered
- `/guidance/ctrl_cmd`: Source of commanded steering angles
- `/hardware_interface/vehicle_status`: Source of actual vehicle steering angles

#### Parameters

- `mcap_path`: Path to MCAP file
- `error_threshold_to_pass_radian`: Threshold for passing the analysis (default: 0.1 radians)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

- Returns a tuple: (is_passed, stats, plot_figure, error_angles, common_timestamps)
- Saves statistics as JSON, data as NPZ, and plot as PNG (if directories are provided)

#### Example Plot

![Guidance Steering Analysis](guidance_steering_analysis.png)

This plot shows:
- Top panel: Comparison of commanded vs actual steering angles over time
- Bottom panel: Steering error over time, including the median and standard deviation

### run_steering_wheel_analysis

Analyzes steering performance by comparing commanded vs actual steering values at PACMod level.
Both instantaneous and statistical measures are considered
- `/hardware_interface/as/pacmod/parsed_tx/steer_rpt`: Source of actual and commanded steering wheel values

#### Parameters

- `mcap_path`: Path to MCAP file
- `error_threshold_to_pass`: Threshold for passing the analysis (default: 0.1)
- `start_time`: Time to start the analysis (optional)
- `end_time`: Time to end the analysis (optional)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output

- Returns a tuple: (is_passed, stats, plot_figure, error_values, timestamps)
- Saves statistics as JSON, data as NPZ, and plot as PNG (if directories are provided)

#### Example Plot

![Steering Wheel Analysis](steering_wheel_analysis.png)

This plot shows:
- Top panel: Comparison of commanded vs actual steering wheel values over time
- Bottom panel: Steering error over time, including the median and standard deviation

#### Analysis Metrics

Both functions calculate the following statistics:
- Minimum error
- Maximum error
- Mean error
- Median error
- Standard deviation
- Root mean square error (RMSE)
- Error percentiles (25th, 75th, 95th)

#### Saved Outputs

If directories are provided:

- Statistics saved as JSON in "guidance_steering_analysis.json" and  "steering_wheel_analysis.json"
- Data saved as NPZ in "guidance_steering_data.npz" and "steering_wheel_data.npz"
- Plots saved as:
  - "guidance_steering_analysis.png"
  - "steering_wheel_analysis.png"

### get_planner_trajectory_intervals
Extract time intervals when a specific planner was active based on trajectory plans.
Uses topic `/guidance/plan_trajectory`

#### Parameters

- mcap_path: Path to MCAP file
- planner_plugin_name: Name of the planner plugin to track (e.g. `guidance/plugins/inlanecruising_plugin`)
- start_time: Optional start time to begin analysis
- end_time: Optional end time to end analysis

#### Output

Returns a list of tuples `[(start_time1, end_time1), (start_time2, end_time2), ...]` representing time intervals when the specified planner was active


## Adding New Analysis Functions

To add a new analysis function:

1. Define the function in this file.
2. Ensure it follows a similar structure to existing functions (e.g., `run_crosstrack_analysis`).
3. Update this README to include documentation for the new function.
4. Integrate the new function into the main analysis pipeline for specific use case(`run_all_control_analysis.py` or others).
5. If the function is going to be reused a lot, please add unit test to the test folder. One can run it by `python3 -m pytest test` in carma-platform folder.
