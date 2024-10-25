# Guidance Scripts

This file contains various analysis functions for CARMA Platform data, focusing on guidance and control aspects.

## Functions

### run_crosstrack_analysis

Analyzes cross-track error from CARMA Platform's internal route logic using topic /guidance/route_state

#### Parameters:
- `mcap_path`: Path to MCAP file
- `error_threshold_to_pass_meter`: Threshold for passing the analysis (default: 2.0 meters)
- `save_stats_dir`: Directory to save extracted statistics (optional)
- `save_data_dir`: Directory to save extracted data (optional)
- `save_plot_dir`: Directory to save generated plots (optional)

#### Output:
- Returns a tuple: (is_passed, plot_figure, cross_tracks, timestamps)
- Saves statistics as JSON, data as NPZ, and plot as PNG (if directories are provided)

#### Example Plot:
![Cross Track Error Over Time](cross_track_error_over_time.png)

This plot shows the cross-track error over time, including the median and standard deviation.


## Adding New Analysis Functions

To add a new analysis function:

1. Define the function in this file.
2. Ensure it follows a similar structure to existing functions (e.g., `run_crosstrack_analysis`).
3. Update this README to include documentation for the new function.
4. Integrate the new function into the main analysis pipeline for specific use case(`run_all_control_analysis.py` or others).
