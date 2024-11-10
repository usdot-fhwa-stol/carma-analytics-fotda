# CARMA Analytics - FOTDA CARMA Platform Related Scripts

This package contains scripts for analyzing MCAP (ROS2 bag) files from CARMA Platform field tests. It provides tools for extracting data, running various analyses, and generating reports.

## Files Overview

1. `parse_ros2_bags.py`: Contains functions for reading and extracting data from MCAP files.
2. `run_all_analysis.py`: Provides a framework for running multiple analyses on a set of MCAP files.
3. `run_all_control_analysis.py`: Implements specific control-related analyses using the framework from `run_all_analysis.py`.
4. `guidance_scripts.py`: Contains individual analysis functions, such as cross-track error analysis.
5. `parse_ros_bags.py`: Contains previous ROS1 functionality for data analysis. It maybe deprecated in the future.
## How to Use

### Prerequisites

- Python 3.6+
- Required Python packages: rosbag2_py, numpy, matplotlib, scipy, argcomplete, pytest (for unit test)
- Required ROS2 messages to be built and sourced

Install the required packages:

```
bash
sudo apt install ros-humble-rosbag2*
pip install numpy matplotlib scipy argcomplete mcap-ros2-support pytest
```

### Running Analyses
Individual use case specific analysis functions should be created in the same manner as `run_all_control_analysis.py` such as `run_all_<use-case>_analysis`.
This guide is written using the control related scripts as an example.
1. To run all control analyses on a set of MCAP files:

```
bash
python run_all_control_analysis.py --input-dir /path/to/mcap/files --output-dir /path/for/results
```

This will:
- Find all MCAP files in the input directory
- Run the specified use case analyses
- Save results, plots, and statistics in the output directory

2. To add new analyses:
   - Implement the analysis function in `guidance_scripts.py` or new functions from different subsystem in a separate file
   - Add the new analysis to the `analyze_mcap_file_for_control_analysis` function in `run_all_control_analysis.py`

### Customizing Analyses

- Modify thresholds and parameters in `run_all_control_analysis.py`
- Add new analysis functions to `guidance_scripts.py`

Similar to how control related scripts were made, the user can also create their own use case specific analysis consisting of multiple performance metric analyses.
Then reuse the `run_all_analysis.py` to automatically detect MCAP files and generate results.

## Output

The script generates:
- A summary JSON file with results for all analyzed files
- Individual directories for each MCAP file containing:
  - Plots (PNG files)
  - Extracted data (NPZ files)
  - Statistics (JSON files)

### Example output folder structure
```
analysis_20241025_065250
├── rosbag2_2024_10_08_20_00_45_0
│ ├── data
│ ├── plots
│ └── stats
├── rosbag2_2024-10-22_213643_0
│ ├── data
│ │ └── extracted_numpy_data.npz
│ │        ...
│ ├── plots
│ │ └── cross_track_error_over_time.png
│ │        ...
│ └── stats
│   └── cross_track_stats_result.json
│          ...
└── analysis_summary.json
```
### Example Analysis Summary (JSON)
```
{
  "analysis_time": "2024-11-10T17:49:13.015897",
  "analysis_type": "analysis",
  "total_files_analyzed": 6,
  "metrics_summary": {
    "run_crosstrack_analysis": {
      "total_files": 6,
      "passed": 2,
      "failed": 0,
      "errors": 0,
      "pass_rate": "33.33%",
      "error_rate": "0.00%"
    },
    "run_turn_accuracy_analysis": {
      "total_files": 6,
      "passed": 2,
      "failed": 0,
      "errors": 0,
      "pass_rate": "33.33%",
      "error_rate": "0.00%"
    },
    "run_acceleration_comfort_analysis": {
      "total_files": 6,
      "passed": 0,
      "failed": 0,
      "errors": 2,
      "pass_rate": "0.00%",
      "error_rate": "33.33%"
    },
    "run_lateral_analysis": {
      "total_files": 6,
      "passed": 2,
      "failed": 0,
      "errors": 0,
      "pass_rate": "33.33%",
      "error_rate": "0.00%"
    },
    "run_guidance_steering_analysis": {
      "total_files": 6,
      "passed": 0,
      "failed": 0,
      "errors": 2,
      "pass_rate": "0.00%",
      "error_rate": "33.33%"
    },
    "run_steering_wheel_analysis": {
      "total_files": 6,
      "passed": 2,
      "failed": 0,
      "errors": 0,
      "pass_rate": "33.33%",
      "error_rate": "0.00%"
    }
  },
  "analyzed_files": {
    "bag1_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/bag1_0",
      "metrics_results": []
    },
    "ros2_bag_mish3_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/ros2_bag_mish3_0",
      "metrics_results": []
    },
    "ros2_bag_mish4_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/ros2_bag_mish4_0",
      "metrics_results": [
        {
          "run_crosstrack_analysis": true,
          "run_turn_accuracy_analysis": true,
          "run_acceleration_comfort_analysis": null,
          "run_lateral_analysis": true,
          "run_guidance_steering_analysis": null,
          "run_steering_wheel_analysis": true
        },
        {
          "run_crosstrack_analysis": true,
          "run_turn_accuracy_analysis": true,
          "run_acceleration_comfort_analysis": null,
          "run_lateral_analysis": true,
          "run_guidance_steering_analysis": null,
          "run_steering_wheel_analysis": true
        }
      ]
    },
    "rosbag2-full_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/rosbag2-full_0",
      "metrics_results": []
    },
    "rosbag2_2024-10-22_213643_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/rosbag2_2024-10-22_213643_0",
      "metrics_results": []
    },
    "rosbag2_2024_10_08-20_00_45_0.mcap": {
      "output_dir": "/workspaces/carma/src/analysis-data/analysis_20241110_174819/rosbag2_2024_10_08-20_00_45_0",
      "metrics_results": []
    }
  }
}
```
