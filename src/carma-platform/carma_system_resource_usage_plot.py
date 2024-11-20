import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os
from guidance_scripts import *

ENVIRONMENT_MAP_UPDATE_TOPIC = "/environment/map_update"
ENVIRONMENT_SEMANTIC_MAP_TOPIC = "/environment/semantic_map"
DEFAULT_CPU_NUM = 8  # 32 for SIM PC, 8 for Spectra PCs in vehicles


def calculate_time_offset(mcap_path, csv_file):
    """
    Calculate the time offset between MCAP and CSV data by comparing their hours

    Args:
        mcap_path (str): Path to the MCAP file
        csv_file (str): Path to the CSV file

    Returns:
        timedelta: Time offset between MCAP and CSV data
    """
    # Get MCAP start time
    reader, type_map, global_start_time = open_bagfile(str(mcap_path))
    mcap_start_time = pd.to_datetime(global_start_time)

    # Get CSV start time
    df = pd.read_csv(csv_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    csv_start_time = df["Timestamp"].min()

    # Calculate hour difference
    hour_diff = mcap_start_time.hour - csv_start_time.hour
    time_offset = pd.Timedelta(hours=hour_diff)

    print(f"MCAP start time: {mcap_start_time}")
    print(f"CSV start time: {csv_start_time}")
    print(f"Hour difference: {hour_diff}")
    print(f"Time offset: {time_offset}")

    return time_offset


def get_notable_events_from_mcap(mcap_path):

    events = []

    # get earliest topic timestamp as global_start_time
    reader, type_map, global_start_time = open_bagfile(str(mcap_path))

    start_date_time = pd.to_datetime(global_start_time)

    (start, end) = get_engage_time(mcap_path)

    events.append(("CARMA Engaged", start_date_time + pd.Timedelta(seconds=(start))))
    events.append(("CARMA Disengaged", start_date_time + pd.Timedelta(seconds=(end))))

    # get semantic map publication timestamp
    # get map update timestamp
    topics = [ENVIRONMENT_SEMANTIC_MAP_TOPIC, ENVIRONMENT_MAP_UPDATE_TOPIC]

    extracted_data = extract_mcap_data(mcap_path, topics)

    semantic_map_timestamps, _ = extracted_data[topics[0]]

    map_update_timestamps, _ = extracted_data[topics[1]]

    for time in semantic_map_timestamps:
        events.append(
            ("Semantic Map Published", start_date_time + pd.Timedelta(seconds=(time)))
        )

    for time in map_update_timestamps:
        events.append(
            ("Map Update Published", start_date_time + pd.Timedelta(seconds=(time)))
        )

    return events


def validate_file(file_path):
    """Validate if the file exists and has .csv extension"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    if not file_path.lower().endswith(".csv"):
        raise ValueError("File must be a CSV file")
    return file_path


def plot_system_usage(
    csv_file, notable_event_stamps=[], cpu_number=DEFAULT_CPU_NUM, output_file=None
):
    """
    Generate CPU and Memory usage plot from CSV file with timestamp grouping

    Args:
        csv_file (str): Path to input CSV file
        notable_event_stamps (list): List of tuples containing (event_name, timestamp)
        cpu_number (int): Number of CPUs
        output_file (str, optional): Path for output PNG file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Verify required columns exist
        required_columns = ["Timestamp", "CPU (%)", "Memory (%)", "Total CPU (%)"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Filter out the monitoring script
        df = df[
            ~df["Command Line"].str.contains("monitor-ros-cpu.py", case=False, na=False)
        ]

        # Convert timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Group by timestamp and calculate metrics
        grouped_df = (
            df.groupby("Timestamp")
            .agg(
                {
                    "CPU (%)": "sum",  # Sum all CPU percentages
                    "Memory (%)": "sum",  # Sum all Memory percentages
                    "Total CPU (%)": "first",  # Take first Total CPU value as it should be same for timestamp
                }
            )
            .reset_index()
        )

        # Divide the summed CPU by number of cores
        grouped_df["CPU (%)"] = grouped_df["CPU (%)"] / cpu_number

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Convert timestamps to numpy arrays for plotting
        timestamps = grouped_df["Timestamp"].to_numpy()
        cpu_values = grouped_df["CPU (%)"].to_numpy()
        total_cpu_values = grouped_df["Total CPU (%)"].to_numpy()
        memory_values = grouped_df["Memory (%)"].to_numpy()

        # Plot CPU usage (top subplot)
        ax1.plot(
            timestamps,
            cpu_values,
            label="Average CPU per Core (%)",
            color="blue",
            linewidth=2,
        )
        ax1.plot(
            timestamps,
            total_cpu_values,
            label="Total CPU (%)",
            color="red",
            linewidth=2,
        )

        # Plot Memory usage (bottom subplot)
        ax2.plot(
            timestamps,
            memory_values,
            label="Total Memory Usage (%)",
            color="green",
            linewidth=2,
        )

        # Add notable events if any
        if notable_event_stamps:
            cpu_max = max(total_cpu_values.max(), cpu_values.max()) * 1.1
            mem_max = memory_values.max() * 1.1

            for event_name, event_time in notable_event_stamps:
                if event_time >= timestamps[0] and event_time <= timestamps[-1]:
                    # Add event lines to both plots
                    ax1.axvline(x=event_time, color="gray", linestyle="--", alpha=0.5)
                    ax2.axvline(x=event_time, color="gray", linestyle="--", alpha=0.5)

                    # Add labels to both plots
                    ax1.text(
                        event_time,
                        cpu_max * 0.95,
                        event_name,
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="right",
                    )
                    ax2.text(
                        event_time,
                        mem_max * 0.95,
                        event_name,
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="right",
                    )

        # Customize CPU plot
        ax1.set_title(f"CPU Usage of CARMA Sampled Every 1 Sec over {cpu_number} CPUs")
        ax1.set_ylabel("CPU Usage Percentage (0-100%)")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend()
        ax1.set_ylim(0, max(total_cpu_values.max(), cpu_values.max()) * 1.1)

        # Customize Memory plot
        ax2.set_title("Memory Usage of CARMA")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Memory Usage Percentage (0-100%)")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend()
        ax2.set_ylim(0, memory_values.max() * 1.1)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Generate output filename if not provided
        if output_file is None:
            output_file = os.path.splitext(csv_file)[0] + "_system_usage.png"

        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Graph has been generated successfully as '{output_file}'")

    except Exception as e:
        print(f"Error generating plot: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate CPU and Memory usage graph from CSV data."
    )
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "-m",
        "--mcap",
        type=str,
        help="Path to the corresponding ROS2 MCAP file (optional)",
    )
    parser.add_argument(mv 
        "-c",
        "--cpu-num",
        type=int,
        help="Number of CPUs used to record the data (optional)",
        default=DEFAULT_CPU_NUM,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path for the output PNG file (optional)"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate input file
        csv_file = validate_file(args.csv_file)

        # List of notable events in the mcap file such as engage, disengage, vector_map broadcasting etc
        notable_events = []
        if args.mcap is not None:
            # Calculate time offset
            time_offset = calculate_time_offset(args.mcap, csv_file)

            # Get events and adjust their timestamps
            events = get_notable_events_from_mcap(args.mcap)
            notable_events = [
                (name, timestamp - time_offset) for name, timestamp in events
            ]

        # Generate the plot
        plot_system_usage(csv_file, notable_events, args.cpu_num, args.output)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
