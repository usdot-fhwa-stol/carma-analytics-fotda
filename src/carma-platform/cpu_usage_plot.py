import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os
from guidance_scripts import *

DEFAULT_CPU_NUM = 8 # 32 for SIM PC, 8 for Spectra PCs
def get_notable_events_from_mcap(mcap_path):
    # get earliest topic timestamp as global_start_time
    reader, type_map, global_start_time = open_bagfile(str(mcap_path))

    (start, end) = get_engage_time(mcap_path)



    return []


def validate_file(file_path):
    """Validate if the file exists and has .csv extension"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    if not file_path.lower().endswith('.csv'):
        raise ValueError("File must be a CSV file")
    return file_path

def plot_cpu_usage(csv_file, notable_event_stamps = [], cpu_number=DEFAULT_CPU_NUM, output_file=None):
    """
    Generate CPU usage plot from CSV file with timestamp grouping

    Args:
        csv_file (str): Path to input CSV file
        output_file (str, optional): Path for output PNG file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Verify required columns exist
        required_columns = ['Timestamp', 'CPU (%)', 'Total CPU (%)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Group by timestamp and calculate metrics
        grouped_df = df.groupby('Timestamp').agg({
            'CPU (%)': 'sum',  # Sum all CPU percentages
            'Total CPU (%)': 'first'  # Take first Total CPU value as it should be same for timestamp
        }).reset_index()

        # Divide the summed CPU by 6 (number of cores)
        grouped_df['CPU (%)'] = grouped_df['CPU (%)'] / cpu_number

        grouped_df.to_csv('output.csv', index=True)

        # Create a figure with a larger size
        plt.figure(figsize=(15, 8))

        # Plot grouped CPU usage
        plt.plot(grouped_df['Timestamp'], grouped_df['CPU (%)'],
                label='Average CPU per Core (%)',
                color='blue',
                linewidth=2)

        # Plot total CPU usage
        plt.plot(grouped_df['Timestamp'], grouped_df['Total CPU (%)'],
                label='Total CPU (%)',
                color='red',
                linewidth=2)

        # Customize the plot
        plt.title(f'CPU Usage of CARMA Sampled Every 1 Sec over {cpu_number} CPUs')
        plt.xlabel('Time')
        plt.ylabel('CPU Usage Percentage (0-100%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Set y-axis limits
        plt.ylim(0, max(grouped_df['Total CPU (%)'].max(), grouped_df['CPU (%)'].max()) * 1.1)

        # Generate output filename if not provided
        if output_file is None:
            output_file = os.path.splitext(csv_file)[0] + '_cpu_usage.png'

        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Graph has been generated successfully as '{output_file}'")

    except Exception as e:
        print(f"Error generating plot: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate CPU usage graph from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('-m', '--mcap', type=str, help='Path to the corresponding ROS2 MCAP file (optional)')
    parser.add_argument('-c', '--cpu-num', type=int, help='Number of CPUs used to record the data (optional)', default=DEFAULT_CPU_NUM)
    parser.add_argument('-o', '--output', type=str, help='Path for the output PNG file (optional)')

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate input file
        csv_file = validate_file(args.csv_file)

        # List of notable events in the mcap file such as engage, disengage, vector_map broadcasting etc
        notable_events = []
        if (args.mcap is not None):
            notable_events = get_notable_events_from_mcap(args.mcap)

        # Generate the plot
        plot_cpu_usage(csv_file, notable_events, args.cpu_num, args.output)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
