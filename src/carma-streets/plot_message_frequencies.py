import argparse
from pathlib import Path
import sys
from parse_kafka_logs import KafkaLogMessageType

from matplotlib import pyplot as plt
from matplotlib import axes
import pandas as pd

def plot_message_frequencies(csv_dir: str, plots_dir: str, simulation: bool = False):
    """Function to plot all message frequencies in a single figure

    Args:
        csv_dir (str): directory that holds CSV message data.
        plots_dir (str): directory where create figure will be saved
        simulation (bool,optional): bool flag to indicate whether data was collected in simulation. Defaults to False.
    """
    csv_dir_path = Path(csv_dir)
    plots_dir_path = Path(plots_dir)
    if csv_dir_path.is_dir():
        if plots_dir_path.is_dir():
            print(f"WARNING: {plots_dir} already exists. Contents will be overwritten.")
        plots_dir_path.mkdir(exist_ok=True)
        # Read CSV data
        message_data = dict([])
        time_sync_data = None
        for csv_file in csv_dir_path.glob("*.csv"):
            print(f'Reading csv file {csv_file.name} ...')
            if csv_file.name == 'time_sync.csv':
                time_sync_data = pd.read_csv(csv_file)
            else:
                df = pd.read_csv(csv_file)
                message_data[csv_file.name.split('.')[0]] = df
        fig, plots = plt.subplots(len(message_data), sharex=True, layout="constrained")
        # Add simulation time to message data
        for idx,( message_name, message_data_frame) in enumerate(message_data.items()):
            print(f'Getting simulation time for {message_name} data ...')
            if simulation:
                message_data_frame["Time (ms)"] = get_simulation_time(message_data_frame["Created Time(ms)"], time_sync_data["Created Time(ms)"], time_sync_data["Timestamp(ms)"])
                message_data_frame["Time (s)"] = message_data_frame["Time (ms)"]/1000
            else :
                message_data_frame["Time (s)"] = message_data_frame["Timestamp(ms)"]/1000
            message_data_frame = get_message_frequency(message_data_frame)
            if KafkaLogMessageType.MAP.value in message_name:
                #
                plot_message_frequency(plots[idx],message_data_frame['Time (s)'], message_data_frame["Average Frequency (Hz)"] ,message_name,1, 1)
            else:
                # Any message with 10 Hz as target frequency
                plot_message_frequency(plots[idx],message_data_frame['Time (s)'], message_data_frame["Average Frequency (Hz)"],message_name)
            fig.suptitle('Message Frequency Plots')
            fig.supxlabel('Time (s)')
            fig.supylabel('Message Frequency (Hz)')
            fig.savefig(f'{plots_dir}/message_frequencies.png')
        
def plot_message_frequency( axes: axes.Axes, time: list, frequency: list , message_name: str, target_frequency: int = 10, absolute_error: int = 2) :
    """Generate subplots for each message frequency

    Args:
        axes (axes.Axes): Subplot of message frequency
        time (list): Time data (s)
        frequency (list): Frequency data (Hz)
        message_name (str): Message Name
        target_frequency (int, optional): Target frequency. Defaults to 10.
        absolute_error (int, optional): Acceptable deviation from target frequency. Defaults to 2.
    """
    axes.scatter(time,frequency, c="blue", marker="^", label="Freq (hz)")
    # Add horizontal lines for differing freq requirements based on message type
    axes.axhline(y=target_frequency-absolute_error, color='r', linestyle='--', label="frequency lower bound")
    axes.axhline(y=target_frequency+absolute_error, color='r', linestyle='-', label="frequency upper bound")
    axes.set_title(message_name)
    axes.set_ylim(target_frequency-2*absolute_error, target_frequency+2*absolute_error)

def get_simulation_time(message_wall_time : list, time_sync_wall_time: list, time_sync_simulation_time : list)-> list:
    message_simulation_time=list()
    for msg_wall_time in message_wall_time:
        for idx, t_sync_wall_time in enumerate(time_sync_wall_time):
            if idx == 0:
                continue
            elif msg_wall_time >= time_sync_wall_time[idx-1] and msg_wall_time < time_sync_wall_time[idx]:
                message_simulation_time.append(time_sync_simulation_time[idx-1])
                break
            elif msg_wall_time >= time_sync_wall_time[idx] and idx == len(time_sync_wall_time)-1:
                message_simulation_time.append(time_sync_simulation_time[idx])
                break
    return message_simulation_time
def get_message_frequency( messages: pd.DataFrame) -> pd.DataFrame:
    messages["Instantaneous Frequency (Hz)"] = 1/messages["Time (s)"].diff() 
    messages["Average Frequency (Hz)"] = messages["Instantaneous Frequency (Hz)"].rolling(window=50, min_periods=1).mean()
    return messages



def main():
    parser = argparse.ArgumentParser(description='Script to plot message frequency from CARMA Streets message csv data.')
    parser.add_argument('--csv-dir', help='Directory to read csv data from.', type=str, required=True)  
    parser.add_argument('--plots-dir', help='Directory to save generated plots.', type=str, required=True) 
    parser.add_argument('--simulation', help='Flag indicating data is from simulation', action='store_true', default=False)

    args = parser.parse_args()
    plot_message_frequencies(args.csv_dir, args.plots_dir, args.simulation)

if __name__ == '__main__':
    main()