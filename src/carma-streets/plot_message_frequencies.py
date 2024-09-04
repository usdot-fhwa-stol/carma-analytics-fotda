#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from matplotlib import axes
from matplotlib import pyplot as plt
from parse_kafka_logs import KafkaLogMessageType


def plot_message_frequencies(
    csv_dir: Path, plots_dir: Path, simulation: bool = False, message_type="all"
):
    """Function to plot all message frequencies in a single figure

    Args:
        csv_dir (Path): directory that holds CSV message data.
        plots_dir (Path): directory where create figure will be saved
        simulation (bool,optional): bool flag to indicate whether data was collected in simulation. Defaults to False.
    """
    if csv_dir.is_dir():
        if plots_dir.is_dir():
            print(f"WARNING: {plots_dir} already exists. Contents will be overwritten.")
        plots_dir.mkdir(exist_ok=True)
        # Read CSV data
        message_data = dict([])
        time_sync_data = None
        is_all_message_types = message_type == "all"
        file_pattern = "*.csv" if is_all_message_types else str(message_type) + ".csv"
        for csv_file in csv_dir.glob(file_pattern):
            print(f"Reading csv file {csv_file.name} ...")
            if KafkaLogMessageType.TimeSync.value in csv_file.name:
                time_sync_data = pd.read_csv(csv_file)
            elif KafkaLogMessageType.DetectedObject.value in csv_file.name:
                # Since Detected Object messages are sent for each individual object they
                # do not have a static target frequency
                continue
            else:
                df = pd.read_csv(csv_file)
                message_data[csv_file.stem] = df

        fig, plots = plt.subplots(
            len(message_data),
            sharex=True,
            layout="constrained",
            figsize=[15, 2.5 * len(message_data)],
        )
        # If it is all message types, the plots is assigned to an array of axes.
        # If there is only one message type, the plots is assigned to an array of one axis.
        plots = plots if is_all_message_types else [plots]
        # Add simulation time to message data
        # Zip call requires each parameter (including plots and message_data items) to be iterable.
        for msg_plot, (message_name, message_data_frame) in zip(
            plots, message_data.items()
        ):
            print(f"Getting simulation time for {message_name} data ...")
            if simulation:
                message_data_frame["Time (ms)"] = get_simulation_time(
                    message_data_frame["System Time (ms)"],
                    time_sync_data["System Time (ms)"],
                    time_sync_data["Message Time (ms)"],
                )
                message_data_frame["Time (s)"] = message_data_frame["Time (ms)"] / 1000
            else:
                message_data_frame["Time (s)"] = (
                    message_data_frame["System Time (ms)"] / 1000
                )
            if KafkaLogMessageType.MAP.value in message_name:
                # Any message with 1 Hz as target frequency using window size 3 (3 s) for rolling average
                add_message_frequency_columns(message_data_frame, 3)
                plot_message_frequency(
                    msg_plot,
                    message_data_frame["Time (s)"],
                    message_data_frame["Average Frequency (Hz)"],
                    message_name,
                    1.0,
                    0.2,
                )
            else:
                # Any message with 10 Hz as target frequency uses window size 30 (3 s) for rolling average
                add_message_frequency_columns(message_data_frame)
                plot_message_frequency(
                    msg_plot,
                    message_data_frame["Time (s)"],
                    message_data_frame["Average Frequency (Hz)"],
                    message_name,
                )
        fig.suptitle("Message Frequency Plots", fontsize=20)
        fig.supxlabel("Time (s)", fontsize=16)
        fig.supylabel("Message Frequency (Hz)", fontsize=16)
        handles, labels = plots[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=16)
        fig.savefig(f"{plots_dir}/message_frequencies.png")


def plot_message_frequency(
    axes: axes.Axes,
    time: list,
    frequency: list,
    message_name: str,
    target_frequency: float = 10.0,
    absolute_error: float = 2.0,
):
    """Generate subplots for each message frequency

    Args:
        axes (axes.Axes): Subplot of message frequency
        time (list): Time data (s)
        frequency (list): Frequency data (Hz)
        message_name (str): Message Name
        target_frequency (int, optional): Target frequency. Defaults to 10.
        absolute_error (int, optional): Acceptable deviation from target frequency. Defaults to 2.
    """
    axes.scatter(time, frequency, c="blue", marker="^", label="Freq (hz)")
    # Add horizontal lines for differing freq requirements based on message type
    axes.axhline(
        y=target_frequency - absolute_error,
        color="r",
        linestyle="--",
        label="frequency lower bound",
    )
    axes.axhline(
        y=target_frequency + absolute_error,
        color="r",
        linestyle="-",
        label="frequency upper bound",
    )
    axes.set_title(message_name, fontsize=18)
    axes.set_ylim(
        target_frequency - 2 * absolute_error, target_frequency + 2 * absolute_error
    )
    axes.minorticks_on()
    axes.grid(which="major", axis="both")


def get_simulation_time(
    message_wall_time: list, time_sync_wall_time: list, time_sync_simulation_time: list
) -> list:
    """Returns a list of simulation times for the provided message wall times.

    Args:
        message_wall_time (list): List of wall timestamp when a given message was sent
        time_sync_wall_time (list): List of wall timestamps when simulation time was update
        time_sync_simulation_time (list): List of simulation time values.

    Returns:
        list: List of simulation times for provided message set.
    """
    message_simulation_time = list()
    for msg_wall_time in message_wall_time:
        for idx, t_sync_wall_time in enumerate(time_sync_wall_time):
            if idx == 0:
                continue
            elif (
                msg_wall_time >= time_sync_wall_time[idx - 1]
                and msg_wall_time < time_sync_wall_time[idx]
            ):
                message_simulation_time.append(time_sync_simulation_time[idx - 1])
                break
            elif (
                msg_wall_time >= time_sync_wall_time[idx]
                and idx == len(time_sync_wall_time) - 1
            ):
                message_simulation_time.append(time_sync_simulation_time[idx])
                break
    return message_simulation_time


def add_message_frequency_columns(
    messages: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Add columns for instantaneous and average (rolling 5 second) frequency for given message data

    Args:
        messages (pd.DataFrame): Message Data
        window (int): window size (number of messages) for rolling average. Default is 30.

    Returns:
        pd.DataFrame: Message data with columns for instantaneous and average frequency.
    """
    messages["Instantaneous Frequency (Hz)"] = 1 / messages["Time (s)"].diff()
    messages["Interval (s)"] = messages["Time (s)"].diff()
    messages["Average Interval (s)"] = (
        messages["Interval (s)"].rolling(window=window).mean()
    )
    messages["Average Frequency (Hz)"] = 1 / messages["Average Interval (s)"]
    print(messages["Average Interval (s)"].describe())
    print(messages["Average Frequency (Hz)"].describe())
    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Script to plot message frequency from CARMA Streets message csv data."
    )
    parser.add_argument(
        "--csv-dir", help="Directory to read csv data from.", type=Path, required=True
    )
    parser.add_argument(
        "--plots-dir",
        help="Directory to save generated plots.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--simulation",
        help="Flag indicating data is from simulation",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--message-type",
        help="Flag indicating which message type [sdsm, map, spat, time_sync, detected_object] to plot. By default, plotting for all message types [all] .",
        type=str,
        default="all",
    )

    args = parser.parse_args()
    plot_message_frequencies(
        args.csv_dir, args.plots_dir, args.simulation, args.message_type
    )


if __name__ == "__main__":
    main()
