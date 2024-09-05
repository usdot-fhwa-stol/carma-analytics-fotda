import math
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


class MeasurementTimeMetric:
    """Read sensor data sharing message (SDSM) message in pandas dataframe.
    Extract the measurement time values from each SDSM message. Group the measurement time
    in a predefined intervals and count the interval. Plot the measurement time count based
    on the list of retured measurement time intervals."""

    def __init__(self, message_data_frame: DataFrame) -> None:
        self._message_data_frame = message_data_frame
        self._measurement_time_data_frame = DataFrame()

    def __calc_mt_interval(self) -> list[int]:
        """
        Interval of 10 ms range: [0, 10, 20, ..., 1000].
        Estimate which interval the measurement_time fall within.
        """
        mt_intervals = []
        for _, item in self._measurement_time_data_frame["measurement_time"].items():
            interval = math.floor(item / 10) * 10
            mt_intervals.append(interval)
        return mt_intervals

    def __count_mt_interval(self, mt_interval: list[int]):
        """Based on the list of measurement time interval, calculate the numbers of
        measurement_time that have the same interval."""
        mt_interval_df = DataFrame(mt_interval, columns=["measurement_time"])
        mt_interval_counts = mt_interval_df.value_counts(sort=False)
        return mt_interval_counts

    def __extract_mt(self):
        """Parse the SDSM message and extract the objects fields.
        Further extract the measurement_time field and its value.
        Output the measurement_time into a dataframe for further processing."""
        objects = self._message_data_frame["Objects"]
        measurement_time = []
        for obj in objects:
            mt_match = re.findall(r"\'measurement_time\': \d+", obj)
            if len(mt_match) > 0:
                mt_value = re.findall(r"\d+", mt_match[0])
                measurement_time.append(int(mt_value[0]))
        self._measurement_time_data_frame = DataFrame(
            measurement_time, columns=["measurement_time"]
        )

    def plot_mt_interval_count(self, plot_dir: Path):
        """Plot measurement_time count in a bar chart,
        and add a text on the chart to describe details of the metric."""
        self.__extract_mt()
        mt_interval = self.__calc_mt_interval()
        mt_interval_count = self.__count_mt_interval(mt_interval)
        print(mt_interval_count)
        # Add 5 to offset the plot x-axis to plot bar in the middle position within an interval range.
        mt_index = [t[0] + 5 for t in mt_interval_count.index.tolist()]
        mt_interval_count_values = mt_interval_count.values.tolist()

        # Plot the measurement time count data in a bar chart
        fig, plot = plt.subplots(
            1, sharex=True, layout="constrained", figsize=[15, 2.5]
        )
        barContainer = plot.bar(mt_index, mt_interval_count_values, 10, 0)

        # Modify x-axis ticks
        xtick_count = 20
        xtick_step = math.floor(max(mt_index) / xtick_count)
        plt.xticks(
            np.arange(
                0,
                max(mt_index) + 10,
                # By default, x-axis ticks every 10 ms and maximum x-axis 200 ms.
                # If x-axis exceeds 200 ms, use xtick_count to determin the numbers of x-ticks
                xtick_step if xtick_step > 10 else 10,
            )
        )
        # Add metadata for bar plot
        plot.bar_label(barContainer, padding=3)
        plot.set_ylim(bottom=0)
        plot.set_xlim(left=0)
        plot.minorticks_on()
        plot.grid(which="major", axis="both")

        # Add metric text on the plot
        plot.text(
            max(mt_index) - 5,
            max(mt_interval_count_values) / 3,
            self._measurement_time_data_frame.describe().round(),
            fontdict={
                "family": "serif",
                "color": "darkred",
                "weight": "normal",
                "size": 8,
            },
            bbox=dict(facecolor="none", edgecolor="brown", pad=2),
        )

        # Figure metadata to describe the title, x and y label, saved location of the plot
        fig.suptitle("SDSM Measurement Time Metric", fontsize=14)
        fig.supxlabel("Measurement Time Interval (ms)", fontsize=12)
        fig.supylabel("Measurement Count", fontsize=12)
        fig.savefig(f"{plot_dir}/measurement_time_metric.png")
