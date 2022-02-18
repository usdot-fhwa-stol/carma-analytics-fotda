from csv import reader
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os

input_directory_path = f'{constants.DATA_DIR}/{constants.LATENCY_DIR}'
output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
all_in_filenames = os.listdir(input_directory_path)

def plotter():
    for file in all_in_filenames:
        filename = file.split(".")[0]
        merged_file = pd.read_csv(f'{input_directory_path}/{file}')
        if len(merged_file['latency(s)'].dropna()) > 0:
            first_timestamp = merged_file['tx_time'].iloc[0]
            merged_file['Overall_time(s)'] = merged_file['tx_time'] - first_timestamp
            merged_file['latency(ms)'] = merged_file['latency(s)'] * 1000

            #plot all calculated values
            figure(figsize=(10,10))
            plt.scatter(merged_file['Overall_time(s)'], merged_file['latency(ms)'])
            plt.xlabel('Test Time (s)')
            plt.ylabel('Latency (ms)')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{output_directory_path}/{filename}_latency.png')


plotter()
