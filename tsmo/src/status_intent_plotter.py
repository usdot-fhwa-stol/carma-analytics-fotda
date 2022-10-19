import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.dates as md
import datetime as dt
pd.options.mode.chained_assignment = None

def status_intent_plotter():
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'

    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if "status_intent" in file and "parsed" in file:
            status_intent_data = pd.read_csv(f'{input_directory_path}/{file}')
            vehicle_ids = status_intent_data['Vehicle_ID'].unique()

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)

    #Iterate through every vehicle id in the parsed file
    for veh in vehicle_ids:
        df = status_intent_data.copy()
        subset = df[df['Vehicle_ID'] == veh]
        subset["Timestamp(s)"] = subset["Timestamp(ms)"] / 1000

        #Convert time since epoch values to date time and plot versus cur_ds 
        dates=[dt.datetime.fromtimestamp(ts) for ts in subset["Timestamp(s)"]]

        DV_first_time = 0
        LV_first_time = 0
        #check if vehicle enters "DV" state
        if 1 in subset['DV'].unique():
            DV_first_index = subset[subset['DV'] == 1].first_valid_index()
            DV_first_time = subset._get_value(DV_first_index, 'Timestamp(s)')
            print("Vehicle: " + veh + " entered DV state at time: " + str(DV_first_time))
        #check if vehicle enters "DV" state
        if 1 in subset['LV'].unique():
            LV_first_index = subset[subset['LV'] == 1].first_valid_index()
            LV_first_time = subset._get_value(LV_first_index, 'Timestamp(s)')
            print("Vehicle: " + veh + " entered LV state at time: " + str(LV_first_time))

        plt.plot(dates,subset['Cur_ds(m)'], label=veh)
        # if DV_first_time != 0:
        #     plt.axvline(x=dt.datetime.fromtimestamp(DV_first_time), color='r', linestyle='-', label=veh+" DV")
        # if LV_first_time != 0:
        #     plt.axvline(x=dt.datetime.fromtimestamp(LV_first_time), color='b', linestyle='-', label=veh+" LV")

    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax1=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax1.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    #TODO update plot title/name once log naming convention has been established
    plt.xlabel('Date-Time')
    plt.ylabel('Distance to End of Current Lane (m)')
    plt.title("Vehicle Distance vs Time")
    plt.legend()
    plt.savefig(f'{output_directory_path}/Vehicle_Distance_Vs_Time.png')


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python status_intent_plotter.py"')
    else:          
        status_intent_plotter()    
