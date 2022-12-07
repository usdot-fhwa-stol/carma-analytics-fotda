""" This is an analysis script for creating plots of entering_times and signal group phase, against timestamps of vehicles
using scheduling_service logs and modified_spat kafka data. The plots are created for EV and DV states. This script requires a scheduling_service 
log csv to be passed as an argument, as well as parsed modified_spat data. To get the parsed modified_spat data, use the modified_spat_parser.py
script in this repo on the modified_spat kafka log. The signal group and vehicle id of interest are also required to run the script """

## How to use this script:
""" Run with python3 entering_time_spat_plotter.py schedulingCsvName spatParsedCsvName desiredSignalGroup vehicleID """

### Additional Note
"""Current implementation adds line breaks when there is no vehicle in the intersection and only has entries when the vehicle has carma engaged.
This was a useful demarcation for separating runs but the script might need to be modified if implementation changes."""

import csv
from time import time
import matplotlib
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as md 
import sys
import pandas as pd
import math
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.dates as mdates
import constants
import os 
import numpy as np 

#Helper method to convert epoch timestamps, in milliseconds, to datetime objects
def convert_to_datetime(x):
    return datetime.datetime.fromtimestamp(int(x)/1000)
    
# Method to handle processing of single run stored in pandas dataframe df
def plot_run(scheduling_df, modified_spat_df, signal_group, vehicle_id, run):
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'

    # Separate out values based on states
    df_ev = scheduling_df[scheduling_df["state"] == "EV"].copy()
    df_dv  = scheduling_df[scheduling_df["state"] == "DV"].copy()

    #get min and max times for scheduling data and bin the spat data with this window
    min_time = df_ev['timestamps'].min()
    max_time = df_ev['timestamps'].max()
    if not df_dv.empty:
        max_time = df_dv['timestamps'].max()   

    spat_subset_df = modified_spat_df[(modified_spat_df['Signal_Group'] == int(signal_group))&(modified_spat_df['Epoch_Time(ms)'] > min_time)&(modified_spat_df['Epoch_Time(ms)'] < max_time)]
    spat_subset_df_copy = spat_subset_df.copy()

    # Convert timestamps and entering timestamps to datetime values
    df_ev['datetime'] = df_ev['timestamps'].apply(convert_to_datetime)
    df_ev['et_datetime'] = df_ev['et'].apply(convert_to_datetime)
    df_dv['datetime'] = df_dv['timestamps'].apply(convert_to_datetime)
    df_dv['et_datetime'] = df_dv['et'].apply(convert_to_datetime)
    spat_subset_df_copy['datetime'] = spat_subset_df_copy['Epoch_Time(ms)'].apply(convert_to_datetime)

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    ax.plot(df_ev['datetime'], df_ev['et_datetime'], color = 'red')
    ax.plot(df_dv['datetime'], df_dv['et_datetime'], color = 'blue')

    #Want to plot the signal phase info off to the right of the entering time lines
    #Just placing it a couple "seconds" to the right of the last data point
    phase_x_min_location = max_time+2000
    diff_ms = 1000
    phase_x_max_location = phase_x_min_location + diff_ms

    #iterate through spat kafka data and draw a horizontal line based on signal state
    for i in range(0, len(spat_subset_df_copy)-1):
            #red state
            if (spat_subset_df_copy['Event_State'].iloc[i] == 3)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 3):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='red', linewidth=10)
            #green state
            elif (spat_subset_df_copy['Event_State'].iloc[i] == 6)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 6):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='green', linewidth=10)
            #yellow state
            elif (spat_subset_df_copy['Event_State'].iloc[i] == 8)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 8):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='yellow', linewidth=10)
            #change in state from red to green, draw green
            elif (spat_subset_df_copy['Event_State'].iloc[i] == 3)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 6):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='green', linewidth=10)
            #change in state from green to yellow, draw yellow
            elif (spat_subset_df_copy['Event_State'].iloc[i] == 6)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 8):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='yellow', linewidth=10)
             #change in state from yellow to red, draw red
            elif (spat_subset_df_copy['Event_State'].iloc[i] == 8)&(spat_subset_df_copy['Event_State'].iloc[i+1] == 3):
                time1 = spat_subset_df_copy['datetime'].iloc[i]
                time2 = spat_subset_df_copy['datetime'].iloc[i+1]

                ax.hlines(time1, convert_to_datetime(phase_x_min_location), convert_to_datetime(phase_x_max_location), color='red', linewidth=10)

    myFmt_timestamp = mdates.DateFormatter('%H:%M:%S%d') # here you can format your datetick labels as desired
    plt.gca().xaxis.set_major_formatter(myFmt_timestamp)
    plt.gca().yaxis.set_major_formatter(myFmt_timestamp)
    ev_red_patch = mpatches.Patch(color='red', label='EV')
    dv_blue_patch = mpatches.Patch(color='blue', label='DV')
    plt.legend(handles=[ev_red_patch, dv_blue_patch], title="State", loc="upper left")
    ax.set_xlabel('Time')
    ax.set_ylabel('Entering Times')
    plotTitle = "Signal Group " + str(signal_group) + " " + str(vehicle_id) + " Entering Time Run " + str(run)
    plt.title(plotTitle)
    plotName = "Signal_Group_" + str(signal_group) + "_" + vehicle_id + "_Entering_Time_Run_" + str(run) + "_plot.png"
    plt.savefig(f'{output_directory_path}/{plotName}')

#This function will first separate the data in the scheduling log csv file into separate dataframes for each run.
#It then hands the separated data frames to the plot_run function along with the modified_spat kafka data
def process_runs(scheduling_log_name, modified_spat_log_name, signal_group, vehicle_id):
    scheduling_log_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    spat_parsed_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'

    all_in_filenames = os.listdir(scheduling_log_directory_path)
    
    #Find the desired scheduling log
    for file in all_in_filenames:
        if scheduling_log_name in file:
            scheduling_df = pd.read_csv(f'{scheduling_log_directory_path}/{scheduling_log_name}', names=('timestamps', 'vehicle_id', 'entry_lane', 'link_id', 
            'eet', 'et', 'dt', 'state'), skip_blank_lines=False)
            
            modified_spat_df = pd.read_csv(f'{spat_parsed_directory_path}/{modified_spat_log_name}')

            # Separate runs based on line breaks
            scheduling_df_list = np.split(scheduling_df, scheduling_df[scheduling_df.isnull().all(1)].index)

            run = 0
            for scheduling_df in scheduling_df_list:
                scheduling_df_veh_subset = scheduling_df[scheduling_df['vehicle_id'] == vehicle_id]

                # Drop all NA values
                scheduling_df_veh_subset = scheduling_df_veh_subset.dropna()
                if not scheduling_df_veh_subset.empty:
                    run += 1
                    plot_run(scheduling_df_veh_subset, modified_spat_df, signal_group, vehicle_id, run)

            print("Number of runs: ", run)

if __name__ == '__main__' :
    if len(sys.argv) < 5:
        print("Run with python3 entering_time_spat_plotter.py schedulingCsvName spatParsedCsvName desiredSignalGroup vehicleID")
        exit()

    scheduling_csv_file_name = sys.argv[1]
    spat_csv_file_name = sys.argv[2]
    signal_group = sys.argv[3]
    vehicle_id = sys.argv[4]
    process_runs(scheduling_csv_file_name, spat_csv_file_name, signal_group, vehicle_id)
