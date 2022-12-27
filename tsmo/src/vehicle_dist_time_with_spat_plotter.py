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
import seaborn as sns 

def plotter(spat_parsed, status_intent_parsed, vehicle_id, signal_group):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    intersection_model_directory_path = f'{constants.DATA_DIR}/{constants.INTERSECTION_MODEL_DIR}'

    #edit based on intersection used during testing
    intersection_model = pd.read_csv(f'{intersection_model_directory_path}/intersection_model_west_intersection_parsed.csv')

    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        if spat_parsed in file:
            spat_data = pd.read_csv(f'{input_directory_path}/{file}')
        elif status_intent_parsed in file:
            status_intent_data = pd.read_csv(f'{input_directory_path}/{file}')
  
    #Need to split parsed status and intent into individual runs. Do this by checking the difference in consecutive
    #timestamps. If the diff is larger than 30000 ms (1 min), this indicates the start of a new run.
    status_intent_list = np.split(status_intent_data, status_intent_data[status_intent_data['Timestamp(ms)'].diff() > 30000].index)

    run = 0   
    #Iterate through each run in the status and intent data, and find the associated spat data
    for status_intent_subset in status_intent_list:
        run += 1

        #Get min and max times of status and intent to bound the spat data
        status_intent_subset['Timestamp(s)'] = status_intent_subset['Timestamp(ms)'] / 1000
        min_epoch_time = status_intent_subset['Timestamp(ms)'].min()
        max_epoch_time = status_intent_subset['Timestamp(ms)'].max()

        min_datetime = dt.datetime.fromtimestamp(status_intent_subset['Timestamp(s)'].min())
        max_datetime = dt.datetime.fromtimestamp(status_intent_subset['Timestamp(s)'].max())

        spat_subset = spat_data[(spat_data['Epoch_Time(ms)'] > min_epoch_time)&(spat_data['Epoch_Time(ms)'] <= max_epoch_time)&(spat_data['Signal_Group'].astype(int) == int(signal_group))]             

        fig, ax1 = plt.subplots()
        fig.set_size_inches(10, 10) 

        #get entry lane id for this run
        lane = status_intent_subset['Entry_lane_id'].iloc[0]
        #get the max distance, just to place the spat data slightly above
        max_ds = status_intent_subset['Cur_ds(m)'].max() + 10

        #iterate through spat kafka data and draw a horizontal line based on signal state
        #syntax for drawing horizontal lines: ax.hlines(y, xmin, xmax)
        for i in range(0, len(spat_subset)-1):
            #red state
            if (spat_subset['Event_State'].iloc[i] == 3)&(spat_subset['Event_State'].iloc[i+1] == 3):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='red', linewidth=10)
            #green state
            elif (spat_subset['Event_State'].iloc[i] == 6)&(spat_subset['Event_State'].iloc[i+1] == 6):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='green', linewidth=10)
            #yellow state
            elif (spat_subset['Event_State'].iloc[i] == 8)&(spat_subset['Event_State'].iloc[i+1] == 8):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='yellow', linewidth=10)
            #change in state from red to green, draw green
            elif (spat_subset['Event_State'].iloc[i] == 3)&(spat_subset['Event_State'].iloc[i+1] == 6):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='green', linewidth=10)
            #change in state from green to yellow, draw yellow
            elif (spat_subset['Event_State'].iloc[i] == 6)&(spat_subset['Event_State'].iloc[i+1] == 8):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='yellow', linewidth=10)
                #change in state from yellow to red, draw red
            elif (spat_subset['Event_State'].iloc[i] == 8)&(spat_subset['Event_State'].iloc[i+1] == 3):
                time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                ax1.hlines(max_ds, time1, time2, color='red', linewidth=10)

        #get lanelet length based on entry lane id from intersection model data
        lanelet_length = intersection_model['Length(m)'][intersection_model['Entry_lane_id'] == lane].iloc[0]
        dates=[dt.datetime.fromtimestamp(ts) for ts in status_intent_subset["Timestamp(s)"]]
        sns.scatterplot(data=status_intent_subset, x=dates, y=lanelet_length - status_intent_subset['Cur_ds(m)'], 
        hue=status_intent_subset['Cur_Speed'], hue_order=status_intent_subset['Cur_Speed'], palette='viridis', ax=ax1)

        plt.xticks(rotation=75)
        axs=plt.gca()
        xfmt = md.DateFormatter('%H:%M:%S') 
        axs.xaxis.set_major_formatter(xfmt)
        fig.autofmt_xdate()
        plt.xlim(min_datetime, max_datetime)
        #TODO update plot title/name once log naming convention has been established
        plt.xlabel('Date-Time')
        plt.ylabel('Distance Travelled (m)')
        fig.suptitle(vehicle_id + " Distance vs Time Signal Group " + str(signal_group) + " Run " + str(run))

        ax1.legend(title='Vehicle Speed (m/s)', loc='center left', bbox_to_anchor=(0.9, 0.5))
        plotName = vehicle_id + "_Distance_Vs_Time_Signal_Group_" + signal_group + "_run_" + str(run) + ".png"
        plt.savefig(f'{output_directory_path}/{plotName}')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Run with: "python3 vehicle_dist_time_with_spat_plotter.py parsedSpatLog parsedStatusIntentLog vehicleID signalGroup"')
    else:       
        spat_parsed = sys.argv[1]
        status_intent_parsed = sys.argv[2]
        vehicle_id = sys.argv[3]
        signal_group = sys.argv[4]
        plotter(spat_parsed, status_intent_parsed, vehicle_id, signal_group)    
