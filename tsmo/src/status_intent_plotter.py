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
    intersection_model_directory_path = f'{constants.DATA_DIR}/{constants.INTERSECTION_MODEL_DIR}'

    intersection_model = pd.read_csv(f'{intersection_model_directory_path}/intersection_model_parsed.csv')

    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        if "status_intent" in file and "parsed" in file:
            status_intent_data = pd.read_csv(f'{input_directory_path}/{file}')
            #Get all entry lanes from test data
            entry_lane_ids = status_intent_data['Entry_lane_id'].unique()

    #Get min and max times of dataset to put all plots on same time-scale
    status_intent_data['Timestamp(s)'] = status_intent_data['Timestamp(ms)'] / 1000
    min_time = dt.datetime.fromtimestamp(status_intent_data['Timestamp(s)'].min())
    max_time = dt.datetime.fromtimestamp(status_intent_data['Timestamp(s)'].max())
    print("Min time: " + str(min_time) + " max time: " + str(max_time))

    #Iterate through every entry lane in the parsed file
    for lane in entry_lane_ids:
        df = status_intent_data.copy()
        lane_subset = df[df['Entry_lane_id'] == lane]
        lane_subset["Timestamp(s)"] = lane_subset["Timestamp(ms)"] / 1000

        #Get the unique vehicle ids in the entry lane for the data set
        vehicles_in_lane = lane_subset['Vehicle_ID'].unique()
        num_vehicles_in_lane = len(vehicles_in_lane)
        print("Vehicles in entry lane " + str(lane) + ": " + str(vehicles_in_lane))

        fig, axs = plt.subplots(num_vehicles_in_lane, squeeze=False)
        axs = axs.flatten()
        fig.set_size_inches(10, 10)        

        #get lanelet length based on entry lane id from intersection model data
        lanelet_length = intersection_model['Length(m)'][intersection_model['Entry_lane_id'] == lane].iloc[0]

        i = 0
        for veh in vehicles_in_lane:
            #Only want to plot when a vehicle is in the 'EV' state
            veh_subset = lane_subset[(lane_subset['Vehicle_ID'] == veh)&(lane_subset['Vehicle_state'] == "EV")]
            dates=[dt.datetime.fromtimestamp(ts) for ts in veh_subset["Timestamp(s)"]]
            axs[i].plot(dates,lanelet_length - veh_subset['Cur_ds(m)'], label=veh)
            i += 1        

        plt.xticks(rotation=75)
        axs=plt.gca()
        xfmt = md.DateFormatter('%H:%M:%S') 
        axs.xaxis.set_major_formatter(xfmt)
        fig.autofmt_xdate()
        plt.xlim(min_time, max_time)
        #TODO update plot title/name once log naming convention has been established
        plt.xlabel('Date-Time')
        plt.ylabel('Distance to End of Current Lane (m)')
        fig.suptitle("Entry Lane " + str(lane) + " Vehicle Distance vs Time")
        plt.legend()
        plotName = "Entry_Lane_" + str(lane) + "_Vehicle_Distance_Vs_Time.png"
        plt.savefig(f'{output_directory_path}/{plotName}')


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python status_intent_plotter.py"')
    else:          
        status_intent_plotter()    
