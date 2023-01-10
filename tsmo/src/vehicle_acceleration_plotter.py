""" This is a plotting script that can be used to generate plots displaying the acceleration of a vehicle versus time.
This script requires that the status and intent kafka log has been parsed using the status_intent_parser.py script. This 
script will split up the parsed data into individual runs and plot each as an individual plot."""

## How to use this script:
""" Run with python3 vehicle_acceleration_plotter.py parsedStatusIntentLog vehicleID"""
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

#Separates the status and intent data into individual runs and plots acceleration vs time for the specified vehicle id.
def plotter(status_parsed, vehicle_id):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    
    all_in_filenames = os.listdir(input_directory_path)
    #find the status and intent parsed file 
    for filename in all_in_filenames:
        if status_parsed in filename:
            status_intent_data = pd.read_csv(f'{input_directory_path}/{filename}')       

    #Need to split parsed status and intent into individual runs. Do this by checking the difference in consecutive
    #timestamps. If the diff is larger than 30000 ms (30 seconds), this indicates the start of a new run.
    status_intent_list = np.split(status_intent_data, status_intent_data[status_intent_data['Timestamp(ms)'].diff() > 30000].index)

    run = 1   
    #Iterate through each run in the status and intent data, and find the associated spat data
    for status_intent_subset in status_intent_list:
        #Get subset of data for the specified vehicle id
        vehicle_subset = status_intent_subset[status_intent_subset['Vehicle_ID'] == vehicle_id]
        vehicle_subset_copy = vehicle_subset.copy()
        vehicle_subset_copy['Timestamp(s)'] = vehicle_subset_copy['Timestamp(ms)'] / 1000

        #Convert epoch timestamps to date-time
        dates=[dt.datetime.fromtimestamp(ts) for ts in vehicle_subset_copy["Timestamp(s)"]]
            
        #plot the vehicle accelartion vs time
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10, 10)        
        plt.scatter(dates, vehicle_subset_copy['Cur_Accel'], c="blue", marker="^")
        # acceleration limits are 3/-3 m/s^2
        plt.axhline(y=3, color='r', linestyle='--', label="accel lower bound")
        plt.axhline(y=-3, color='r', linestyle='-', label="accel upper bound")
        plt.xticks(rotation=75)
        axs=plt.gca()
        xfmt = md.DateFormatter('%H:%M:%S') 
        axs.xaxis.set_major_formatter(xfmt)
        fig.autofmt_xdate()
        plt.xlabel('Date-Time')
        plt.ylabel('Acceleration (m/s^2)')
        plt.ylim(-6, 6)
        fig.suptitle(vehicle_id + " Acceleration vs Time Run " + str(run))
        plt.legend()
        plotName = vehicle_id + "_Acceleration_vs_Time_Run_" + str(run) + ".png"
        plt.savefig(f'{output_directory_path}/{plotName}')

        run += 1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run with: "python3 vehicle_acceleration_plotter.py parsedStatusIntentLog vehicleID"')
    else:  
        status_parsed = sys.argv[1]
        vehicle_id = sys.argv[2] 

        plotter(status_parsed, vehicle_id)  
