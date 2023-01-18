""" This is a plotting script that can be used to generate plots displaying the frequency of transmission
or reception of a particular message. This script requires that timestamps have already been extracted from
the appropriate kafka log or pcap file, with the use of either the "timestamp_parser.py" script or the
"platform_obu_pcap_parser.py" script."""

## How to use this script:
""" Run with python3 frequency_plotter.py messageType(MOM, SPAT, BSM_in, MOM_in, MPM_in) startTime(epoch milliseconds) endTime(epoch milliseconds)"""
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

#This method will return an array containing the average tx/rx frequency of the message over a window of configurable size and
#a datetime object for that frequency value.
def averageCalc(df, window, messageType):
    time_mean_arr = []
    #iterate through the data in increments of "window" size
    for i in range(0, len(df), window):
        #Perform a diff on the timestamp column and store values in array
        df["Diff(ms)"] = df['Timestamp(ms)'].diff()
        windowVals = df["Diff(ms)"].iloc[i:i+window]

        #convert to "frequency", need to check if there are "window" values to average first
        if windowVals.mean() > 0 and len(windowVals) == window:
            frequency = 1000 / float(windowVals.mean())
            time_mean_arr.append([dt.datetime.fromtimestamp(df["Timestamp(ms)"].iloc[i]/1000), frequency])

    return time_mean_arr

#Plots the frequency values versus datetime for the specific message type using the start and end timestamps provided
#by the user
def plotter(messageType, start, end):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    
    all_in_filenames = os.listdir(input_directory_path)
    #find the parsed file with the message type of interest
    for filename in all_in_filenames:
        if messageType in filename and "timestamps" in filename:
            data = pd.read_csv(f'{input_directory_path}/{filename}')
    
    #bin the data using the start and end time
    start_end_data = data[(data['Timestamp(ms)'] >= int(start))&(data['Timestamp(ms)'] <= int(end))]
    
    min_time = dt.datetime.fromtimestamp(start_end_data['Timestamp(ms)'].iloc[0] / 1000)
    max_time = dt.datetime.fromtimestamp(start_end_data['Timestamp(ms)'].iloc[-1] / 1000)

    #Get the array of frequencies and datetimes for the message type, using a bin size of 10 messages
    tenValueAverages = averageCalc(data, 10, messageType)           

    #plot the message frequencies vs time, as well as the max/min range value based on message type
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)        
    plt.scatter([i[0] for i in tenValueAverages], [i[1] for i in tenValueAverages], c="blue", marker="^", label=messageType)
    # spat broadcast requirement is 10Hz +/- 5, MOM is 1Hz +/- 0.5
    if messageType == "spat":
        plt.axhline(y=5, color='r', linestyle='--', label="frequency lower bound")
        plt.axhline(y=15, color='r', linestyle='-', label="frequency upper bound")
    elif messageType == "scheduling_plan":
        plt.axhline(y=0.5, color='r', linestyle='--', label="frequency lower bound")
        plt.axhline(y=1.5, color='r', linestyle='-', label="frequency upper bound")

    plt.xticks(rotation=75)
    axs=plt.gca()
    xfmt = md.DateFormatter('%H:%M:%S') 
    axs.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    plt.xlabel('Date-Time')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(min_time, max_time)
    fig.suptitle(str(messageType).replace("_", " ") + " message frequency")
    plt.legend()
    plotName = str(messageType) + "_message_frequency.png"
    plt.savefig(f'{output_directory_path}/{plotName}')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Run with: "python3 frequency_plotter.py messageType(MOM, SPAT, BSM_in, MOM_in, MPM_in) startTime(epoch milliseconds) endTime(epoch milliseconds)"')
    else:  
        message = sys.argv[1] 
        start = sys.argv[2]
        end = sys.argv[3]

        #map user input to the file naming convention used in the kafka logs and then call the plotter method
        messageType = ""
        if message == "MOM":
            messageType = "scheduling_plan"
        elif message == "SPAT":
            messageType = "spat"
        elif message == "BSM_in":
            messageType = "bsm_in"
        elif message == "MOM_in":
            messageType = "mom_in"
        elif message == "MPM_in":
            messageType = "mpm_in"
        else:
            print("Please input a message type from the available options (MOM, SPAT, BSM_in, MOM_in, MPM_in)")
            exit()

        plotter(messageType, start, end)  
