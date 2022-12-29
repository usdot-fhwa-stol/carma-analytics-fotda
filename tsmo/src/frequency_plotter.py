""" This is a plotting script that can be used to generate plots displaying the frequency of transmission
or reception of a particular message. This script requires that timestamps have already been extracted from
the appropriate kafka log, with the use of the "timestamp_parser.py" script."""

## How to use this script:
""" Run with python3 frequency_plotter.py messageType(MOM, SPAT) startTime(epoch milliseconds) endTime(epoch milliseconds)"""
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
        #check if the message type is spat first b/c the parsed data has a slightly different naming convention than the other
        #message types
        if messageType != "spat":
            #Perform a diff on the timestamp column and store values in array
            df["Diff(ms)"] = df['Timestamp(ms)'].diff()
            windowVals = df["Diff(ms)"].iloc[i:i+window]

            #convert to "frequency", need to check if there are "window" values to average first
            if windowVals.mean() > 0 and len(windowVals) == window:
                frequency = 1000 / float(windowVals.mean())
                time_mean_arr.append([df["Timestamp(ms)"].iloc[i], dt.datetime.fromtimestamp(df["Timestamp(ms)"].iloc[i]/1000), frequency])

        #if not a spat message, perform similar operations with different column names
        else:
            df["Diff(ms)"] = df['Epoch_Time(ms)'].diff()
            windowVals = df["Diff(ms)"].iloc[i:i+window]

            if windowVals.mean() > 0 and len(windowVals) == window:
                frequency = 1000 / float(windowVals.mean())
                time_mean_arr.append([df["Epoch_Time(ms)"].iloc[i], dt.datetime.fromtimestamp(df["Epoch_Time(ms)"].iloc[i]/1000), frequency])

    return time_mean_arr

#Plots the frequency values versus datetime for the specific message type
def plotter(messageType, start, end):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    
    all_in_filenames = os.listdir(input_directory_path)
    for filename in all_in_filenames:
        if messageType in filename:
            data = pd.read_csv(f'{input_directory_path}/{filename}')
    
    start_end_data = data[(data['Timestamp(ms)'] >= int(start))&(data['Timestamp(ms)'] <= int(end))]

    #Get the array of frequencies and datetimes for the message type, using a bin size of 10 messages
    tenValueAverages = averageCalc(data, 10, messageType)           

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)        
    plt.scatter([i[1] for i in tenValueAverages], [i[2] for i in tenValueAverages], c="blue", marker="^")
    # spat broadcast requirement is 10Hz +/- 5, MOM is 5Hz +/- 3
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
    fig.suptitle(filename.split("_")[4] + " " + filename.split("_")[5] + " " + filename.split("_")[6] + " " + str(messageType).replace("_", " ") + " message frequency")
    plt.legend()
    plotName = filename.split("_")[4] + "_" + filename.split("_")[5] + "_" + filename.split("_")[6] + "_" + str(messageType) + "_message_frequency.png"
    plt.savefig(f'{output_directory_path}/{plotName}')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Run with: "python3 frequency_plotter.py messageType(MOM, SPAT) startTime(epoch milliseconds) endTime(epoch milliseconds)"')
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
        else:
            print("Please input a message type from the available options (MOM, SPAT)")
            exit()

        plotter(messageType, start, end)  
