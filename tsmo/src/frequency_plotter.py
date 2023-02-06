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
def plotter():
    input_directory_path = f'{constants.DATA_DIR}/{constants.TIMESTAMP_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    
    all_in_filenames = os.listdir(input_directory_path)
    #find the parsed file with the message type of interest
    for filename in all_in_filenames:
        try:
            messageType = filename.split(".")[0].split("_timestamps")[0]

            data = pd.read_csv(f'{input_directory_path}/{filename}')

            start = data['Timestamp(ms)'].iloc[0]
            end = data['Timestamp(ms)'].iloc[-1]

            #bin the data using the start and end time
            start_end_data = data[(data['Timestamp(ms)'] >= int(start))&(data['Timestamp(ms)'] <= int(end))]
            
            min_time = dt.datetime.fromtimestamp(start_end_data['Timestamp(ms)'].iloc[0] / 1000)
            max_time = dt.datetime.fromtimestamp(start_end_data['Timestamp(ms)'].iloc[-1] / 1000)

            #Get the array of frequencies and datetimes for the message type, using a bin size of 10 messages
            tenValueAverages = averageCalc(data, 10, messageType)           

            #plot the message frequencies vs time, as well as the max/min range value based on message type
            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 10)        
            plt.scatter([i[0] for i in tenValueAverages], [i[1] for i in tenValueAverages], c="blue", marker="^", label="Freq (hz)")
            # Add horizontal lines for differing freq requirements based on message type
            if "Streets_mom" in filename:
                plt.axhline(y=2, color='r', linestyle='--', label="frequency lower bound")
                plt.axhline(y=8, color='r', linestyle='-', label="frequency upper bound")
            elif "Streets_spat" in filename :
                plt.axhline(y=5, color='r', linestyle='--', label="frequency lower bound")
                plt.axhline(y=15, color='r', linestyle='-', label="frequency upper bound")
            else:
                plt.axhline(y=7, color='r', linestyle='--', label="frequency lower bound")
                plt.axhline(y=13, color='r', linestyle='-', label="frequency upper bound")

            plt.xticks(rotation=75)
            axs=plt.gca()
            xfmt = md.DateFormatter('%H:%M:%S') 
            axs.xaxis.set_major_formatter(xfmt)
            fig.autofmt_xdate()
            plt.xlabel('Date-Time', fontsize=18)
            plt.ylabel('Frequency (Hz)', fontsize=18)
            plt.xlim(min_time, max_time)
            plt.ylim(0, 20)
            fig.suptitle(str(messageType).replace("_", " ") + " message frequency", fontsize=18)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15)
            plotName = filename.split(".")[0].split("_timestamps")[0]
            fig.suptitle(plotName.replace("_", " ") + " Frequency", fontsize=18)
            plt.savefig(f'{output_directory_path}/{plotName}')
            fig.clf()
            plt.close()
        except:
            print("Error producing freq plot for " + str(filename))
            continue


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python3 frequency_plotter.py"')
    else:        
        plotter()  
