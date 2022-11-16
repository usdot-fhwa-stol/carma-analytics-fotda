import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
# from datetime import datetime
import datetime as dt
import matplotlib.dates as md
from matplotlib.collections import LineCollection

pd.options.mode.chained_assignment = None

def modified_spat_plotter():
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
   
    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        if "modified_spat" in file and "parsed" in file:
            modified_spat_data = pd.read_csv(f'{input_directory_path}/{file}')
            signal_groups = modified_spat_data['Signal_Group'].unique()

            first_day_epoch = dt.datetime(2022, 1, 1, 0, 0, 0).timestamp() * 1000 #get time since epoch for beggining of year in ms
            modified_spat_data['Epoch_Time(ms)'] = (modified_spat_data['Moy'] * 60000) + modified_spat_data['Timestamp'] + first_day_epoch #convert moy to milliseconds              
            modified_spat_data['Epoch_Time(s)'] = modified_spat_data['Epoch_Time(ms)'] / 1000
            min_time = dt.datetime.fromtimestamp(modified_spat_data['Epoch_Time(s)'].min())
            max_time = dt.datetime.fromtimestamp(modified_spat_data['Epoch_Time(s)'].max())

            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 10)

            #Iterate through every signal group in the parsed file
            for group in signal_groups:
                df = modified_spat_data.copy()
                group_subset = df[df['Signal_Group'] == group]
                dates=[dt.datetime.fromtimestamp(ts) for ts in group_subset["Epoch_Time(s)"]]               

                seg_count = 0
                #iterate through group data and plot based on current and next state
                for i in range(0, len(group_subset)-1):
                    if (group_subset['Event_State'].iloc[i] == 3)&(group_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='red', linewidth=10)
                        seg_count += 1
                    elif (group_subset['Event_State'].iloc[i] == 6)&(group_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='green', linewidth=10)
                        seg_count += 1
                    elif (group_subset['Event_State'].iloc[i] == 8)&(group_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='yellow', linewidth=10)
                        seg_count += 1
                    #change in state from red to green, draw green
                    elif (group_subset['Event_State'].iloc[i] == 3)&(group_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='green', linewidth=10)
                        seg_count += 1
                    #change in state from green to yellow, draw yellow
                    elif (group_subset['Event_State'].iloc[i] == 6)&(group_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='yellow', linewidth=10)
                        seg_count += 1
                     #change in state from yellow to red, draw red
                    elif (group_subset['Event_State'].iloc[i] == 8)&(group_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='red', linewidth=10)
                        seg_count += 1
                    else:
                        print("Time 1: " + str(group_subset['Event_State'].iloc[i]))
                        print("Time 2: " + str(group_subset['Event_State'].iloc[i+1]))

            plt.xticks(rotation=75)
            axs=plt.gca()
            xfmt = md.DateFormatter('%H:%M:%S.%f') 
            axs.xaxis.set_major_formatter(xfmt)
            fig.autofmt_xdate()
            plt.xlim(min_time, max_time)
            plt.ylim(0, 9)
            #TODO update plot title/name once log naming convention has been established
            plt.xlabel('Date-Time')
            plt.ylabel('Signal Group')
            fig.suptitle("Signal Group Event State vs Time")
            plotName = "Signal_Groups_Event_State_Vs_Time.png"
            plt.savefig(f'{output_directory_path}/{plotName}')

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python modified_spat_plotter.py"')
    else:          
        modified_spat_plotter()    
