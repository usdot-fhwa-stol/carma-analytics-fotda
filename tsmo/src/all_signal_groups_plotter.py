""" This is a plotting script that can be used to generate a plot displaying the status of every signal group 
over time. It requires the modified spat topic data to be extracted using the modified_spat_parser.py script and
the desired phase plan topic to be extracted using the desired_phase_plan_parser.py script."""

## How to use this script:
""" Run with python3 all_signal_groups_plotter.py spatParsedFile desiredPhasePlanParsedFile"""
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
import datetime as dt
import matplotlib.dates as md
from matplotlib.lines import Line2D
pd.options.mode.chained_assignment = None

# This method will read the parsed spat data produced by the modified_spat_parser script. It then will
# iterate through all signal groups and plot the signal state as a horizontal line with a corresponding
# color.
def plotter(spatParsedFile, desiredPhasePlanParsedFile):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
   
    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        #read the parsed spat data file generated by the modified_spat_parser script
        if spatParsedFile in file:
            modified_spat_data = pd.read_csv(f'{input_directory_path}/{file}')
        elif desiredPhasePlanParsedFile in file:
            dpp_data = pd.read_csv(f'{input_directory_path}/{file}')

    signal_groups = modified_spat_data['Signal_Group'].unique()

    #Split up desired phase plan data into 1 minute bins
    run = 0
    dpp_list = np.split(dpp_data, dpp_data[dpp_data['Timestamp(ms)'].diff() > 60000].index)

    #Iterate through each run's desired phase plan
    for dpp in dpp_list:
        run += 1
        #Get min and max times to use for spat plotting, add 3 second buffer on either side
        min_time = (dpp['Start_Time(ms)'].min() - 5000) / 1000
        max_time = (dpp['End_Time(ms)'].max() + 5000) / 1000

        min_datetime = dt.datetime.fromtimestamp(min_time)
        max_datetime = dt.datetime.fromtimestamp(max_time)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(14, 14)

        #Get unique create times and signal groups for the dpp run data
        create_times = dpp['Timestamp(ms)'].unique()
        dpp_groups = dpp['Signal_Group'].unique()
        #Iterate through each unique create time
        for time in create_times:
            #Then iterate through the available signal groups
            for group in dpp_groups:
                try:
                    group_start_time = dpp.loc[(dpp['Timestamp(ms)'] == time)&(dpp['Signal_Group'] == group), 'Start_Time(ms)'].iloc[0]
                    group_end_time = dpp.loc[(dpp['Timestamp(ms)'] == time)&(dpp['Signal_Group'] == group), 'End_Time(ms)'].iloc[0]

                    time1 = dt.datetime.fromtimestamp(group_start_time / 1000)
                    time2 = dt.datetime.fromtimestamp(group_end_time / 1000)
                    ax1.hlines(1, time1, time2, color='green', linewidth=10)

                    #add text box with signal group number in middle of the green box
                    text_time = (group_end_time + group_start_time) / 2
                    text_time_dt = dt.datetime.fromtimestamp(text_time / 1000)
                    
                    label = ""
                    if group == 2:
                        label = "East"
                    elif group == 5:
                        label = "South"
                    elif group == 8:
                        label = "West"
                    elif group == 11:
                        label = "North"
                    plt.text(text_time_dt, 0.95, label, fontweight='bold', fontsize=16, ha='center')
                except:
                    continue            

        #Iterate through every signal group in the parsed file
        for group in signal_groups:
            #Excluding unused signal group 10 data 
            if group != 10:
                df = modified_spat_data.copy()
                group_subset = df[(df['Signal_Group'] == group)&(df['Epoch_Time(s)'] >= min_time)&(df['Epoch_Time(s)'] <= max_time)]
                dates=[dt.datetime.fromtimestamp(ts) for ts in group_subset["Epoch_Time(s)"]]               

                #iterate through group data and plot based on current and next state
                for i in range(0, len(group_subset)-1):
                    #red state
                    if (group_subset['Event_State'].iloc[i] == 3)&(group_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='red', linewidth=10)
                    #green state
                    elif (group_subset['Event_State'].iloc[i] == 6)&(group_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='green', linewidth=10)
                    #yellow state
                    elif (group_subset['Event_State'].iloc[i] == 8)&(group_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='yellow', linewidth=10)
                    #change in state from red to green, draw green
                    elif (group_subset['Event_State'].iloc[i] == 3)&(group_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='green', linewidth=10)
                    #change in state from green to yellow, draw yellow
                    elif (group_subset['Event_State'].iloc[i] == 6)&(group_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='yellow', linewidth=10)
                        #change in state from yellow to red, draw red
                    elif (group_subset['Event_State'].iloc[i] == 8)&(group_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(group_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(group, time1, time2, color='red', linewidth=10)
   
        myFmt_timestamp = md.DateFormatter('%H:%M:%S.%d')
        plt.gca().xaxis.set_major_formatter(myFmt_timestamp)
        # Set minor tick interval for 5
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))
        # Rotate major tick labels on x axis 45 degreess for readability
        plt.xticks(rotation=45, ha='right')

        fig.autofmt_xdate()
        # Create grid lines for major ticks for both x and y axis
        plt.grid()
        # Create dashed grid lines for minor ticks for both x and y axis
        plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
        plt.xlim(min_datetime, max_datetime)
        plt.ylim(0, 12)
        ax1.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
        ax1.set_yticklabels(["", "DPP", "East", "", "", "South", "", "", "West", "", "", "North", ""])
        plt.xlabel('Date-Time', fontsize=18)
        plt.ylabel('Signal Group', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        #create custom legend
        custom_lines = [Line2D([0], [0], color="g", lw=4),
                        Line2D([0], [0], color="yellow", lw=4),
                        Line2D([0], [0], color="r", lw=4)]
        ax1.legend(custom_lines, ['Green\nPhase', 'Yellow\nPhase', 'Red\nPhase'], loc='upper right', bbox_to_anchor=(1.14, 1.02),
        fontsize=15)

        fig.suptitle("Signal Group Event State vs Time Run " + str(run), fontsize=18)
        plotName = "Signal_Groups_Event_State_Vs_Time_Run_"+str(run)+".png"
        plt.savefig(f'{output_directory_path}/{plotName}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run with: "python3 all_signal_groups_plotter.py spatParsedFile desiredPhasePlanParsedFile"')
    else:          
        spatParsedFile = sys.argv[1]
        desiredPhasePlanParsedFile = sys.argv[2]
        plotter(spatParsedFile, desiredPhasePlanParsedFile)    
