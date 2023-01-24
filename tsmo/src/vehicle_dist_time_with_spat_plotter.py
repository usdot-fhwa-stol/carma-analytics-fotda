""" This is an analysis script for creating plots of vehicle trajectory vs time, along with spat signal data. It requires
the prior use of the modified_spat_parser.py and the status_intent_parser.py scripts to generate the necessary data from
the spat and status and intent kafka topic logs. Once those csv files have been generated, select the vehicle id and signal
group of interest. """

## How to use this script:
""" Run with python3 vehicle_dist_time_with_spat_plotter.py parsedSpatLog parsedStatusIntentLog vehicleID signalGroup"""
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

#This function will plot spat signal state data, as well as the vehicle trajectory data, for each run in the
#status and intent kafka log.
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
    #timestamps. If the diff is larger than 30000 ms (30 seconds), this indicates the start of a new run.
    status_intent_list = np.split(status_intent_data, status_intent_data[status_intent_data['Timestamp(ms)'].diff() > 30000].index)

    run = 1   
    #Iterate through each run in the status and intent data, and find the associated vehicle/spat data
    for status_intent_subset in status_intent_list:
        #Only want data where vehicle is in EV state
        status_intent_subset_ev = status_intent_subset[(status_intent_subset['Vehicle_state'] == "EV")&(status_intent_subset['Vehicle_ID'] == str(vehicle_id))]

        #need to check if the input vehicle participated in the run
        if len(status_intent_subset_ev) != 0:
            status_intent_subset_ev_copy = status_intent_subset_ev.copy()
            #Get min and max times of status and intent to bound the spat data
            status_intent_subset_ev_copy['Timestamp(s)'] = status_intent_subset_ev_copy['Timestamp(ms)'] / 1000
            min_epoch_time = status_intent_subset_ev_copy['Timestamp(ms)'].min()
            max_epoch_time = status_intent_subset_ev_copy['Timestamp(ms)'].max() + 5000 #add extra 5 seconds of data for better visualization

            #get subset of spat data using min and max times for the run and the desired signal group
            spat_subset = spat_data[(spat_data['Epoch_Time(ms)'] > min_epoch_time)&(spat_data['Epoch_Time(ms)'] <= max_epoch_time)&(spat_data['Signal_Group'].astype(int) == int(signal_group))]             

            #get entry lane id for this run
            lane = status_intent_subset_ev_copy['Entry_lane_id'].iloc[0]

            if lane != 0:
                #get lanelet length/stop bar location based on entry lane id from intersection model data
                lanelet_length = intersection_model['Length(m)'][intersection_model['Entry_lane_id'] == lane].iloc[0]

                #place the spat data slightly above the stop bar value
                plot_spat_location = lanelet_length + 10

                fig, ax1 = plt.subplots()
                fig.set_size_inches(10, 10)         

                #iterate through spat kafka data and draw a horizontal line based on signal state
                #syntax for drawing horizontal lines: ax.hlines(y, xmin, xmax)
                for i in range(0, len(spat_subset)-1):
                    #red state
                    if (spat_subset['Event_State'].iloc[i] == 3)&(spat_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='red', linewidth=10)
                    #green state
                    elif (spat_subset['Event_State'].iloc[i] == 6)&(spat_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='green', linewidth=10)
                    #yellow state
                    elif (spat_subset['Event_State'].iloc[i] == 8)&(spat_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='yellow', linewidth=10)
                    #change in state from red to green, draw green
                    elif (spat_subset['Event_State'].iloc[i] == 3)&(spat_subset['Event_State'].iloc[i+1] == 6):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='green', linewidth=10)
                    #change in state from green to yellow, draw yellow
                    elif (spat_subset['Event_State'].iloc[i] == 6)&(spat_subset['Event_State'].iloc[i+1] == 8):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='yellow', linewidth=10)
                        #change in state from yellow to red, draw red
                    elif (spat_subset['Event_State'].iloc[i] == 8)&(spat_subset['Event_State'].iloc[i+1] == 3):
                        time1 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i])
                        time2 = dt.datetime.fromtimestamp(spat_subset['Epoch_Time(s)'].iloc[i+1])
                        ax1.hlines(plot_spat_location, time1, time2, color='red', linewidth=10)

                #convert epoch times to datetime 
                dates=[dt.datetime.fromtimestamp(ts) for ts in status_intent_subset_ev_copy["Timestamp(s)"]]

                #plot distance travelled vs time using the vehicle speed as the color of the dots
                #need to take the vehicle's distance from the rear axle to the front bumper into account
                if vehicle_id == "DOT-45244":
                    sns.scatterplot(data=status_intent_subset_ev_copy, x=dates, y=lanelet_length - (status_intent_subset_ev_copy['Cur_ds(m)']-constants.DOT_45244_FRONT_BUMPER_DIST), 
                    hue=status_intent_subset_ev['Cur_Speed'], hue_order=status_intent_subset_ev_copy['Cur_Speed'], palette='viridis', ax=ax1)
                elif vehicle_id == "DOT-45245":
                    sns.scatterplot(data=status_intent_subset_ev_copy, x=dates, y=lanelet_length - (status_intent_subset_ev_copy['Cur_ds(m)']-constants.DOT_45245_FRONT_BUMPER_DIST), 
                    hue=status_intent_subset_ev['Cur_Speed'], hue_order=status_intent_subset_ev_copy['Cur_Speed'], palette='viridis', ax=ax1)
                elif vehicle_id == "DOT-45254":
                    sns.scatterplot(data=status_intent_subset_ev_copy, x=dates, y=lanelet_length - (status_intent_subset_ev_copy['Cur_ds(m)']-constants.DOT_45254_FRONT_BUMPER_DIST), 
                    hue=status_intent_subset_ev['Cur_Speed'], hue_order=status_intent_subset_ev_copy['Cur_Speed'], palette='viridis', ax=ax1)

                plt.xticks(rotation=75)
                axs=plt.gca()
                xfmt = md.DateFormatter('%H:%M:%S.%d') 
                axs.xaxis.set_major_formatter(xfmt)
                axs.xaxis.set_major_locator(md.SecondLocator(interval=3)) #add tick mark every 3 seconds
                axs.xaxis.set_minor_locator(AutoMinorLocator(3))
                axs.yaxis.set_minor_locator(AutoMinorLocator(10))
                fig.autofmt_xdate()

                min_datetime = dt.datetime.fromtimestamp(status_intent_subset_ev_copy['Timestamp(s)'].min())
                max_datetime = dt.datetime.fromtimestamp(status_intent_subset_ev_copy['Timestamp(s)'].max()+5) #add extra 5 seconds of data
                #plot horizontal line at stop bar location
                ax1.hlines(lanelet_length, min_datetime, max_datetime, color='orange', linewidth=2, label="stop\n bar")

                plt.xlim(min_datetime, max_datetime)
                plt.xlabel('Time')
                plt.ylabel('Distance Travelled (m)')

                # Setup grid
                # Create grid lines for major ticks for both x and y axis
                plt.grid()
                # Create dashed grid lines for minor ticks for both x and y axis
                plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')                
                
                fig.suptitle(vehicle_id + " Distance vs Time Signal Group " + str(signal_group) + " Run " + str(run))
                ax1.legend(title='Veh Speed \n(m/s)', loc='center left', bbox_to_anchor=(1, 0.5))
                plotName = vehicle_id + "_Distance_Vs_Time_Signal_Group_" + signal_group + "_run_" + str(run) + ".png"
                plt.savefig(f'{output_directory_path}/{plotName}')
            else:
                print("No valid entry lane id for run " + str(run))
        else:
            print("No " + str(vehicle_id) + " data for run " + str(run))

        run += 1

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Run with: "python3 vehicle_dist_time_with_spat_plotter.py parsedSpatLog parsedStatusIntentLog vehicleID signalGroup"')
    else:       
        spat_parsed = sys.argv[1]
        status_intent_parsed = sys.argv[2]
        vehicle_id = sys.argv[3]
        signal_group = sys.argv[4]
        plotter(spat_parsed, status_intent_parsed, vehicle_id, signal_group)    
