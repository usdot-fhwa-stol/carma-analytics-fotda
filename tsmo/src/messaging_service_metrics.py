from csv import reader
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os
import latency_parser
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
pd.options.mode.chained_assignment = None

input_directory_path = f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}'
output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
all_in_filenames = os.listdir(input_directory_path)

#get average frequency over window
def averageCalc(df, window, messageType):
    time_mean_arr = []
    for i in range(0, len(df), window):
        timediff_col = messageType + "_Message_Timediff(s)"
        timestamp_col = messageType + "_Message_Timestamp_adjusted(s)"
        tenVals = df[timediff_col].iloc[i:i+window]

        #convert to "frequency", need to check if there are ten values to average first
        if tenVals.mean() > 0 and len(tenVals) == window:
            frequency = 1000 / (float(tenVals.mean())*1000)
            time_mean_arr.append([df[timestamp_col].iloc[i], frequency])

    return time_mean_arr

def calculateBSMFrequency(consumer_bsm_data, test_start, test_end):
    #parse BSM machine time and convert to epoch time
    consumer_bsm_data['Machine_Time_BSM'] = consumer_bsm_data['Machine_Time'].str.split(":", expand=True)[2]
    consumer_bsm_data['Machine_Data_And_Time'] = consumer_bsm_data['Machine_Date'] + " " + consumer_bsm_data['Machine_Time']
    consumer_bsm_data['Machine_Time_BSM_to_Epoch'] = pd.to_datetime(consumer_bsm_data['Machine_Data_And_Time']).map(pd.Timestamp.timestamp).astype(float)

    #bound data by start and end times
    indexNames = consumer_bsm_data[(consumer_bsm_data['Machine_Time_BSM_to_Epoch'] < float(test_start))|(consumer_bsm_data['Machine_Time_BSM_to_Epoch'] > float(test_end))].index
    consumer_bsm_data.drop(indexNames, inplace=True)

    first_bsm_timestamp = float(consumer_bsm_data['Machine_Time_BSM_to_Epoch'].iloc[0])

    #create "adjusted" column to get normalized test time
    consumer_bsm_data['BSM_Message_Timestamp_adjusted(s)'] = consumer_bsm_data['Machine_Time_BSM_to_Epoch'].astype(float) - first_bsm_timestamp

    #get difference in transmit times
    consumer_bsm_data['BSM_Message_Timediff(s)'] = consumer_bsm_data['Machine_Time_BSM_to_Epoch'].diff()

    #get array of bsm frequency rates over specified time window
    return averageCalc(consumer_bsm_data, 10, "BSM"), consumer_bsm_data

def calculateMOMFrequency(consumer_mom_data, test_start, test_end):
    #parse MOM machine time and convert to epoch time
    consumer_mom_data['Machine_Time_MOM'] = consumer_mom_data['Machine_Time'].str.split(":", expand=True)[2]
    consumer_mom_data['Machine_Data_And_Time'] = consumer_mom_data['Machine_Date'] + " " + consumer_mom_data['Machine_Time']
    consumer_mom_data['Machine_Time_MOM_to_Epoch'] = pd.to_datetime(consumer_mom_data['Machine_Data_And_Time']).map(pd.Timestamp.timestamp)

    #bound data by start and end times
    indexNames = consumer_mom_data[(consumer_mom_data['Machine_Time_MOM_to_Epoch'] < float(test_start))|(consumer_mom_data['Machine_Time_MOM_to_Epoch'] > float(test_end))].index
    consumer_mom_data.drop(indexNames, inplace=True)

    first_mom_timestamp = float(consumer_mom_data['Machine_Time_MOM_to_Epoch'].iloc[0])

    #create "adjusted" column to get normalized test time
    consumer_mom_data['MOM_Message_Timestamp_adjusted(s)'] = consumer_mom_data['Machine_Time_MOM_to_Epoch'].astype(float) - first_mom_timestamp

    #get difference in transmit times
    consumer_mom_data['MOM_Message_Timediff(s)'] = consumer_mom_data['Machine_Time_MOM_to_Epoch'].diff()

    #calulate mom frequency every n seconds
    return averageCalc(consumer_mom_data, 10, "MOM")

def calculateMPMFrequency(consumer_mpm_data):
    #parse MOM machine time and convert to epoch time
    consumer_mpm_data['Machine_Time_MPM'] = consumer_mpm_data['Machine_Time'].str.split(":", expand=True)[2]
    consumer_mpm_data['Machine_Data_And_Time'] = consumer_mpm_data['Machine_Date'] + " " + consumer_mpm_data['Machine_Time']
    consumer_mpm_data['Machine_Time_MPM_to_Epoch'] = pd.to_datetime(consumer_mpm_data['Machine_Data_And_Time']).map(pd.Timestamp.timestamp)

    first_mpm_timestamp = float(consumer_mpm_data['Machine_Time_MPM_to_Epoch'].iloc[0])
    last_mpm_timestamp = float(consumer_mpm_data['Machine_Time_MPM_to_Epoch'].iloc[-1])

    #create "adjusted" column to get normalized test time
    consumer_mpm_data['MPM_Message_Timestamp_adjusted(s)'] = consumer_mpm_data['Machine_Time_MPM_to_Epoch'].astype(float) - first_mpm_timestamp

    #get difference in transmit times
    consumer_mpm_data['MPM_Message_Timediff(s)'] = consumer_mpm_data['Machine_Time_MPM_to_Epoch'].diff()

    return first_mpm_timestamp, last_mpm_timestamp

def MPMPlotter(consumer_mpm_data, test_start, test_end):
    #bound data by start and end times
    indexNames = consumer_mpm_data[(consumer_mpm_data['Machine_Time_MPM_to_Epoch'] < float(test_start))|(consumer_mpm_data['Machine_Time_MPM_to_Epoch'] > float(test_end))].index
    consumer_mpm_data.drop(indexNames, inplace=True)

    #calulate mpm frequency every n seconds
    return averageCalc(consumer_mpm_data, 10, "MPM")

def runner(filename, vehicle_id_1, vehicle_id_2):
    consumer_data = pd.read_csv(f'{input_directory_path}/{filename}_MS_consumer_parsed.csv')

    #create individual df with specific message types
    consumer_bsm = consumer_data[(consumer_data['Message_Type'] == "BSM")]
    #get the vehicle ids used during testing
    veh_ids = consumer_bsm['Vehicle_ID'].unique()

    #call the MPM frequency function first to retrieve the first and last timestamps
    #need to do this bc MPM generation starts after BSM and MOM
    consumer_mpm_data_1 = consumer_data[(consumer_data['Message_Type'] == "MPM")&(consumer_data['Vehicle_ID'] == vehicle_id_1)]
    consumer_mpm_data_2 = consumer_data[(consumer_data['Message_Type'] == "MPM")&(consumer_data['Vehicle_ID'] == vehicle_id_2)]
    veh1_mpm_start, veh1_mpm_end = calculateMPMFrequency(consumer_mpm_data_1)
    veh2_mpm_start, veh2_mpm_end = calculateMPMFrequency(consumer_mpm_data_2)

    #figure out the time window to use based on smallest time window
    start_time = veh1_mpm_start
    if veh2_mpm_start >= veh1_mpm_start:
        start_time = veh2_mpm_start

    end_time = veh1_mpm_end
    if veh2_mpm_end <= veh1_mpm_end:
        end_time = veh2_mpm_end


    consumer_bsm_data_1 = consumer_data[(consumer_data['Message_Type'] == "BSM")&(consumer_data['Vehicle_ID'] == veh_ids[0])]
    consumer_mom_data_1 = consumer_data[(consumer_data['Message_Type'] == "MOM")&(consumer_data['Vehicle_ID'] == vehicle_id_1)]
    consumer_bsm_data_2 = consumer_data[(consumer_data['Message_Type'] == "BSM")&(consumer_data['Vehicle_ID'] == veh_ids[1])]
    consumer_mom_data_2 = consumer_data[(consumer_data['Message_Type'] == "MOM")&(consumer_data['Vehicle_ID'] == vehicle_id_2)]

    #calculate message frequency for all message types
    veh1_bsm_frequency_averages, bsm_trimmed_1 = calculateBSMFrequency(consumer_bsm_data_1, start_time, end_time)
    veh2_bsm_frequency_averages, bsm_trimmed_2 = calculateBSMFrequency(consumer_bsm_data_2, start_time, end_time)
    veh1_mom_frequency_averages = calculateMOMFrequency(consumer_mom_data_1, start_time, end_time)
    veh2_mom_frequency_averages = calculateMOMFrequency(consumer_mom_data_2, start_time, end_time)
    veh1_mpm_frequency_averages = MPMPlotter(consumer_mpm_data_1, start_time, end_time)
    veh2_mpm_frequency_averages = MPMPlotter(consumer_mpm_data_2, start_time, end_time)

    # print("Start time: " + str(start_time) + " end time: " + str(end_time))
    # print("veh1: ")
    # print(veh1_mpm_frequency_averages)
    # print("veh2: ")
    # print(veh2_mpm_frequency_averages)

    #now compare first timestamps for both dataframes and create final offset timestamp column for same reference timeframe
    consumer_df_vehicle_1_first_time = bsm_trimmed_1['Machine_Time_BSM_to_Epoch'].iloc[0]
    consumer_df_vehicle_2_first_time = bsm_trimmed_2['Machine_Time_BSM_to_Epoch'].iloc[0]
    useOne = True

    if consumer_df_vehicle_1_first_time > consumer_df_vehicle_2_first_time:
        diff = consumer_df_vehicle_1_first_time - consumer_df_vehicle_2_first_time
        bsm_trimmed_1['Final_Offset_Timestamp'] = bsm_trimmed_1['Machine_Time_BSM_to_Epoch']
        bsm_trimmed_2['Final_Offset_Timestamp'] = bsm_trimmed_2['Machine_Time_BSM_to_Epoch'] + diff
    elif consumer_df_vehicle_2_first_time >= consumer_df_vehicle_1_first_time:
        diff = consumer_df_vehicle_2_first_time - consumer_df_vehicle_1_first_time
        bsm_trimmed_2['Final_Offset_Timestamp'] = bsm_trimmed_2['Machine_Time_BSM_to_Epoch']
        bsm_trimmed_1['Final_Offset_Timestamp'] = bsm_trimmed_1['Machine_Time_BSM_to_Epoch'] + diff
        useOne = False

    bsm_trimmed_1['Final_Offset_Timestamp_Adjusted'] = bsm_trimmed_1['Final_Offset_Timestamp'] - consumer_df_vehicle_1_first_time
    bsm_trimmed_2['Final_Offset_Timestamp_Adjusted'] = bsm_trimmed_2['Final_Offset_Timestamp'] - consumer_df_vehicle_2_first_time

    #plot acceleration vs time
    # fig, ax1 = plt.subplots()
    # fig.set_size_inches(10, 10)
    # plt.plot(bsm_trimmed_1['Final_Offset_Timestamp_Adjusted'], bsm_trimmed_1['BSM_Accel_Long(m/s^2)'].astype(float), c="blue", label=vehicle_id_1)
    # plt.plot(bsm_trimmed_2['Final_Offset_Timestamp_Adjusted'], bsm_trimmed_2['BSM_Accel_Long(m/s^2)'].astype(float), c="green", label=vehicle_id_2)
    # plt.xlabel('Test Time (s)')
    # plt.ylabel('Acceleration (m/s^2)')
    # if bsm_trimmed_2['Final_Offset_Timestamp_Adjusted'].max() >= bsm_trimmed_1['Final_Offset_Timestamp_Adjusted'].max():
    #     plt.xticks(np.arange(0, bsm_trimmed_2['Final_Offset_Timestamp_Adjusted'].max(), 2))
    # else:
    #     plt.xticks(np.arange(0, bsm_trimmed_1['Final_Offset_Timestamp_Adjusted'].max(), 2))
    #
    # plt.title("Vehicle 1 and Vehicle 2 BSM Acceleration")
    # plt.legend()
    # plt.savefig(f'{output_directory_path}/{filename}_Acceleration_vs_Time.png')

    #plot BSM frequency
    figure(figsize=(10,10))
    plt.scatter([i[0] for i in veh1_bsm_frequency_averages], [i[1] for i in veh1_bsm_frequency_averages], c="blue", marker="^", label=vehicle_id_1)
    plt.scatter([i[0] for i in veh2_bsm_frequency_averages], [i[1] for i in veh2_bsm_frequency_averages], c="green", label=vehicle_id_2)
    plt.axhline(y=5, color='r', linestyle='--', label="frequency lower bound")
    plt.axhline(y=15, color='r', linestyle='-', label="frequency upper bound")
    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    plt.xlim(0,consumer_bsm_data_1['BSM_Message_Timestamp_adjusted(s)'].max())
    plt.ylim(0,20)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " BSM Frequency", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig(f'{output_directory_path}/{filename}_BSM_Frequency.png')

    #plot MOM frequency
    figure(figsize=(10,10))
    plt.scatter([i[0] for i in veh1_mom_frequency_averages], [i[1] for i in veh1_mom_frequency_averages], c="blue", marker="^", label=vehicle_id_1)
    plt.scatter([i[0] for i in veh2_mom_frequency_averages], [i[1] for i in veh2_mom_frequency_averages], c="green", label=vehicle_id_2)
    plt.axhline(y=5, color='r', linestyle='--', label="frequency lower bound")
    plt.axhline(y=15, color='r', linestyle='-', label="frequency upper bound")
    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    plt.xlim(0,consumer_mom_data_1['MOM_Message_Timestamp_adjusted(s)'].max())
    plt.ylim(0,20)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " MOM Frequency", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig(f'{output_directory_path}/{filename}_MOM_Frequency.png', fontsize=18)

    #plot MPM frequency
    figure(figsize=(10,10))
    plt.scatter([i[0] for i in veh1_mpm_frequency_averages], [i[1] for i in veh1_mpm_frequency_averages], c="blue", marker="^", label=vehicle_id_1)
    plt.scatter([i[0] for i in veh2_mpm_frequency_averages], [i[1] for i in veh2_mpm_frequency_averages], c="green", label=vehicle_id_2)
    plt.axhline(y=5, color='r', linestyle='--', label="frequency lower bound")
    plt.axhline(y=15, color='r', linestyle='-', label="frequency upper bound")
    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    if consumer_mpm_data_1['MPM_Message_Timestamp_adjusted(s)'].max() <= consumer_mpm_data_2['MPM_Message_Timestamp_adjusted(s)'].max():
        plt.xlim(0,consumer_mpm_data_1['MPM_Message_Timestamp_adjusted(s)'].max())
    else:
        plt.xlim(0,consumer_mpm_data_2['MPM_Message_Timestamp_adjusted(s)'].max())

    plt.ylim(0,20)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " MPM Frequency", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig(f'{output_directory_path}/{filename}_MPM_Frequency.png', fontsize=18)


    # if useOne == True:
    #     latency_parser.runner(consumer_bsm_data_1['Final_Offset_Timestamp'].iloc[0], consumer_bsm_data_1['Final_Offset_Timestamp'].iloc[-1])
    # else:
    #     latency_parser.runner(consumer_bsm_data_2['Final_Offset_Timestamp'].iloc[0], consumer_bsm_data_2['Final_Offset_Timestamp'].iloc[-1])