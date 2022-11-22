from csv import reader
from shutil import which
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
pd.options.mode.chained_assignment = None

input_directory_path = f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}'
output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
all_in_filenames = os.listdir(input_directory_path)

#get average frequency over window
def averageCalc(df, window, messageType):
    time_mean_arr = []
    for i in range(0, len(df), window):
        timediff_col = messageType + "_Message_Timediff(s)"
        timestamp_col = "Message_Timestamp_Adjusted_To_Consumer(s)"
        tenVals = df[timediff_col].iloc[i:i+window]

        #convert to "frequency", need to check if there are ten values to average first
        if tenVals.mean() > 0 and len(tenVals) == window:
            frequency = 1000 / (float(tenVals.mean())*1000)
            time_mean_arr.append([df[timestamp_col].iloc[i], frequency])

    return time_mean_arr

#trim the initial dataset and create common time scale
def initializeData(consumer_df, producer_df, vehicle_1, vehicle_2):
    #create individual df specific to vehicle
    consumer_df_vehicle_1 = consumer_df[consumer_df['v_id'] == vehicle_1]
    producer_df_vehicle_1 = producer_df[producer_df['v_id'] == vehicle_1]
    consumer_df_vehicle_2 = consumer_df[consumer_df['v_id'] == vehicle_2]
    producer_df_vehicle_2 = producer_df[producer_df['v_id'] == vehicle_2]

    consumer_diff_1 = 0
    consumer_diff_2 = 0
    producer_diff_1 = 0
    producer_diff_2 = 0

    print("vehicle 1: " + str(vehicle_1) + " vehicle 2: " + str(vehicle_2))
    #now compare first timestamps for both dataframes and create final offset timestamp column for same reference timeframe
    consumer_df_vehicle_1_first_time = consumer_df_vehicle_1['Message_Timestamp'].iloc[0]
    consumer_df_vehicle_2_first_time = consumer_df_vehicle_2['Message_Timestamp'].iloc[0]
    if consumer_df_vehicle_1_first_time > consumer_df_vehicle_2_first_time:
        consumer_diff_1 = consumer_df_vehicle_1_first_time - consumer_df_vehicle_2_first_time
    elif consumer_df_vehicle_2_first_time >= consumer_df_vehicle_1_first_time:
        consumer_diff_2 = consumer_df_vehicle_2_first_time - consumer_df_vehicle_1_first_time

    producer_df_vehicle_1_first_time = producer_df_vehicle_1['Message_Timestamp'].iloc[0]
    producer_df_vehicle_2_first_time = producer_df_vehicle_2['Message_Timestamp'].iloc[0]
    if producer_df_vehicle_1_first_time > producer_df_vehicle_2_first_time:
        producer_diff_1 = producer_df_vehicle_1_first_time - producer_df_vehicle_2_first_time
    elif producer_df_vehicle_2_first_time >= producer_df_vehicle_1_first_time:
        producer_diff_2 = producer_df_vehicle_2_first_time - producer_df_vehicle_1_first_time

    return consumer_df_vehicle_1, consumer_df_vehicle_2, producer_df_vehicle_1, producer_df_vehicle_2, consumer_diff_1, consumer_diff_2, producer_diff_1, producer_diff_2

def extractData(consumer_data, producer_data, consumer_diff, producer_diff):
    #get the first consumer and producer message timestamps as reference points for calculations
    first_consumer_time = consumer_data['Message_Timestamp'].iloc[0]
    first_producer_time = producer_data['Message_Timestamp'].iloc[0]

    #extract stopping time from last column array
    producer_data['Stopping_Time(ms)'] = producer_data['st']

    #adjusted producer timestamp values take into account the first consumed message time
    #divide by 1000 to get values in seconds
    producer_data['Stopping_Time_Adjusted_To_Consumer(s)'] = (producer_data['Stopping_Time(ms)'] - first_consumer_time) / 1000
    producer_data['Message_Timestamp_Adjusted_To_Consumer(s)'] = (producer_data['Message_Timestamp'] + producer_diff - first_producer_time) / 1000

    #get message timestamp relative to first producer time
    producer_data['Message_Timestamp_Adjusted_To_Producer(s)'] = (producer_data['Message_Timestamp'] - first_producer_time) / 1000
    #compare stopping time to message timestamp
    producer_data['Stopping_Time_Diff(s)'] = (producer_data['Stopping_Time(ms)'] - producer_data['Message_Timestamp']) / 1000

    #adjust consumer timestamp values
    consumer_data['Message_Timestamp_Adjusted_To_Consumer(s)'] = (consumer_data['Message_Timestamp'] + consumer_diff - first_consumer_time) / 1000

    #convert speed and calculate acceleration based on speed and time differences
    consumer_data['Speed_Converted(m/s)'] = consumer_data['cur_speed']*0.02
    consumer_data['Accel_Converted(m/s^2)'] = consumer_data['cur_accel']*0.01

    consumer_data['S_And_I_Message_Timediff(s)'] = consumer_data['Message_Timestamp_Adjusted_To_Consumer(s)'].diff()

    consumer_data['Speed_diff(s)'] = consumer_data['Speed_Converted(m/s)'].diff()

    #categorical for is_allowed field
    consumer_data['Access'] = consumer_data['is_allowed'].astype(int)


#returns when the two vehicles were granted access and when they exited the intersection(Adjusted for consumer timestamp)
def getVehicleAccessTimesConsumer(producer_veh_1, producer_veh_2):

    #find first time granted access
    try:
        veh1_first_access_index = producer_veh_1[producer_veh_1['access'] == 1].first_valid_index()
        veh1_first_access_time = producer_veh_1['Message_Timestamp_Adjusted_To_Consumer(s)'].loc[veh1_first_access_index]
        veh1_last_access_index = producer_veh_1[producer_veh_1['access'] == 1].last_valid_index()
        veh1_last_access_time = producer_veh_1['Message_Timestamp_Adjusted_To_Consumer(s)'].loc[veh1_last_access_index]
        print("Vehicle 1 first access " + str(veh1_first_access_time) + " (consumer)")
    except:
        print("Vehicle 1 never received access")
        veh1_first_access_time = 0
        veh1_last_access_time = 0


    #find time vehicle exited
    try:
        veh2_first_access_index = producer_veh_2[producer_veh_2['access'] == 1].first_valid_index()
        veh2_first_access_time = producer_veh_2['Message_Timestamp_Adjusted_To_Consumer(s)'].loc[veh2_first_access_index]
        veh2_last_access_index = producer_veh_2[producer_veh_2['access'] == 1].last_valid_index()
        veh2_last_access_time = producer_veh_2['Message_Timestamp_Adjusted_To_Consumer(s)'].loc[veh2_last_access_index]
        print("Vehicle 2 first access " + str(veh2_first_access_time) + " (consumer)")
    except:
        print("Vehicle 2 never received access")
        veh2_first_access_time = 0
        veh2_last_access_time = 0


    return veh1_first_access_time, veh2_first_access_time, veh1_last_access_time, veh2_last_access_time
#returns when the two vehicles were granted access and when they exited the intersection (Adjusted for producer timestamp)
def getVehicleAccessTimesProducer(producer_veh_1, producer_veh_2):

    #find first time granted access
    try:
        veh1_first_access_index = producer_veh_1[producer_veh_1['access'] == 1].first_valid_index()
        veh1_first_access_time = producer_veh_1['Message_Timestamp_Adjusted_To_Producer(s)'].loc[veh1_first_access_index]
        veh1_last_access_index = producer_veh_1[producer_veh_1['access'] == 1].last_valid_index()
        veh1_last_access_time = producer_veh_1['Message_Timestamp_Adjusted_To_Producer(s)'].loc[veh1_last_access_index]
        print("Vehicle 1 first access " + str(veh1_first_access_time) + " (producer)")
    except:
        print("Vehicle 1 never received access")
        veh1_first_access_time = 0
        veh1_last_access_time = 0


    #find time vehicle exited
    try:
        veh2_first_access_index = producer_veh_2[producer_veh_2['access'] == 1].first_valid_index()
        veh2_first_access_time = producer_veh_2['Message_Timestamp_Adjusted_To_Producer(s)'].loc[veh2_first_access_index]
        veh2_last_access_index = producer_veh_2[producer_veh_2['access'] == 1].last_valid_index()
        veh2_last_access_time = producer_veh_2['Message_Timestamp_Adjusted_To_Producer(s)'].loc[veh2_last_access_index]
        print("Vehicle 2 first access " + str(veh2_first_access_time) + " (producer)")
    except:
        print("Vehicle 2 never received access")
        veh2_first_access_time = 0
        veh2_last_access_time = 0


    return veh1_first_access_time, veh2_first_access_time, veh1_last_access_time, veh2_last_access_time

def runner(filename, vehicle_id_1, vehicle_id_2):
    #using individual consumer/parsed files
    consumer_data = pd.read_csv(f'{input_directory_path}/{filename}_SS_consumer_parsed.csv')
    producer_data = pd.read_csv(f'{input_directory_path}/{filename}_SS_producer_parsed.csv')

    #individual dataframes for the specific vehicles
    consumer_data_vehicle_1, consumer_data_vehicle_2, producer_data_vehicle_1, producer_data_vehicle_2, consumer_diff_1, consumer_diff_2, producer_diff_1, producer_diff_2 = initializeData(consumer_data, producer_data, vehicle_id_1, vehicle_id_2)

    #calculate values of interest
    extractData(consumer_data_vehicle_1, producer_data_vehicle_1, consumer_diff_1, producer_diff_1)
    # consumer_data_vehicle_1.to_csv(f'{output_directory_path}/test1.csv')

    veh_1_status_intent_frequency_averages = averageCalc(consumer_data_vehicle_1, 10, "S_And_I")
    extractData(consumer_data_vehicle_2, producer_data_vehicle_2, consumer_diff_2, producer_diff_2)
    # consumer_data_vehicle_2.to_csv(f'{output_directory_path}/test2.csv')

    veh_2_status_intent_frequency_averages = averageCalc(consumer_data_vehicle_2, 10, "S_And_I")

    veh1_first_access, veh2_first_access, veh1_last_access, veh2_last_access = getVehicleAccessTimesConsumer(producer_data_vehicle_1, producer_data_vehicle_2)

    #plot vehicle stopping distance and access versus time
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Stopping Distance (m)', fontsize=18)
    ax1.plot(consumer_data_vehicle_1['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_1['cur_ds'], color="blue", label=vehicle_id_1+" Stopping Distance", linewidth=2)
    ax1.plot(consumer_data_vehicle_2['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_2['cur_ds'], color="orange", label=vehicle_id_2+" Stopping Distance", linewidth=2, linestyle='--')
    if ( veh1_first_access != 0 ):
        plt.axvline(x=veh1_first_access, linestyle='-.', color="blue", label=vehicle_id_1+" Access")
    if ( veh1_last_access != 0 ):
        plt.axvline(x=veh1_last_access, linestyle='-.', color="blue",)
    if ( veh2_first_access != 0 ):
        plt.axvline(x=veh2_first_access, linestyle=':', color="orange", label=vehicle_id_2+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh2_last_access, linestyle=':', color="orange")
    ax1.tick_params(axis='both', which='major', labelsize=15)
    # Stopping Condition
    plt.axhline(y= constants.CARMA_STREETS_STOP_DISTANCE+constants.VEHICLE_LENGTH_1, linestyle= '--', color="green", label="Distance Stopping Condition", dashes=[10, 5, 20, 5])
    if ( constants.VEHICLE_LENGTH_1 != constants.VEHICLE_LENGTH_2 ):
        plt.axhline(y= constants.CARMA_STREETS_STOP_DISTANCE+constants.VEHICLE_LENGTH_2, linestyle= '-.', color="green", label="Distance Stopping Condition (vehicle 2)" )

    if producer_data_vehicle_2['Message_Timestamp_Adjusted_To_Producer(s)'].max() >= producer_data_vehicle_1['Message_Timestamp_Adjusted_To_Producer(s)'].max():
        plt.xticks(np.arange(0, veh2_last_access+2, 2))
        plt.xlim(0,veh2_last_access+2)
    else:
        plt.xticks(np.arange(0, veh1_last_access+2, 2))
        plt.xlim(0,veh1_last_access+2)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.legend(fontsize=18)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " Stopping Distance/Access Versus Time", fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_Stopping_Distance_Vs_Time.png')

    #plot vehicle speed and access versus time
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.plot(consumer_data_vehicle_1['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_1['Speed_Converted(m/s)'], label=vehicle_id_1+" Speed", color="blue", linewidth=2)
    if ( veh1_first_access != 0 ):
        plt.axvline(x=veh1_first_access, linestyle='--', color="blue", label=vehicle_id_1+" Access")
    if ( veh1_last_access != 0 ):
        plt.axvline(x=veh1_last_access, linestyle='--', color="blue",)
    plt.plot(consumer_data_vehicle_2['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_2['Speed_Converted(m/s)'], label=vehicle_id_2+" Speed", color="orange", linewidth=2, linestyle='-.')
    if ( veh2_first_access != 0 ):
        plt.axvline(x=veh2_first_access, linestyle=':', color="orange", label=vehicle_id_2+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh2_last_access, linestyle=':', color="orange")
    ax1.tick_params(axis='both', which='major', labelsize=15)

    # Stopping Condition
    plt.axhline(y= constants.CARMA_STREETS_STOP_SPEED, linestyle= '--', color="green", label="Speed Stopping Condition" , dashes=[10, 5, 20, 5])
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Speed (m/s)', fontsize=18)

    if producer_data_vehicle_2['Message_Timestamp_Adjusted_To_Producer(s)'].max() >= producer_data_vehicle_1['Message_Timestamp_Adjusted_To_Producer(s)'].max():
        plt.xticks(np.arange(0, veh2_last_access+2, 2))
        plt.xlim(0,veh2_last_access+2)
    else:
        plt.xticks(np.arange(0, veh1_last_access+2, 2))
        plt.xlim(0,veh1_last_access+2)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " Vehicle Speed/Access Versus Time", fontsize=18)
    ax1.legend(loc="upper right", fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_Speed_Vs_Time.png')

    #plot vehicle acceleration vs time
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.plot(consumer_data_vehicle_1['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_1['Accel_Converted(m/s^2)'].astype(float), c="blue", label=vehicle_id_1)
    plt.plot(consumer_data_vehicle_2['Message_Timestamp_Adjusted_To_Consumer(s)'], consumer_data_vehicle_2['Accel_Converted(m/s^2)'].astype(float), c="green", linestyle='dashed', label=vehicle_id_2)
    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Acceleration (m/s^2)', fontsize=18)
    if producer_data_vehicle_2['Message_Timestamp_Adjusted_To_Producer(s)'].max() >= producer_data_vehicle_1['Message_Timestamp_Adjusted_To_Producer(s)'].max():
        plt.xticks(np.arange(0, veh2_last_access+2, 2))
        plt.xlim(0,veh2_last_access+2)
    else:
        plt.xticks(np.arange(0, veh1_last_access+2, 2))
        plt.xlim(0,veh1_last_access+2)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    plt.title("Vehicle 1 and Vehicle 2 BSM Acceleration", fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_Acceleration_vs_Time.png')

    #plot vehicle status and intent frequency
    figure(figsize=(10,10))
    plt.scatter([i[0] for i in veh_1_status_intent_frequency_averages], [i[1] for i in veh_1_status_intent_frequency_averages], c="blue", label=vehicle_id_1)
    plt.scatter([i[0] for i in veh_2_status_intent_frequency_averages], [i[1] for i in veh_2_status_intent_frequency_averages], c="green", marker=',', label=vehicle_id_2)
    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    plt.xlim(0,consumer_data_vehicle_1['Message_Timestamp_Adjusted_To_Consumer(s)'].max())
    plt.ylim(0,20)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " Status and Intent Message Frequency", fontsize=18)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_S_And_I_Message_Frequency.png')

    #plot vehicle state
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.plot( producer_data_vehicle_1["Message_Timestamp_Adjusted_To_Consumer(s)"], producer_data_vehicle_1["State"], label=vehicle_id_1+ " State", color="blue", linewidth=2)
    if ( veh1_first_access != 0 ):
        plt.axvline(x=veh1_first_access, linestyle='--', color="blue", label=vehicle_id_1+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh1_last_access, linestyle='--', color="blue")
    plt.plot( producer_data_vehicle_2["Message_Timestamp_Adjusted_To_Consumer(s)"], producer_data_vehicle_2["State"], label=vehicle_id_2+ " State", color="orange", linewidth=2, linestyle='-.')
    if ( veh2_first_access != 0 ):
        plt.axvline(x=veh2_first_access, linestyle=':', color="orange", label=vehicle_id_2+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh2_last_access, linestyle=':', color="orange")
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('State', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    if producer_data_vehicle_2['Message_Timestamp_Adjusted_To_Consumer(s)'].max() >= producer_data_vehicle_1['Message_Timestamp_Adjusted_To_Consumer(s)'].max():
        plt.xticks(np.arange(0, veh2_last_access+2, 2))
        plt.xlim(0,veh2_last_access+2)
    else:
        plt.xticks(np.arange(0, veh1_last_access+2, 2))
        plt.xlim(0,veh1_last_access+2)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.legend(loc="upper left", fontsize=18)
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " State", fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_State_Vs_Time.png')

    veh1_first_access_pr, veh2_first_access_pr, veh1_last_access_pr, veh2_last_access_pr = getVehicleAccessTimesProducer(producer_data_vehicle_1, producer_data_vehicle_2)

    # Departure position
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.plot( producer_data_vehicle_1["Message_Timestamp_Adjusted_To_Producer(s)"], producer_data_vehicle_1["dp"], label=vehicle_id_1+ " Departure Position", color="blue", linewidth=2)
    if ( veh2_first_access != 0 ):
        plt.axvline(x=veh1_first_access_pr, linestyle='-.', color="blue", label=vehicle_id_2+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh1_last_access_pr, linestyle='-.', color="blue")
    plt.plot( producer_data_vehicle_2["Message_Timestamp_Adjusted_To_Producer(s)"], producer_data_vehicle_2["dp"], label=vehicle_id_2+ " Departure Position", color="orange", linestyle='--', linewidth=2)
    if ( veh2_first_access != 0 ):
        plt.axvline(x=veh2_first_access_pr, linestyle=':', color="orange", label=vehicle_id_2+" Access")
    if ( veh2_last_access != 0 ):
        plt.axvline(x=veh2_last_access_pr, linestyle=':', color="orange")
    ax1.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Departure Position', fontsize=18)
    plt.ylim(0,5)

    if producer_data_vehicle_2['Message_Timestamp_Adjusted_To_Producer(s)'].max() >= producer_data_vehicle_1['Message_Timestamp_Adjusted_To_Producer(s)'].max():
        plt.xticks(np.arange(0, veh2_last_access+2, 2))
        plt.xlim(0,veh2_last_access_pr+2)
    else:
        plt.xticks(np.arange(0, veh1_last_access+2, 2))
        plt.xlim(0,veh1_last_access_pr+2)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.title(vehicle_id_1 + " and " + vehicle_id_2 + " Departure Position", fontsize=18)
    ax1.legend(loc="upper right", fontsize=18)
    plt.savefig(f'{output_directory_path}/{filename}_Departure_Position_Vs_Time.png')