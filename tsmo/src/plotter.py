#This script uses all previously generated analysis csv files and plots several metrics of interest for each test case.
import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
pd.options.mode.chained_assignment = None

def metrics(testnum):
    merged_directory_path = f'{constants.DATA_DIR}/{constants.MERGED_DIR}'
    docker_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    final_directory_path = f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}'
    plot_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'

    all_final = pd.read_csv(f'{final_directory_path}/All_final.csv')
    #all dsrc file used so that metrics dont get skewed by purposely skipped rows in the "final" files
    all_dsrc = pd.read_csv(f'{merged_directory_path}/All_merged.csv')
    all_docker = pd.read_csv(f'{docker_directory_path}/All_docker_parsed.csv')

    all_final_subset = all_final[all_final['Test_Num'] == int(testnum)]
    all_dsrc_subset = all_dsrc[all_dsrc['Test_Num'] == int(testnum)]
    all_dsrc_subset = all_dsrc_subset.iloc[1:]
    all_docker_subset = all_docker[all_docker['Test_Num'] == int(testnum)]

    #need to drop bc of difference in times for changing trials
    all_docker_subset['rx_diff'] = all_docker_subset['V2XHub_Ped_Rx_Time_converted'].astype(float).diff()
    all_docker_subset['tx_diff'] = all_docker_subset['V2XHub_PSM_Tx_Time_converted'].astype(float).diff()
    all_docker_subset.drop(all_docker_subset[(all_docker_subset['rx_diff'].astype(float).diff() > 1)|(all_docker_subset['rx_diff'].astype(float).diff()< 0)].index, inplace=True)


    with open(f'{plot_directory_path}/Test_{testnum}_Metrics.csv', 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)

        csv_writer.writerow(["V2X_Hub_Rx_Interval_Sensor_Data_Mean(s)", "V2X_Hub_Rx_Interval_Sensor_Data_Std(s)", "V2X_Hub_Rx_Interval_Sensor_Data_Min(s)", "V2X_Hub_Rx_Interval_Sensor_Data_25th_Percentile(s)",
        "V2X_Hub_Rx_Interval_Sensor_Data_50th_Percentile(s)", "V2X_Hub_Rx_Interval_Sensor_Data_75th_Percentile(s)", "V2X_Hub_Rx_Interval_Sensor_Data_Max(s)", "V2X_Hub_Rx_Interval_Sensor_Data_Mean_Freq(Hz)"])
        flir_rx_freq = 1/all_docker_subset['rx_diff'].astype(float).mean()
        csv_writer.writerow([all_docker_subset['rx_diff'].mean(), all_docker_subset['rx_diff'].std(), all_docker_subset['rx_diff'].min(), all_docker_subset['rx_diff'].quantile(0.25),
        all_docker_subset['rx_diff'].quantile(0.50), all_docker_subset['rx_diff'].quantile(0.75), all_docker_subset['rx_diff'].max(), flir_rx_freq])
        csv_writer.writerow(" ")

        csv_writer.writerow(["V2X_Hub_Processing_Time_Mean(s)", "V2X_Hub_Processing_Time_Std(s)", "V2X_Hub_Processing_Time_Min(s)", "V2X_Hub_Processing_Time_25th_Percentile(s)",
        "V2X_Hub_Processing_Time_50th_Percentile(s)", "V2X_Hub_Processing_Time_75th_Percentile(s)", "V2X_Hub_Processing_Time_Max(s)"])
        csv_writer.writerow([all_docker_subset['V2XHub_Processing_Time(s)'].mean(), all_docker_subset['V2XHub_Processing_Time(s)'].std(), all_docker_subset['V2XHub_Processing_Time(s)'].min(), all_docker_subset['V2XHub_Processing_Time(s)'].quantile(0.25),
        all_docker_subset['V2XHub_Processing_Time(s)'].quantile(0.50), all_docker_subset['V2XHub_Processing_Time(s)'].quantile(0.75), all_docker_subset['V2XHub_Processing_Time(s)'].max()])
        csv_writer.writerow(" ")

        csv_writer.writerow(["V2XHub_PSM_TTI_Mean(s)", "V2XHub_PSM_TTI_Std(s)", "V2XHub_PSM_TTI_Min(s)", "V2XHub_PSM_TTI_25th_Percentile(s)",
        "V2XHub_PSM_TTI_50th_Percentile(s)", "V2XHub_PSM_TTI_75th_Percentile(s)", "V2XHub_PSM_TTI_Max(s)", "V2XHub_PSM_TTI_Mean_Freq(Hz)"])
        v2xhub_psm_freq = 1/all_docker_subset['tx_diff'].mean()
        csv_writer.writerow([all_docker_subset['tx_diff'].mean(), all_docker_subset['tx_diff'].std(), all_docker_subset['tx_diff'].min(), all_docker_subset['tx_diff'].quantile(0.25),
        all_docker_subset['tx_diff'].quantile(0.50), all_docker_subset['tx_diff'].quantile(0.75), all_docker_subset['tx_diff'].max(), v2xhub_psm_freq])
        csv_writer.writerow(" ")

        csv_writer.writerow(["RSU_TTI_Mean(s)", "RSU_TTI_Std(s)", "RSU_TTI_Min(s)", "RSU_TTI_25th_Percentile(s)",
        "RSU_TTI_50th_Percentile(s)", "RSU_TTI_75th_Percentile(s)", "RSU_TTI_Max(s)", "RSU_TTI_Mean_Freq(Hz)"])
        rsu_psm_freq = 1/all_dsrc_subset['TTI'].mean()
        csv_writer.writerow([all_dsrc_subset['TTI'].mean(), all_dsrc_subset['TTI'].std(), all_dsrc_subset['TTI'].min(), all_dsrc_subset['TTI'].quantile(0.25),
        all_dsrc_subset['TTI'].quantile(0.50), all_dsrc_subset['TTI'].quantile(0.75), all_dsrc_subset['TTI'].max(), rsu_psm_freq])
        csv_writer.writerow(" ")

        csv_writer.writerow(["CARMA_OBU_IPG_Mean(s)", "CARMA_OBU_IPG_Std(s)", "CARMA_OBU_IPG_Min(s)", "CARMA_OBU_IPG_25th_Percentile(s)",
        "CARMA_OBU_IPG_50th_Percentile(s)", "CARMA_OBU_IPG_75th_Percentile(s)", "CARMA_OBU_IPG_Max(s)", "CARMA_OBU_IPG_Mean_Freq(Hz)"])
        obu_psm_freq = 1/all_dsrc_subset['IPG'].mean()
        csv_writer.writerow([all_dsrc_subset['IPG'].mean(), all_dsrc_subset['IPG'].std(), all_dsrc_subset['IPG'].min(), all_dsrc_subset['IPG'].quantile(0.25),
        all_dsrc_subset['IPG'].quantile(0.50), all_dsrc_subset['IPG'].quantile(0.75), all_dsrc_subset['IPG'].max(), obu_psm_freq])
        csv_writer.writerow(" ")

        csv_writer.writerow(["RSU_OBU_Latency_Mean(s)", "RSU_OBU_Latency_Std(s)", "RSU_OBU_Latency_Min(s)", "RSU_OBU_Latency_25th_Percentile(s)",
        "RSU_OBU_Latency_50th_Percentile(s)", "RSU_OBU_Latency_75th_Percentile(s)", "RSU_OBU_Latency_Max(s)"])
        csv_writer.writerow([all_dsrc_subset['Latency'].mean(), all_dsrc_subset['Latency'].std(), all_dsrc_subset['Latency'].min(), all_dsrc_subset['Latency'].quantile(0.25),
        all_dsrc_subset['Latency'].quantile(0.50), all_dsrc_subset['Latency'].quantile(0.75), all_dsrc_subset['Latency'].max()])
        csv_writer.writerow(" ")


        #------------------for flir camera vs v2x hub offset calcullation----------------------
        csv_writer.writerow(["Sensor_V2XHub_Time_Offset_Mean(s)", "Sensor_V2XHub_Time_Offset_Std(s)", "Sensor_V2XHub_Time_Offset_Min(s)", "Sensor_V2XHub_Time_Offset_25th_Percentile(s)",
        "Sensor_V2XHub_Time_Offset_50th_Percentile(s)", "Sensor_V2XHub_Time_Offset_75th_Percentile(s)", "Sensor_V2XHub_Time_Offset_Max(s)"])
        csv_writer.writerow([all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].mean(), all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].std(), all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].min(), all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].quantile(0.25),
        all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].quantile(0.50), all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].quantile(0.75), all_final_subset['FLIR_Ped_V2XHub_Diff(s)'].max()])
        csv_writer.writerow(" ")
        #----------------------------------------------------------------------------------------

        csv_writer.writerow(["V2X_Hub_Create_RSU_Tx_Mean(s)", "V2X_Hub_Create_RSU_Tx_Std(s)", "V2X_Hub_Create_RSU_Tx_Min(s)", "V2X_Hub_Create_RSU_Tx_25th_Percentile(s)",
        "V2X_Hub_Create_RSU_Tx_50th_Percentile(s)", "V2X_Hub_Create_RSU_Tx_75th_Percentile(s)", "V2X_Hub_Create_RSU_Tx_Max(s)"])
        csv_writer.writerow([all_final_subset['V2XHub_RSU_Communication(s)'].mean(), all_final_subset['V2XHub_RSU_Communication(s)'].std(), all_final_subset['V2XHub_RSU_Communication(s)'].min(), all_final_subset['V2XHub_RSU_Communication(s)'].quantile(0.25),
        all_final_subset['V2XHub_RSU_Communication(s)'].quantile(0.50), all_final_subset['V2XHub_RSU_Communication(s)'].quantile(0.75), all_final_subset['V2XHub_RSU_Communication(s)'].max()])
        csv_writer.writerow(" ")

        csv_writer.writerow(["V2XHub_Rx_OBU_Latency_Mean(s)", "V2XHub_Rx_OBU_Latency_Std(s)", "V2XHub_Rx_OBU_Latency_Min(s)", "V2XHub_Rx_OBU_Latency_25th_Percentile(s)",
        "V2XHub_Rx_OBU_Latency_50th_Percentile(s)", "V2XHub_Rx_OBU_Latency_75th_Percentile(s)", "V2XHub_Rx_OBU_Latency_Max(s)"])
        csv_writer.writerow([all_final_subset['FLIR_Rx_OBU_Latency(s)'].mean(), all_final_subset['FLIR_Rx_OBU_Latency(s)'].std(), all_final_subset['FLIR_Rx_OBU_Latency(s)'].min(), all_final_subset['FLIR_Rx_OBU_Latency(s)'].quantile(0.25),
        all_final_subset['FLIR_Rx_OBU_Latency(s)'].quantile(0.50), all_final_subset['FLIR_Rx_OBU_Latency(s)'].quantile(0.75), all_final_subset['FLIR_Rx_OBU_Latency(s)'].max()])
        csv_writer.writerow(" ")

        PER = all_dsrc_subset['PER'].dropna()
        csv_writer.writerow(["PER_Mean", "PER_Min", "PER_Max"])
        csv_writer.writerow([all_dsrc_subset['PER'].mean(), all_dsrc_subset['PER'].min(), all_dsrc_subset['PER'].max()])


def plot(testnum):
    final_directory_path = f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}'
    plot_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
    merged_directory_path = f'{constants.DATA_DIR}/{constants.MERGED_DIR}'
    docker_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'

    all_final_removed = pd.read_csv(f'{final_directory_path}/All_final_matched.csv')
    #all dsrc file used so that metrics dont get skewed by purposely skipped rows in the "final" files
    all_dsrc = pd.read_csv(f'{merged_directory_path}/All_merged.csv')
    all_docker = pd.read_csv(f'{docker_directory_path}/All_docker_parsed.csv')

    all_final_removed.to_csv('docker.csv')

    #plot FLIR data receive rate
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_docker[(all_docker['Test_Num'] == int(testnum)) & (all_docker['Trial_Num'] == i)]
            subset['rx_diff'] = subset['V2XHub_Ped_Rx_Time_converted'].astype(float).diff()

            test_start = subset['V2XHub_Ped_Rx_Time_converted'].iloc[0]
            subset['Test_time'] = subset['V2XHub_Ped_Rx_Time_converted'] - test_start
            plt.scatter(subset['Test_time'], subset['rx_diff'], c="blue")
        except:
            print("No flir rx data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('V2X Hub Receive Interval of Sensor Data (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['rx_diff'].quantile(0.75) + 0.1)
    plt.title("Test " + str(test_num) + " V2X Hub Receive Interval of Sensor Data", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_1_V2X_Hub_Rx_Interval"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot V2X Hub processing time
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_docker[(all_docker['Test_Num'] == int(testnum)) & (all_docker['Trial_Num'] == i)]
            test_start = subset['V2XHub_Ped_Rx_Time_converted'].iloc[0]
            subset['Test_time'] = subset['V2XHub_Ped_Rx_Time_converted'] - test_start
            plt.scatter(subset['Test_time'], subset['V2XHub_Processing_Time(s)'], c="blue")
        except:
            print("No v2x hub processing data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Time to Generate PSM (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['V2XHub_Processing_Time(s)'].quantile(0.75)+0.004)
    plt.title("Test " + str(test_num) + " V2X Hub PSM Processing Time", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_5_V2X_Hub_PSM_Processing_Time"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot V2X Hub TTI
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_docker[(all_docker['Test_Num'] == int(testnum)) & (all_docker['Trial_Num'] == i)]
            subset['tx_diff'] = subset['V2XHub_PSM_Tx_Time_converted'].astype(float).diff()

            test_start = subset['V2XHub_Ped_Rx_Time_converted'].iloc[0]
            subset['Test_time'] = subset['V2XHub_Ped_Rx_Time_converted'] - test_start
            plt.scatter(subset['Test_time'], subset['tx_diff'], c="blue")
        except:
            print("No V2X Hub TTI data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('V2X Hub PSM TTI (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['tx_diff'].quantile(0.75) + 0.1)
    plt.title("Test " + str(test_num) + " V2X Hub PSM TTI", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_2_V2X_Hub_PSM_TTI"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot RSU TTI
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_dsrc[(all_dsrc['Test_Num'] == int(testnum)) & (all_dsrc['Trial_Num'] == i)]
            test_start = subset['timestamp_converted_tx'].iloc[0]
            subset['Test_time'] = subset['timestamp_converted_tx'] - test_start
            plt.scatter(subset['Test_time'], subset['TTI'], c="blue")
        except:
            print("No RSU TTI data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('RSU TTI (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['TTI'].quantile(0.75)+0.2)
    plt.title("Test " + str(test_num) + " RSU TTI", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_3_RSU_TTI"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot OBU IPG
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_dsrc[(all_dsrc['Test_Num'] == int(testnum)) & (all_dsrc['Trial_Num'] == i)]
            test_start = subset['timestamp_converted_rx'].iloc[0]
            subset['Test_time'] = subset['timestamp_converted_rx'] - test_start
            plt.scatter(subset['Test_time'], subset['IPG'], c="blue")
        except:
            print("No OBU IPG data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('OBU Rx Interval (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['IPG'].quantile(0.75)+0.2)
    plt.title("Test " + str(test_num) + " OBU Receive Interval", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_4_OBU_Rx"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot RSU OBU Latency
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_dsrc[(all_dsrc['Test_Num'] == int(testnum)) & (all_dsrc['Trial_Num'] == i)]
            test_start = subset['timestamp_converted_rx'].iloc[0]
            subset['Test_time'] = subset['timestamp_converted_rx'] - test_start
            plt.scatter(subset['Test_time'], subset['Latency'], c="blue")
        except:
            print("No OBU IPG data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('RSU-OBU Latency (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['Latency'].quantile(0.75)+0.005)
    plt.title("Test " + str(test_num) + " RSU-OBU Latency", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_7_RSU_OBU_Latency"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot flir obu processing time
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_final_removed[(all_final_removed['Test_Num'] == int(testnum)) & (all_final_removed['Trial_Num'] == i)]
            test_start = subset['V2XHub_PSM_Create_Time'].iloc[0]
            subset['Test_time'] = subset['V2XHub_PSM_Create_Time'] - test_start
            plt.scatter(subset['Test_time'], subset['FLIR_Rx_OBU_Latency(s)'], c="blue")
        except:
            print("No flir rx obu latency data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('V2X Hub Rx -> OBU Latency (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['FLIR_Rx_OBU_Latency(s)'].quantile(0.75)+0.01)
    plt.title("Test " + str(test_num) + " V2X Hub Rx -> Carma OBU Latency", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_8_V2XHub_Rx_OBU_Latency"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot V2X Hub PSM create to RSU Tx
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_final_removed[(all_final_removed['Test_Num'] == int(testnum)) & (all_final_removed['Trial_Num'] == i)]
            test_start = subset['V2XHub_PSM_Create_Time'].iloc[0]
            subset['Test_time'] = subset['V2XHub_PSM_Create_Time'] - test_start
            plt.scatter(subset['Test_time'], subset['V2XHub_RSU_Communication(s)'], c="blue")
        except:
            print("No v2xhub rsu communication data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('V2X Hub PSM Create-RSU Tx (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['V2XHub_RSU_Communication(s)'].quantile(0.75)+0.01)
    plt.title("Test " + str(test_num) + " V2X Hub PSM Create-RSU Tx", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_6_V2X_Hub_Create_RSU_Tx"
    plt.savefig(f'{plot_directory_path}/{filename}.png')

    #plot difference in flir and v2xhub clocks
    figure(figsize=(10,10))
    for i in range(1,9):
        try:
            subset = all_final_removed[(all_final_removed['Test_Num'] == int(testnum)) & (all_final_removed['Trial_Num'] == i)]
            test_start = subset['V2XHub_PSM_Create_Time'].iloc[0]
            subset['Test_time'] = subset['V2XHub_PSM_Create_Time'] - test_start
            plt.scatter(subset['Test_time'], subset['FLIR_Ped_V2XHub_Diff(s)'], c="blue")
        except:
            print("No flir v2xhub offset data for test: " + str(testnum) + " trial: " + str(i))

    plt.xlabel('Test Time (s)', fontsize=18)
    plt.ylabel('Sensor - V2X Hub Time Offset (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,subset['FLIR_Ped_V2XHub_Diff(s)'].quantile(0.75) + 0.01)
    plt.title("Test " + str(test_num) + " Sensor - V2X Hub Time Offset", fontsize=18)
    plt.grid(True)
    filename = "Test_" + str(test_num) + "_Plot_9_Sensor_V2XHub_Time_Offset"
    plt.savefig(f'{plot_directory_path}/{filename}.png')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python plotter.py testnum"')
    else:
        test_num = sys.argv[1]
        metrics(test_num)
        plot(test_num)
