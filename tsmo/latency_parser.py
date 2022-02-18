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
import shutil
import latency_metrics

input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_PCAP_DIR}'
tshark_output_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'
payload_output_path = f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}'
latency_output_path = f'{constants.DATA_DIR}/{constants.LATENCY_DIR}'
plot_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
all_in_directories = os.listdir(input_directory_path)

#logic for merging files together will go here

# txaFileList = []
# txbFileList = []
# rxaFileList = []
# rxbFileList = []
#
# for dirpath, dirs, files in os.walk(input_directory_path):
#     if constants.VEHICLE_ID_1 in dirpath and "2022" in dirpath:
#         veh1_top_dir = dirpath
#     elif constants.VEHICLE_ID_2 in dirpath and "2022" in dirpath:
#         veh2_top_dir = dirpath
#
# veh2_logs = os.listdir(veh2_top_dir)
#
# for file in veh2_logs:
#     if "rxa" in file:
#         rxaFileList.append(file)
#     elif "rxb" in file:
#         rxbFileList.append(file)
#     elif "txa" in file:
#         txaFileList.append(file)
#     elif "txb" in file:
#         txbFileList.append(file)
#
# #mergecap -a RSU_4_1-cw-mon-rxa-20220128195405.pcap RSU_4_1-cw-mon-rxa-20220128195405.pcap1 -w output_file.pcap
# if len(rxaFileList) > 1:
#     mergeString = "mergecap -a "
#     for i in range(0, len(rxaFileList)):
#         mergeString += rxaFileList[i] + " "
#
#     print(mergeString)
#      status = os.system(f'mergecap {mergeString} -w testOutput.pcap')

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.TSHARK_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.TSHARK_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.TSHARK_DIR}')

    if os.path.isdir(f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}')

    if os.path.isdir(f'{constants.DATA_DIR}/{constants.LATENCY_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.LATENCY_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.LATENCY_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.LATENCY_DIR}')

#writes the timestamp and whole pcap hex to a csv
def convert_pcap_to_csv():
    for file in all_in_directories:
        abs_filename = file.split(".")[0] # gets the name of file by removing pcap suffix
        status = os.system(f'tshark -r {input_directory_path}/{file} \
                         --disable-protocol wsmp \
                             --disable-protocol fp \
                                 --disable-protocol udp \
                                     --disable-protocol skype \
                                         --disable-protocol dvb-s2_gse -Tfields -Eseparator=, \
                                             -e frame.time_epoch -e data.data > {tshark_output_path}/tshark_{abs_filename}.csv')

#creates csv with timestamp and only the J2735 payload
def get_payload():
    #Searches for the string that signals the start of the payload
    for filename in os.listdir(tshark_output_path):
        abs_filename = filename.split('.csv')[0]

        #adjust the  substring based on device type
        substring_to_parse = constants.MOBILITY_IDENTIFIER
        if "RSU" in filename and "rx" in filename:
            substring_to_parse = constants.BSM_IDENTIFIER
        elif "OBU" in filename and "tx" in filename:
            substring_to_parse = constants.BSM_IDENTIFIER

        # load the tshark generated csv files as pandas dataframes
        try:
            #assuming that radio B is used for all devices
            if "xb" in filename:
                df_tshark_output = pd.read_csv(f'{tshark_output_path}/{filename}')
                df_tshark_output.columns = ['timestamp', 'payload'] ## add column names

                # parse the payload column to filter substring for output
                df_tshark_output['payload'] = df_tshark_output['payload'].apply(lambda x: x[x.find(substring_to_parse):] if str(x).find(substring_to_parse)>=0 else 'failed')
                df_tshark_output = df_tshark_output[df_tshark_output['payload'] != 'failed'] # drop the records where substring not found
                df_tshark_output.to_csv(f'{payload_output_path}/{abs_filename}_payload.csv', index=False)
        except:
            print("ERROR empty file for: " + filename)

#returns the size of the payload based on J2735 rule (removing any checksums from payload)
def payloadHelper(row, substring):
    idx = row['payload'].find(substring)
    # print(row['payload'], idx, file)
    adjusted_payload = " "

    #check if the substring is in the payload and if it is a valid size
    if (idx > -1 and len(row['payload']) > 20):
        if (int('0x'+row['payload'][idx+4],16)==8):
            lenstr=int('0x'+row['payload'][idx+5:idx+8],16)*2+6
        else:
            lenstr=int('0x'+row['payload'][idx+4:idx+6],16)*2+6
        adjusted_payload = row['payload'][idx:idx+lenstr]

        return adjusted_payload

#create "adjusted" file which removes any checksums from payload
def payload_adjust():
    for file in os.listdir(payload_output_path):
        #adjust the  substring based on device type
        substring_to_parse = constants.MOBILITY_IDENTIFIER
        if "RSU" in file and "rx" in file:
            substring_to_parse = constants.BSM_IDENTIFIER
        elif "OBU" in file and "tx" in file:
            substring_to_parse = constants.BSM_IDENTIFIER

        filename = file.split(".")[0]
        file_data = pd.read_csv(f'{payload_output_path}/{filename}.csv')
        file_data['payload'] = file_data.apply(lambda row: payloadHelper(row, substring_to_parse), axis=1)
        file_data.to_csv(f'{payload_output_path}/{filename}_adjusted.csv', index=False)

#calculate latency between pairs
def latency_calc(start_time, end_time):
    tx_files = []
    rx_files = []

    for filename in os.listdir(payload_output_path):
        if "txb" in filename and "adjusted" in filename:
            tx_files.append(filename)
        elif "rxb" in filename and "adjusted" in filename:
            rx_files.append(filename)

    for tx in tx_files:
        tx_device = tx.split("_")[2]
        df_tx = pd.read_csv(f'{payload_output_path}/{tx}')
        tx_subset = df_tx[(df_tx['timestamp'] >= start_time)&(df_tx['timestamp'] <= end_time)]
        for rx in rx_files:
            rx_device = rx.split("_")[2]
            if tx_device != rx_device:
                df_rx = pd.read_csv(f'{payload_output_path}/{rx}')
                rx_subset = df_rx[(df_rx['timestamp'] >= start_time)&(df_rx['timestamp'] <= end_time)]

                merged_df = tx_subset.merge(rx_subset, how='left', on='payload', suffixes=('_tx', '_rx'))
                latency_df = merged_df['timestamp_rx']-merged_df['timestamp_tx']
                final = pd.concat([merged_df, latency_df], axis=1)
                final.insert(0, "txdevice", tx_device)
                final.insert(1, "rxdevice", rx_device)
                final.columns=["txdevice", "rxdevice", "tx_time","payload", "rx_time", "latency(s)"]

                out_filename = tx_device + "_tx_" + rx_device + "_rx_merged.csv"
                final.to_csv(f'{latency_output_path}/{out_filename}', index=False)

def runner(first_time, last_time):
    # cleaningDirectories()
    # convert_pcap_to_csv()
    # get_payload()
    # payload_adjust()
    # latency_calc(first_time, last_time)
    latency_metrics.plotter()
