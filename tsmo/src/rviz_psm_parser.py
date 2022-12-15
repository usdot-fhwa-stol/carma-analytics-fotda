#This script is used to convert the external object timestamps created on Carma Platform to epoch times. These timestamps
#will be used for further analysis. The external object timestamps are part of the raw test data and are contained in a csv
#file for each test case.
import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import numpy as np
import sys
import shutil

rviz_directory_path = f'{constants.DATA_DIR}/{constants.RVIZ_PSM_DIR}'

#This function will concat all of the individual object timestamp files into one file.
def concatFiles():
    all_in_filenames = os.listdir(rviz_directory_path)
    final_files = []

    for file in all_in_filenames:
        if "Test" in file:
            final_files.append(file)

    matched_files = (pd.read_csv(f'{rviz_directory_path}/{f}') for f in final_files if f in os.listdir(rviz_directory_path))
    concatenated_df = pd.concat(matched_files, ignore_index=True).drop_duplicates()
    out_filename = "All_timestamps.csv"
    concatenated_df.rename({'Test Case':'Test_Case', 'Msg ID':'Msg_ID', 'External Object timestamp':'External_Object_timestamp',
    'Incoming psm timestamp': 'Incoming_psm_timestamp', 'Encoded timestamp':'Encoded_timestamp'}, axis=1, inplace=True)

    
    concatenated_df.to_csv(f'{rviz_directory_path}/{out_filename}', index=False)

#the platform logs give object speed in m/s, need to convert to 0.02 m/s units that are encoded in PSM
def speedHelper(row):
    final = ""
    try:
        final = row['Speed'] * 50 #convert to 0.02 m/s
    except:
        print("error with row: " + str(row))

    return final

#Convert timestamps to epoch times and perform unit conversion for speed values
def converter():
    all_timestamps = pd.read_csv(f'{rviz_directory_path}/All_timestamps.csv')
    all_timestamps['Speed_Converted'] = all_timestamps.apply(lambda row: speedHelper(row), axis=1)
    all_timestamps['Speed_Converted'].dropna(inplace=True)
    #round to get whole number values of speed
    all_timestamps['Speed_Converted_Rounded'] = all_timestamps['Speed_Converted'].round().astype(int)
    #convert incoming psm/external object datetime to time since epoch
    all_timestamps['External_Object_timestamp_converted'] = pd.to_datetime(all_timestamps['External_Object_timestamp']).map(pd.Timestamp.timestamp, na_action='ignore')
    
    all_timestamps['Incoming_psm_timestamp_converted'] = pd.to_datetime(all_timestamps['Incoming_psm_timestamp']).map(pd.Timestamp.timestamp, na_action='ignore')
    all_timestamps.to_csv(f'{rviz_directory_path}/All_timestamps_converted.csv', index=False)

    #now keeping two versions of the file, one with empty external objects and one without
    all_timestamps.dropna(subset=['External_Object_timestamp_converted'], axis=0, inplace=True)
    all_timestamps.to_csv(f'{rviz_directory_path}/All_timestamps_converted_removed_empty.csv', index=False)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python3 rviz_psm_parser.py"')
    else:
        concatFiles()
        converter()
