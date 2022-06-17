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
pd.options.mode.chained_assignment = None

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}')

def combiner():
    text_directory_path = f'{constants.DATA_DIR}/{constants.RAW_TEXT_DIR}'
    docker_log_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    merged_directory_path = f'{constants.DATA_DIR}/{constants.MERGED_DIR}'
    final_directory_path = f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}'
    external_object_directory_path = f'{constants.DATA_DIR}/{constants.RVIZ_PSM_DIR}'

    external_object_timestamps = pd.read_csv(f'{external_object_directory_path}/All_timestamps_converted_removed_empty.csv')

    test_timestamps = pd.read_csv(f'{text_directory_path}/CP_Test_timestamps_converted.csv')
    tests_day = test_timestamps[test_timestamps['Date'] == date]
    tests = tests_day['Test'].unique().astype(int)

    #iterate through tests on a specific day
    for testnum in tests:
        for i in range(1,9):
            try:
                print("Combining Test: " + str(testnum) + " Trial: " + str(i))
                #need the docker logs,merged file, and external object list for the test/trial
                dockerLog = pd.read_csv(f'{docker_log_directory_path}/Test_{int(testnum)}_Trial_{i}_docker_log_parsed.csv')
                mergedFile = pd.read_csv(f'{merged_directory_path}/Test_{int(testnum)}_Trial_{i}_merged.csv')
                external_object_subset = external_object_timestamps[(external_object_timestamps['Test_Case'] == int(testnum)) & (external_object_timestamps['Trial'] == int(i))]

                #rename these two columns so merge asof can be performed
                dockerLog.rename({'V2XHub_PSM_Tx_Time_converted':'PSM_Tx_Time'}, axis=1, inplace=True)
                mergedFile.rename({'timestamp_converted_tx':'PSM_Tx_Time'}, axis=1, inplace=True)

                #merge the docker log csv with the rsu-obu merged file, on closest V2xhub psm tx/RSU tx time
                temp = pd.merge_asof(dockerLog,mergedFile,on='PSM_Tx_Time',direction='nearest',allow_exact_matches=False)
                temp.drop(['V2XHub_Ped_Rx_Time', 'V2XHub_PSM_Tx_Time', 'payload', 'timestamp_rx'], axis=1, inplace=True)
                #conversion from GMT
                temp['RSU_Tx_Time'] = temp['timestamp_tx'] - 14400
                temp.drop(['timestamp_tx'], axis=1, inplace=True)
                temp.rename({'timestamp_converted_rx':'Carma_OBU_Rx_Time', 'PSM_Tx_Time':'V2XHub_PSM_Create_Time', 'V2XHub_Ped_Rx_Time_converted':'V2XHub_FLIR_Rx_Time'}, axis=1, inplace=True)

                # merge the above created file with the external object timestamps file
                # create time_to_match column to merge on
                # converting the carma obu rx time to a datetime for closest match on external object timestamps
                external_object_subset['time_to_match'] = pd.to_datetime(external_object_subset['External_Object_timestamp'])
                temp['time_to_match'] = pd.to_datetime(temp['Carma_OBU_Rx_Time'], unit='s')
                temp['time_to_match'].fillna(method='ffill', inplace=True)

                #merge the above created dataframe with the PSM timestamps for the test/trial
                #merge on carma obu rx time/psm external_object generation time
                #with additional "incoming psm timestamp" field, now need to drop rows where external object is not created
                external_object_subset.sort_values(by=['External_Object_timestamp_converted'], inplace=True)
                # external_object_subset.drop_duplicates(subset=['External_Object_timestamp_converted'], keep='first', inplace=True)
                external_object_subset.drop(external_object_subset[external_object_subset['Incoming_psm_timestamp_converted'] > external_object_subset['External_Object_timestamp_converted']].index, inplace=True)

                external_object_subset.to_csv('external_object.csv')
                final = pd.merge_asof(temp,external_object_subset,on='time_to_match',direction='forward',allow_exact_matches=False)
                #need to drop duplicates caused by above forward fill
                final.drop_duplicates(subset=['time_to_match'], keep='first', inplace=True)

                final.drop(['Msg_ID', 'Encoded_timestamp', 'psm_decoded_datetime', 'time_to_match', 'Speed', 'Speed_Converted'], axis=1, inplace=True)
                final.rename({'Speed_Converted_Rounded':'External_Object_Speed'}, axis=1, inplace=True)

                # set external object timestamps to nan if carma obu did not receive psm (need to do this bc of ffill and merge)
                final.loc[final["Carma_OBU_Rx_Time"].isnull(),'External_Object_timestamp'] = np.NaN

                #convert flir ped datetime and external object datetime to time since epoch
                final['FLIR_Ped_Time_converted'] = pd.to_datetime(final['FLIR_Ped_Time']).map(pd.Timestamp.timestamp, na_action='ignore')
                final['External_Object_timestamp_converted'] = pd.to_datetime(final['External_Object_timestamp']).map(pd.Timestamp.timestamp, na_action='ignore')

                final.drop(['FLIR_Ped_Time', 'External_Object_timestamp'], axis=1, inplace=True)
                final.rename({'FLIR_Ped_Time_converted':'FLIR_Ped_Time'}, axis=1, inplace=True)
                final['FLIR_Ped_V2XHub_Diff(s)'] = final['V2XHub_FLIR_Rx_Time'] - final['FLIR_Ped_Time']

                final['PSM_Count_Diff'] = final['PSM_Count'].diff()
                #need to do this because PSM count is from 0 to 127
                final['PSM_Count_Diff']= final['PSM_Count_Diff'].replace(to_replace=-127,value=1)
                final['V2XHub_RSU_Communication(s)'] = final['RSU_Tx_Time'] - final['V2XHub_PSM_Create_Time']
                final['V2XHub_PSM_TTI(s)'] = final['V2XHub_PSM_Create_Time'].diff()
                final['V2XHub_FLIR_IPG(s)'] = final['V2XHub_FLIR_Rx_Time'].diff()
                final['RSU_TTI(s)'] = final['RSU_Tx_Time'].diff()
                final['OBU_IPG(s)'] = final['Carma_OBU_Rx_Time'].diff()
                final['RSU_OBU_Latency(s)'] = final['Carma_OBU_Rx_Time']-final['RSU_Tx_Time']
                final.to_csv('test.csv')
                final['End_To_End_Latency(s)'] = final['External_Object_timestamp_converted']-final['V2XHub_FLIR_Rx_Time']
                final['V2XHub_Processing_Time(s)'] = final['V2XHub_PSM_Create_Time'] - final['V2XHub_FLIR_Rx_Time']
                final['Platform_Processing_Time(s)'] = final['External_Object_timestamp_converted']-final['Incoming_psm_timestamp_converted']
                final['FLIR_Rx_OBU_Latency(s)'] = final['Carma_OBU_Rx_Time']-final['V2XHub_FLIR_Rx_Time']
                final['FLIR_Rx_Incoming_PSM_Latency(s)'] = final['Incoming_psm_timestamp_converted']-final['V2XHub_FLIR_Rx_Time']

                PER = (1-(final['Carma_OBU_Rx_Time'].dropna().count()/final['RSU_Tx_Time'].count()))*100
                final.loc[final.index[len(final)-1], 'PER'] = PER

                final['Test_Num'] = int(testnum)
                final['Trial_Num'] = i

                #creating two versions of "final" file
                #one with all entries, the other will contain only entries where the PSM speeds match the external object speeds
                final = final[['Test_Num', 'Trial_Num', 'PSM_Count', 'PSM_Count_Diff', 'PSM_Hex', 'FLIR_Ped_Time', 'V2XHub_FLIR_Rx_Time', 'FLIR_Ped_V2XHub_Diff(s)', 'V2XHub_FLIR_IPG(s)', 'V2XHub_PSM_Create_Time', 'V2XHub_Processing_Time(s)',
                'V2XHub_RSU_Communication(s)', 'RSU_Tx_Time', 'Carma_OBU_Rx_Time', 'psm_speed_tx', 'Incoming_psm_timestamp_converted', 'External_Object_timestamp_converted', 'External_Object_Speed',
                'V2XHub_PSM_TTI(s)', 'RSU_TTI(s)', 'OBU_IPG(s)', 'RSU_OBU_Latency(s)', 'FLIR_Rx_OBU_Latency(s)', 'FLIR_Rx_Incoming_PSM_Latency(s)', 'Platform_Processing_Time(s)',
                'End_To_End_Latency(s)', 'PER']]
                final.to_csv(f'{final_directory_path}/Test_{testnum}_Trial_{i}_final.csv', index=False)

                # final.drop_duplicates(subset=['Incoming_psm_timestamp_converted'], keep='last', inplace=True)
                final.drop(final[final['psm_speed_tx'] != final['External_Object_Speed']].index, inplace=True)
                final.loc[final.index[len(final)-1], 'PER'] = PER
                final.to_csv(f'{final_directory_path}/Test_{testnum}_Trial_{i}_final_matched.csv', index=False)
            except:
                print("Error combining for test: " + str(testnum) + " trial: " + str(i))
                continue

def concatFiles():
    final_directory_path = f'{constants.DATA_DIR}/{constants.FINAL_OUTPUT_DIR}'

    all_in_filenames = os.listdir(final_directory_path)
    final_files = []
    final_duplicates_removed_files = []

    for file in all_in_filenames:
        if "final.csv" in file:
            final_files.append(file)
        if "_final_matched.csv" in file:
            final_duplicates_removed_files.append(file)

    matched_files = (pd.read_csv(f'{final_directory_path}/{f}') for f in final_files if f in os.listdir(final_directory_path))
    matched_removed_files = (pd.read_csv(f'{final_directory_path}/{f}') for f in final_duplicates_removed_files if f in os.listdir(final_directory_path))
    concatenated_df = pd.concat(matched_files, ignore_index=True).drop_duplicates()
    concatenated_removed_df = pd.concat(matched_removed_files, ignore_index=True).drop_duplicates()

    out_filename = "All_final.csv"
    concatenated_df.drop_duplicates(subset=['PSM_Hex'], inplace=True)
    concatenated_df.to_csv(f'{final_directory_path}/{out_filename}', index=False)

    out_filename = "All_final_matched.csv"
    concatenated_removed_df.drop_duplicates(subset=['PSM_Hex'], inplace=True)
    concatenated_removed_df.to_csv(f'{final_directory_path}/{out_filename}', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python final_combiner.py 05/23/2022"')
    else:
        date = sys.argv[1]

        # cleaningDirectories()
        combiner()
        # concatFiles()
