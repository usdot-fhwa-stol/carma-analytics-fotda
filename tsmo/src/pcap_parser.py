"""This script is used to extract the timestamp and payload for every packet sent by the Carma platform OBU/Carma streets RSU. 
The raw data is in the form of a pcap file and is extracted using the Wireshark command line tool "tshark". """
## How to use this script:
""" Run with python3 pcap_parser.py"""
import sys
import os
import constants
import json
import pandas as pd
import sys
import shutil

#This function utilizes tshark to extract each packet's timestamp and full hex payload to a csv
def convert_pcap_to_csv():
    pcap_directory_path = f'{constants.DATA_DIR}/{constants.RAW_PCAP_DIR}'
    tshark_directory_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'

    all_in_files = os.listdir(pcap_directory_path)

    for file in all_in_files:
            abs_filename = file.split(".")[0] # gets the name of file by removing pcap suffix
            status = os.system(f'tshark -r {pcap_directory_path}/{file} \
                                --disable-protocol wsmp \
                                    --disable-protocol fp \
                                        --disable-protocol udp \
                                            --disable-protocol skype \
                                                --disable-protocol dvb-s2_gse -Tfields -Eseparator=, \
                                                    -e frame.time_epoch -e data.data > {tshark_directory_path}/tshark_{abs_filename}.csv')

#This helper method will search through the payload for each of the unique identifiers of the BSM, MPM, and MOM messages. It 
#returns an array of the index where the message identifier was located. If the identifier was not found in the payload, the
#initial set value of the index is returned.
def rowHelper(row):
    substrings_to_parse = [constants.BSM_IDENTIFIER, constants.MOM_IDENTIFIER, constants.MPM_IDENTIFIER, constants.SPAT_IDENTIFIER]

    bsm_index = 10000
    mom_index = 10000
    mpm_index = 10000
    spat_index = 10000

    #check for bsm
    if substrings_to_parse[0] in str(row[1]):
        bsm_index = row[1].find(substrings_to_parse[0])
    #check for mom
    if substrings_to_parse[1] in str(row[1]):
        mom_index = row[1].find(substrings_to_parse[1])
    #check for mpm
    if substrings_to_parse[2] in str(row[1]):
        mpm_index = row[1].find(substrings_to_parse[2])
    #check for spat
    if substrings_to_parse[3] in str(row[1]):
        spat_index = row[1].find(substrings_to_parse[3])

    return [bsm_index, mom_index, mpm_index, spat_index]

#This function uses the rowHelper function to retrieve the indices for the identifiers of the three messages of interest.
#It then compares those indices to each other and if the BSM index is the smallest, it returns the BSM payload.
def get_payload_bsm(row):
    indices = rowHelper(row)
    bsm = ""
    if indices[0] < indices[1] and indices[0] < indices[2] and indices[0] < indices[3]:
        bsm = row[1][indices[0]:]

    return bsm

#This function uses the rowHelper function to retrieve the indices for the identifiers of the three messages of interest.
#It then compares those indices to each other and if the MOM index is the smallest, it returns the MOM payload.
def get_payload_mom(row):
    indices = rowHelper(row)
    mom = ""
    if indices[1] < indices[0] and indices[1] < indices[2] and indices[1] < indices[3]:
        mom = row[1][indices[1]:]

    return mom

#This function uses the rowHelper function to retrieve the indices for the identifiers of the three messages of interest.
#It then compares those indices to each other and if the MPM index is the smallest, it returns the MPM payload.
def get_payload_mpm(row):
    indices = rowHelper(row)
    mpm = ""
    if indices[2] < indices[1] and indices[2] < indices[0] and indices[2] < indices[3]:
        mpm = row[1][indices[2]:]

    return mpm

#This function uses the rowHelper function to retrieve the indices for the identifiers of the three messages of interest.
#It then compares those indices to each other and if the SPAT index is the smallest, it returns the SPAT payload.
def get_payload_spat(row):
    indices = rowHelper(row)

    spat = ""
    if indices[3] < indices[0] and indices[3] < indices[1] and indices[3] < indices[2]:
        spat = row[1][indices[3]:]

    return spat

#This function will read the tshark file and create separate files containing the timestamps and hex payloads of each desired
#message type.
def payloadHelper():
    tshark_directory_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.TIMESTAMP_DIR}'

    for filename in os.listdir(tshark_directory_path):        
        abs_filename = filename.split('.csv')[0]
        abs_filename = abs_filename.replace("tshark_", "")
        
        #read the tshark file and create a file containing timestamps and payloads
        df_tshark_output = pd.read_csv(f'{tshark_directory_path}/{filename}')
        
        if "rsu" in filename:
            df_data_mom = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_mom(row), axis=1)}
            df_data_spat = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_spat(row), axis=1)}

            payload_mom = pd.DataFrame(data=df_data_mom)
            payload_spat = pd.DataFrame(data=df_data_spat)

            # dropping empty rows
            payload_mom = payload_mom[payload_mom["payload"] != ""]
            payload_spat = payload_spat[payload_spat["payload"] != ""]

            payload_mom.to_csv(f'{output_directory_path}/{abs_filename}_mom_timestamps.csv', index=False)
            payload_spat.to_csv(f'{output_directory_path}/{abs_filename}_spat_timestamps.csv', index=False)

        #analyze bsm data if looking at obu tx file
        elif "obu" in filename:
            df_data_bsm = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_bsm(row), axis=1)}
            df_data_mpm = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_mpm(row), axis=1)}
            df_data_mom = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_mom(row), axis=1)}

            payload_bsm = pd.DataFrame(data=df_data_bsm)
            payload_mpm = pd.DataFrame(data=df_data_mpm)
            payload_mom = pd.DataFrame(data=df_data_mom)

            payload_bsm = payload_bsm[payload_bsm["payload"] != ""]
            payload_mpm = payload_mpm[payload_mpm["payload"] != ""]
            payload_mom = payload_mom[payload_mom["payload"] != ""]

            payload_bsm.to_csv(f'{output_directory_path}/{abs_filename}_bsm_timestamps.csv', index=False)
            payload_mpm.to_csv(f'{output_directory_path}/{abs_filename}_mpm_timestamps.csv', index=False)
            payload_mom.to_csv(f'{output_directory_path}/{abs_filename}_mom_timestamps.csv', index=False)



if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with python3 pcap_parser.py')
    else:
        convert_pcap_to_csv()
        payloadHelper()
        