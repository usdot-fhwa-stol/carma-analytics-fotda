"""This script is used to extract the timestamp and payload for every packet sent and received by the Carma platform OBU. The raw data
is in the form of a pcap file and is extracted using the Wireshark command line tool "tshark". """
import sys
import os
import constants
import json
import pandas as pd
import sys
import shutil

#writes the timestamp and whole pcap hex to a csv
def convert_pcap_to_csv(pcapFile):
    pcap_directory_path = f'{constants.DATA_DIR}/{constants.RAW_PCAP_DIR}'
    tshark_directory_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'

    all_in_files = os.listdir(pcap_directory_path)

    for file in all_in_files:
        if pcapFile in file:
            abs_filename = file.split(".")[0] # gets the name of file by removing pcap suffix
            status = os.system(f'tshark -r {pcap_directory_path}/{file} \
                                --disable-protocol wsmp \
                                    --disable-protocol fp \
                                        --disable-protocol udp \
                                            --disable-protocol skype \
                                                --disable-protocol dvb-s2_gse -Tfields -Eseparator=, \
                                                    -e frame.time_epoch -e data.data > {tshark_directory_path}/tshark_{abs_filename}.csv')

def rowHelper(row):
    substrings_to_parse = [constants.BSM_IDENTIFIER, constants.MOM_IDENTIFIER, constants.MPM_IDENTIFIER]

    bsm_index = 10000
    mom_index = 10000
    mpm_index = 10000
    message = ""
    
    #check for bsm
    if substrings_to_parse[0] in str(row[1]):
        bsm_index = row[1].find(substrings_to_parse[0])
    #check for mom
    elif substrings_to_parse[1] in str(row[1]):
        mom_index = row[1].find(substrings_to_parse[1])
    #check for mpm
    elif substrings_to_parse[2] in str(row[1]):
        mpm_index = row[1].find(substrings_to_parse[2])

    return [bsm_index, mom_index, mpm_index]

#creates csv with timestamp and only the J2735 payload
def get_payload_bsm(row):
    indices = rowHelper(row)
    bsm = ""
    if indices[0] < indices[1] and indices[0] < indices[2]:
        bsm = row[1][indices[0]:]

    return bsm

def get_payload_mom(row):
    indices = rowHelper(row)
    mom = ""
    if indices[1] < indices[0] and indices[1] < indices[2]:
        mom = row[1][indices[1]:]

    return mom

def get_payload_mpm(row):
    indices = rowHelper(row)
    mpm = ""
    if indices[2] < indices[1] and indices[2] < indices[0]:
        mpm = row[1][indices[2]:]

    return mpm
    
#gets the payload string from each row and decodes the desired fields
def payloadHelper(pcapFile):
    tshark_directory_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'
    payload_directory_path = f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}'

    for filename in os.listdir(tshark_directory_path):
        searchName = pcapFile.split(".")[0]
        if searchName in filename:
            abs_filename = filename.split('.csv')[0]

            #read the tshark file and create a file containing timestamps and payloads
            df_tshark_output = pd.read_csv(f'{tshark_directory_path}/{filename}')
            df_data_bsm = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_bsm(row), axis=1)}
            df_data_mom = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_mom(row), axis=1)}
            df_data_mpm = {'Timestamp(ms)': (df_tshark_output.iloc[:,0]*1000), 'payload': df_tshark_output.apply(lambda row: get_payload_mpm(row), axis=1)}

            payload_bsm = pd.DataFrame(data=df_data_bsm)
            payload_mom = pd.DataFrame(data=df_data_mom)
            payload_mpm = pd.DataFrame(data=df_data_mpm)

            payload_bsm.dropna(subset=['payload'], inplace=True)
            payload_mom.dropna(subset=['payload'], inplace=True)
            payload_mpm.dropna(subset=['payload'], inplace=True)
            payload_bsm.to_csv(f'{payload_directory_path}/{abs_filename}_payload_bsm.csv', index=False)
            payload_mom.to_csv(f'{payload_directory_path}/{abs_filename}_payload_mom.csv', index=False)
            payload_mpm.to_csv(f'{payload_directory_path}/{abs_filename}_payload_mpm.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 platform_obu_pcap_parser.py" pcapFileName')
    else:
        pcapFile = sys.argv[1]

        convert_pcap_to_csv(pcapFile)
        payloadHelper(pcapFile)
        