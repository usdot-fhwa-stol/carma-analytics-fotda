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

#creates csv with timestamp and only the J2735 payload
def get_payload(row):
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

    if bsm_index < mom_index and bsm_index < mpm_index:
        message = row[1][bsm_index:]
    if mom_index < bsm_index and mom_index < mpm_index:
        message = row[1][mom_index:]
    if mpm_index < bsm_index and mpm_index < mom_index:
        message = row[1][mpm_index:]

    return message

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
            df_data = {'timestamp': df_tshark_output.iloc[:,0], 'payload': df_tshark_output.apply(lambda row: get_payload(row), axis=1)}

            payload = pd.DataFrame(data=df_data)
            payload.dropna(subset=['payload'], inplace=True)
            payload.to_csv(f'{payload_directory_path}/{abs_filename}_payload.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 platform_obu_pcap_parser.py" pcapFileName')
    else:
        pcapFile = sys.argv[1]

        convert_pcap_to_csv(pcapFile)
        payloadHelper(pcapFile)
        