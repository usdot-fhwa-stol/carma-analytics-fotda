#This script is used to extract the timestamp and payload for every packet sent and received by both the OBU and RSU. The raw data
#is in the form of a pcap file and is extracted using the Wireshark command line tool "tshark". The payload is then decoded with 
#the use of the J2735.py script. Finally the transmit and receive files are merged together based on payload, giving the communication
#between the OBU and RSU.
import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import shutil
import J2735
from binascii import hexlify, unhexlify

text_directory_path = f'{constants.DATA_DIR}/{constants.RAW_TEXT_DIR}'
pcap_directory_path = f'{constants.DATA_DIR}/{constants.RAW_PCAP_DIR}'
tshark_directory_path = f'{constants.DATA_DIR}/{constants.TSHARK_DIR}'
payload_directory_path = f'{constants.DATA_DIR}/{constants.PAYLOAD_DIR}'
merged_directory_path = f'{constants.DATA_DIR}/{constants.MERGED_DIR}'

#writes the timestamp and whole pcap hex to a csv
def convert_pcap_to_csv():
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

#used as part of lambda to decode various desired elements from the UPER encoded hex
def decoder_helper(payload, field):
    msg = J2735.DSRC.MessageFrame
    value = " "

    try:
        msg.from_uper(unhexlify(payload))

        if field == "year":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['year']
            except:
                print("Error decoding year in: " + str(payload))
        if field == "month":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['month']
            except:
                print("Error decoding month in: " + str(payload))
        if field == "day":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['day']
            except:
                print("Error decoding day in: " + str(payload))
        if field == "hour":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['hour']
            except:
                print("Error decoding hour in: " + str(payload))
        if field == "minute":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['minute']
            except:
                print("Error decoding minute in: " + str(payload))
        if field == "second":
            try:
                value = msg()['value'][1]['pathHistory']['initialPosition']['utcTime']['second']
            except:
                print("Error decoding second in: " + str(payload))
        if field == "speed":
            try:
                value = msg()['value'][1]['speed']
            except:
                print("Error decoding speed in: " + str(payload))
    except:
        print("Error converting payload from binary to hex string for payload: " + str(payload))

    return value

#creates csv with timestamp and only the J2735 payload
def get_payload(row):
    map_index = 100000
    spat_index = 100000
    psm_index = 100000
    substrings_to_parse = [constants.MAP_IDENTIFIER, constants.SPAT_IDENTIFIER, constants.PSM_IDENTIFIER]

    #check for map
    if substrings_to_parse[0] in row[1]:
        map_index = row[1].find(substrings_to_parse[0])
    #check for spat
    elif substrings_to_parse[1] in row[1]:
        spat_index = row[1].find(substrings_to_parse[1])
    #check for psm
    elif substrings_to_parse[2] in row[1]:
        psm_index = row[1].find(substrings_to_parse[2])
        psm = row[1][psm_index:]

    if psm_index < map_index and psm_index < spat_index:
        return psm

#gets the payload string from each row and decodes the desired fields
def payloadHelper():
    for filename in os.listdir(tshark_directory_path):
        abs_filename = filename.split('.csv')[0]

        #read the tshark file and create a file containing timestamps and payloads
        df_tshark_output = pd.read_csv(f'{tshark_directory_path}/{filename}')
        df_data = {'timestamp': df_tshark_output.iloc[:,0], 'payload': df_tshark_output.apply(lambda row: get_payload(row), axis=1)}

        payload = pd.DataFrame(data=df_data)
        payload['payload'] = payload['payload'].str.replace(':','')
        payload['psm_year_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "year"), axis=1)
        payload['psm_month_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "month"), axis=1)
        payload['psm_day_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "day"), axis=1)
        payload['psm_hour_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "hour"), axis=1)
        payload['psm_minute_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "minute"), axis=1)
        payload['psm_second_decoded'] = payload.apply(lambda row: decoder_helper(row['payload'], "second"), axis=1)
        payload['psm_speed'] = payload.apply(lambda row: decoder_helper(row['payload'], "speed"), axis=1)

        payload.dropna(subset=['payload'], inplace=True)

        if "Rx" in abs_filename:
            #removing checksum from OBU receive
            payload['payload'] = payload['payload'].str[:-8]
        payload.to_csv(f'{payload_directory_path}/{abs_filename}_payload.csv', index=False)



#perform a left merge on tx and rx file to get communication between RSU and OBU
def merge(date, timestamp_file):
    with open(f'{merged_directory_path}/DSRC_Test_Metrics.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["Test", "Trial", "Tx_Count", "Rx_Count", "Missed_Packet_Count", "PER"])

        tx_files = []
        rx_files = []

        for file in os.listdir(payload_directory_path):
            if "Tx" in file:
                tx_files.append(file)
            if "Rx" in file:
                rx_files.append(file)

        test_date_splitter = date.split("/")
        test_month = test_date_splitter[0].lstrip("0")
        test_day = test_date_splitter[1]
        test_year = test_date_splitter[2]

        for txFile in tx_files:
            tx_date_month = txFile.split("_")[3] 
            tx_date_day = txFile.split("_")[4] 
            tx_date_year = txFile.split("_")[5]

            tx_date = tx_date_month + "/" + tx_date_day + "/" + tx_date_year
            for rxFile in rx_files:
                rx_date_month = rxFile.split("_")[3] 
                rx_date_day = rxFile.split("_")[4] 
                rx_date_year = rxFile.split("_")[5]

                rx_date = rx_date_month + "/" + rx_date_day + "/" + rx_date_year

                #only want to match data from same test dates
                if tx_date == rx_date and tx_date_month == test_month and tx_date_day == test_day and tx_date_year == test_year:
                    txFile = pd.read_csv(f'{payload_directory_path}/{txFile}')
                    #subtract 4 hours for conversion from GMT to local time
                    txFile['timestamp_converted'] = txFile['timestamp'] - 14400
                    rxFile = pd.read_csv(f'{payload_directory_path}/{rxFile}')
                    rxFile['timestamp_converted'] = rxFile['timestamp'] - 14400

                    test_timestamps = pd.read_csv(f'{text_directory_path}/{timestamp_file}')
                    tests_day = test_timestamps[test_timestamps['Date'] == date]
                    tests = tests_day['Test'].unique()

                    #iterate test number for the specific day
                    for testnum in tests:
                        #iterate trial number
                        for j in range(1,9):
                            try:
                                #get test start and end times
                                test_start = tests_day['Start_converted'][(tests_day['Test'] == float(testnum)) & (tests_day['Trial'] == float(j))]
                                test_stop = tests_day['End_converted'][(tests_day['Test'] == float(testnum)) & (tests_day['Trial'] == float(j))]

                                #split payload files based on start/end time of each trial
                                txFileSubset = txFile[(txFile['timestamp_converted'] > float(test_start)) & (txFile['timestamp_converted'] < float(test_stop))]
                                rxFileSubset = rxFile[(rxFile['timestamp_converted'] > float(test_start)) & (rxFile['timestamp_converted'] < float(test_stop))]

                                #left merge on PSM payload
                                mergedFile = txFileSubset.merge(rxFileSubset, how='left', on='payload', suffixes=('_tx', '_rx'))

                                mergedFile.drop(['psm_year_decoded_rx','psm_month_decoded_rx','psm_day_decoded_rx','psm_hour_decoded_rx','psm_minute_decoded_rx','psm_second_decoded_rx'], axis=1, inplace=True)

                                #need to do this bc of error in V2XHub code not properly incrementing seconds in utc timestamp of psm
                                mergedFile['psm_second'] = mergedFile['psm_second_decoded_tx'].astype(int) / 1000
                                mergedFile['psm_second'] = mergedFile['psm_second'].astype(str).str.replace('0.','00.')
                                mergedFile['psm_second_mod'] = mergedFile['psm_second_decoded_tx'].astype(int) % 1000

                                mergedFile['psm_decoded_datetime'] = (mergedFile['psm_year_decoded_tx'].astype(str) + "-0" + mergedFile['psm_month_decoded_tx'].astype(str)
                                + "-" + mergedFile['psm_day_decoded_tx'].astype(str) + " " + mergedFile['psm_hour_decoded_tx'].astype(str) + ":"
                                + mergedFile['psm_minute_decoded_tx'].astype(str) + ":00." + mergedFile['psm_second_mod'].astype(str))

                                txCount = mergedFile['timestamp_converted_tx'].count()
                                rxCount = mergedFile['timestamp_converted_rx'].dropna().count()
                                PER = (1-(rxCount/txCount))*100

                                #calculate various dsrc metrics from tx/rx timestamps
                                mergedFile['TTI'] = mergedFile['timestamp_converted_tx'].diff()
                                mergedFile['IPG'] = mergedFile['timestamp_converted_rx'].diff()
                                mergedFile['Latency'] = mergedFile['timestamp_converted_rx'] - mergedFile['timestamp_converted_tx']

                                mergedFile['Test_Num'] = int(testnum)
                                mergedFile['Trial_Num'] = int(j)

                                mergedFile.drop(['psm_year_decoded_tx','psm_month_decoded_tx','psm_day_decoded_tx','psm_hour_decoded_tx','psm_minute_decoded_tx',
                                'psm_second_decoded_tx', 'psm_second', 'psm_second_mod'], axis=1, inplace=True)

                                csv_writer.writerow([str(testnum), j, txCount, rxCount, (txCount-rxCount),PER])
                                mergedFile.loc[mergedFile.index[len(mergedFile)-1], 'PER'] = PER

                                mergedFile = mergedFile[['Test_Num', 'Trial_Num', 'timestamp_tx', 'payload', 'psm_speed_tx', 'timestamp_converted_tx',
                                'timestamp_rx', 'psm_speed_rx', 'timestamp_converted_rx', 'psm_decoded_datetime','TTI','IPG','Latency','PER']]
                                mergedFile.to_csv(f'{merged_directory_path}/Test_{int(testnum)}_Trial_{j}_merged.csv', index=False)
                            except:
                                print("Error with data for test: " + str(testnum) + " trial: " + str(j))
                                continue

#will concat all of the individual merged files into one file
#***should only be run once all days of testing have been analyzed
def concatFiles():
    merged_directory_path = f'{constants.DATA_DIR}/{constants.MERGED_DIR}'

    all_in_filenames = os.listdir(merged_directory_path)
    merged_files = []

    for file in all_in_filenames:
        if "merged.csv" in file:
            merged_files.append(file)

    matched_files = (pd.read_csv(f'{merged_directory_path}/{f}') for f in merged_files if f in os.listdir(merged_directory_path))
    concatenated_df = pd.concat(matched_files, ignore_index=False)
    out_filename = "All_merged.csv"
    concatenated_df.to_csv(f'{merged_directory_path}/{out_filename}', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run with: "python3 pcap_parser.py" 05/23/2022 timestampFileName')
    else:
        date = sys.argv[1]
        timestamp_file = sys.argv[2]

        convert_pcap_to_csv()
        payloadHelper()
        merge(date, timestamp_file)
        concatFiles()
