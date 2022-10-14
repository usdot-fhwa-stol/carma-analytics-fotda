import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')

#parser method to extract necessary fields from raw text file
def kafkaParser(logName):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logName in file:
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{logName}.log', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            with open(f'{output_directory_path}/{logName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Timestamp(ms)", "Vehicle_ID", "Cur_ds(m)"])

                for i in range(0, len(textList)):
                    json_beg_index = textList[i].find("{")
                    status_intent_message = textList[i][json_beg_index:]
                    status_intent_message_json = json.loads(status_intent_message)

                    timestamp = status_intent_message_json['metadata']['timestamp']
                    veh_id = status_intent_message_json['payload']['v_id']
                    cur_ds = status_intent_message_json['payload']['cur_ds']

                    csv_writer.writerow([timestamp, veh_id, cur_ds])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python status_intent_parser.py log_name"')
    else:
        status_intent_log = sys.argv[1].split(".")[0]
        
        kafkaParser(status_intent_log)