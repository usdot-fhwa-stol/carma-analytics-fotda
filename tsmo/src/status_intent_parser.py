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
def kafkaParser():
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if "status_intent" in file:
            fileName = file.split(".")[0]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #write data of interest to csv which will be used to produce plots
            with open(f'{output_directory_path}/{fileName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Timestamp(ms)", "Vehicle_ID", "Cur_ds(m)", "Cur_lane_id", "Entry_lane_id", "Link_lane_id", "Dest_lane_id", "EV", "DV", "LV"])

                #extract relevant elements from the json
                for i in range(0, len(textList)):
                    try:
                        json_beg_index = textList[i].find("{")
                        status_intent_message = textList[i][json_beg_index:]
                        status_intent_message_json = json.loads(status_intent_message)

                        timestamp = status_intent_message_json['metadata']['timestamp']
                        veh_id = status_intent_message_json['payload']['v_id']
                        cur_ds = status_intent_message_json['payload']['cur_ds']
                        cur_lane_id = status_intent_message_json['payload']['cur_lane_id']
                        entry_lane_id = status_intent_message_json['payload']['entry_lane_id']
                        link_lane_id = status_intent_message_json['payload']['link_lane_id']
                        dest_lane_id = status_intent_message_json['payload']['dest_lane_id']

                        #check state of vehicle using cur_lane_id
                        EV = 0
                        DV = 0
                        LV = 0
                        if cur_lane_id == entry_lane_id:
                            EV = 1
                        if cur_lane_id == link_lane_id:
                            DV = 1 
                        if cur_lane_id == dest_lane_id:
                            LV = 1
                        csv_writer.writerow([timestamp, veh_id, cur_ds, cur_lane_id, entry_lane_id, link_lane_id, dest_lane_id, EV, DV, LV])
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python status_intent_parser.py"')
    else:       
        kafkaParser()