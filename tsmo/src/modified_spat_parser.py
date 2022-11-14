import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil

#parser method to extract necessary fields from raw text file
def kafkaParser():
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if "modified_spat" in file:
            fileName = file.split(".")[0]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #write data of interest to csv which will be used to produce plots
            with open(f'{output_directory_path}/{fileName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Intersection_Name", "Intersection_ID", "Moy", "Timestamp", "Signal_Group", "Event_State"])

                #extract relevant elements from the json
                for i in range(0, len(textList)):
                    try:
                        json_beg_index = textList[i].find("{")
                        modified_spat_message = textList[i][json_beg_index:]
                        modified_spat_message_json = json.loads(modified_spat_message)

                        #retrieve message metadata
                        intersectionName = modified_spat_message_json['intersections'][0]['name']
                        intersectionID = modified_spat_message_json['intersections'][0]['id']
                        moy = modified_spat_message_json['intersections'][0]['moy']
                        timestamp = modified_spat_message_json['intersections'][0]['time_stamp']

                        states = modified_spat_message_json['intersections'][0]['states']

                        #iterate through all states
                        for i in range(0, len(states)):
                            signal_group = states[i]['signal_group']

                            #retrieve the first element in state_time_speed object
                            state_time_speed = states[i]['state_time_speed'][0]
                            event_state = state_time_speed['event_state']

                            #match event state to signal head color
                            event_state_color = ""
                            if event_state == 3:
                                event_state_color = "red"
                            elif event_state == 6:
                                event_state_color = "green"
                            elif event_state == 8:
                                event_state_color = "yellow"
                            csv_writer.writerow([intersectionName, intersectionID, moy, timestamp, signal_group, event_state_color])
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python3 modified_spat_parser.py"')
    else:       
        kafkaParser()