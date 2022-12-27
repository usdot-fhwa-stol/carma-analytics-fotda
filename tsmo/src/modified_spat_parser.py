import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil
from datetime import datetime, timezone
import pytz

#parser method to extract necessary fields from raw text file
def kafkaParser(logname):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logname in file:
            fileName = file.split(".")[0]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #write data of interest to csv which will be used to produce plots
            with open(f'{output_directory_path}/{fileName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Intersection_Name", "Intersection_ID", "Moy", "Timestamp", "Signal_Group", 
                "Event_State", "Event_State_Color", "Epoch_Time(ms)", "Epoch_Time(s)"])

                #Need to get time since epoch of first day of year to use with moy and timestamp
                #Our local timezone GMT-5 actually needs to be implemented as GMT+5 with pytz library
                #documentation: https://stackoverflow.com/questions/54842491/printing-datetime-as-pytz-timezoneetc-gmt-5-yields-incorrect-result
                naive = datetime(datetime.now(timezone.utc).year, 1, 1, 0, 0, 0)
                utc = pytz.utc
                gmt5 = pytz.timezone('Etc/GMT+5')
                first_day_epoch = utc.localize(naive).astimezone(gmt5).timestamp()*1000

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

                        #Use moy and timestamp fields to get epoch time for each record
                        epoch_ms = (moy* 60000) + timestamp + first_day_epoch #convert moy to milliseconds              
                        epoch_sec = epoch_ms / 1000


                        #iterate through all states
                        for j in range(0, len(states)):
                            signal_group = states[j]['signal_group']

                            #retrieve the first element in state_time_speed object
                            state_time_speed = states[j]['state_time_speed'][0]
                            event_state = state_time_speed['event_state']

                            #match event state to signal head color
                            event_state_color = ""
                            if event_state == 3:
                                event_state_color = "red"
                            elif event_state == 6:
                                event_state_color = "green"
                            elif event_state == 8:
                                event_state_color = "yellow"
                            csv_writer.writerow([intersectionName, intersectionID, moy, timestamp, signal_group, event_state, event_state_color, 
                            epoch_ms, epoch_sec])
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 modified_spat_parser.py logname"')
    else:       
        logname = sys.argv[1]
        kafkaParser(logname)