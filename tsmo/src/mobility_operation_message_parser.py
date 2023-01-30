import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil
import re

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
                csv_writer.writerow(["Create_Time(ms)", "Timestamp(ms)"])

                #extract relevant elements from the json
                for i in range(0, len(textList)):
                    try:
                        #get the create time stamped by kafka
                        create_index = textList[i].find("CreateTime")
                        if (create_index != -1):
                            create_time = re.sub("[^0-9]", "", textList[i].split(":")[1])      

                            for j in range(i+1,i+12):
                                params = textList[j].find("strategy_params")

                                #if we find the strategy params, form the json
                                if params != -1:
                                    json_text = textList[i+1:j+2]
                                    split_text = ' '.join(json_text)
                                    split_text = "{" + split_text

                                    full_json = json.loads(split_text)
                                    timestamp = full_json['metadata']['timestamp']
                            
                                    csv_writer.writerow([create_time, timestamp])
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 desired_phase_plan_parser.py logname"')
    else:       
        logname = sys.argv[1]
        kafkaParser(logname)