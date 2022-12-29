""" This is a parsing script that can be used to extract timestamps from any of the kafka logfiles. """

## How to use this script:
""" Run with python3 timestamp_parser.py logName"""
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
                    textList.append(line.strip().replace("\n", ""))

            #write data of interest to csv which will be used to produce plots
            with open(f'{output_directory_path}/{fileName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Message_Count", "Timestamp(ms)"])

                count = 1
                #Extract the CreateTime timestamp from the kafka data
                for i in range(0, len(textList)):
                    try:
                        json_beg_index = textList[i].find("CreateTime")
                        if (json_beg_index != -1):
                            timestamp = re.sub("[^0-9]", "", textList[i].split(":")[1])                            
                            csv_writer.writerow([count, timestamp])

                            count += 1
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 timestamp_parser.py logname"')
    else:       
        logname = sys.argv[1]
        kafkaParser(logname)