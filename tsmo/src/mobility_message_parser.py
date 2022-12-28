""" This is a parsing script that can be used to extract timestamps from the mobility operations/path message kafka logs. """

## How to use this script:
""" Run with python3 mobility_operation_message_parser.py logName"""
import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil

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
                #The following logic is used to iteratively extract relevant elements from the json. B/c the mobility message jsons 
                #are logged with a newline character after every line of text, additional processing needed to be performed to compile
                #a usable json string.
                for i in range(0, len(textList)):
                    try:
                        #Find the "metadata" string portion of the json message
                        json_beg_index = textList[i].find("metadata")
                        if (json_beg_index != -1):
                            #Find "}," indicating the end of the metadata section
                            for j in range(0,10):
                                json_end_index = textList[i+j].find("},")

                                if (json_end_index != -1): 
                                    #Using the end index to compile a list of all lines containing the metadata string     
                                    mobility_operation_message = textList[i-1:i+j]
                                    
                                    #Convert the list to string
                                    mobility_operation_message = ' '.join(mobility_operation_message).replace(" ", "")
                                    mobility_operation_message = mobility_operation_message.replace("\t", "")
                                    mobility_operation_message += "}}"

                                    #Convert to json string and extract timestamp
                                    json_beg_index = mobility_operation_message.find("{")
                                    mobility_operation_message_json = mobility_operation_message[json_beg_index:]
                                    mobility_operation_message_json = json.loads(mobility_operation_message_json)

                                    timestamp = mobility_operation_message_json['metadata']['timestamp'].lstrip("0")
                                    csv_writer.writerow([count, timestamp])

                                    count += 1
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 mobility_message_parser.py logname"')
    else:       
        logname = sys.argv[1]
        kafkaParser(logname)