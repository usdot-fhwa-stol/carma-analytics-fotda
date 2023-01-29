import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil

from datetime import datetime, timezone
import pytz
import re
#parser method to extract necessary fields from raw text file
def serviceLogParser(logname):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logname in file:
            fileName = file.split("_")[2]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #write data of interest to csv which will be used to produce plots
            print("Creating ", fileName + '_spat_processing_time.csv' )
            with open(f'{output_directory_path}/{fileName}_spat_processing_time.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Epoch_Time(ms)", "Processing_Time(ms)"])

                #Need to get time since epoch of first day of year to use with moy and timestamp
                #Our local timezone GMT-5 actually needs to be implemented as GMT+5 with pytz library
                #documentation: https://stackoverflow.com/questions/54842491/printing-datetime-as-pytz-timezoneetc-gmt-5-yields-incorrect-result
                
                #extract relevant elements from the json
                for i in range(0, len(textList)):
                    try:
                        #get the create time stamped by kafka
                        processing_time_index = textList[i].find("SPat average processing time")
                        if (processing_time_index != -1):
                            split_logs = re.split(r"\[([^]]+)\]", textList[i])
                            date_time = split_logs[1]
                            timestamp = datetime.strptime(date_time,  "%Y-%m-%d  %H:%M:%S.%f" ).timestamp()
                            log_payload = split_logs[6]
                            res = [float(i) for i in log_payload.split() if isfloat(i)]
                            processing_time = res[1]
                            csv_writer.writerow([timestamp, processing_time])
                            if processing_time > 10 : 
                                print("Warning Spat processing time is ", processing_time, ". If value exceeds 50 ms, this would be off conern. Any value above 10 ms should be noted as it is significantly larger than values normally observed.")

                                
     
                    except:
                        print("Error extracting termine log info for line: " + str(textList[i]))
            print("Creating ", fileName + '_dpp_processing_time.csv' )
            with open(f'{output_directory_path}/{fileName}_dpp_processing_time.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["DPP_Received_Time(ms)", "Command_Queue_Update_Time(ms)", "Processing_Time(ms)"])

                #Need to get time since epoch of first day of year to use with moy and timestamp
                #Our local timezone GMT-5 actually needs to be implemented as GMT+5 with pytz library
                #documentation: https://stackoverflow.com/questions/54842491/printing-datetime-as-pytz-timezoneetc-gmt-5-yields-incorrect-result
                
                #extract relevant elements from the json
                for i in range(0, len(textList)):
                    try: 
                        dpp_consumed_index = textList[i].find("Consumed")
                        if ( dpp_consumed_index != -1) :
                            split_logs = re.split(r"\[([^]]+)\]", textList[i])
                            consumed_date_time = split_logs[1]
                            consumed_timestamp = datetime.strptime(consumed_date_time,  "%Y-%m-%d  %H:%M:%S.%f" ).timestamp()
                            dpp_queue_updated_index = textList[i+1].find("Updated queue")
                            if (dpp_queue_updated_index != -1):
                                split_logs = re.split(r"\[([^]]+)\]", textList[i+1])
                                updated_date_time = split_logs[1]
                                updated_timestamp = datetime.strptime(updated_date_time,  "%Y-%m-%d  %H:%M:%S.%f" ).timestamp()
                                dpp_queue_update_time = updated_timestamp - consumed_timestamp
                                csv_writer.writerow([consumed_timestamp, updated_timestamp, dpp_queue_update_time])     
                    except:
                        print("Error extracting termine log info for line: " + str(textList[i]))

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 tsc_service_log_parser.py logname"')
    else:       
        logname = sys.argv[1]
        serviceLogParser(logname)