""" This is a parsing script that can be used to extract start and end timestamps for a service's processing times. """

## How to use this script:
""" Run with python3 so_service_log_parser.py logName"""
import sys
from csv import writer
import os
import constants
import shutil
import re

#parser method to extract necessary fields from raw text file
def logParser(logName):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logName in file:
            fileName = file.split("_")[2].split(".")[0]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #write data of interest to csv which will be used to produce plots
            with open(f'{output_directory_path}/{fileName}_signal_optimization_processing_time.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)

                startSearchString = "Signal Optimization iteration start time"
                endSearchString = "Signal Optimization iteration end time"
                noVehicleString = "No vehicles present"

                csv_writer.writerow(["Signal_Optimization_Start_Time(ms)", "Signal_Optimization_End_Time(ms)", "Processing_Time(ms)"])

                #extract relevant elements from the log
                for i in range(0, len(textList)):
                    try:
                        #first start by checking for signal optimization start times
                        if startSearchString in textList[i]:

                            #Next check if the "no vehicles present" string exists in the next two lines (this indicates we can
                            #skip this block)
                            if noVehicleString not in textList[i+1] and noVehicleString not in textList[i+2]:
                                
                                #Now search for the signal optimization end time in the next 200 lines of output
                                for j in range(i+1, i+200):
                                    #first need to check if there is no end time in this block
                                    #do this by checking if we see the "start time" string again
                                    if j < len(textList) and startSearchString in textList[j]:
                                        break

                                    #next check if the end time string is in the current line
                                    elif j < len(textList) and endSearchString in textList[j]:
                                        startTime = textList[i].split(" ")[9].split("!")[0]
                                        endTime = textList[j].split(" ")[9].split("!")[0]
                                        processingTime = int(endTime) - int(startTime)

                                        csv_writer.writerow([startTime, endTime, processingTime])
                                        #break this for loop bc we have found the matching end time
                                        break
                    except:
                        print("Error extracting json info for line: " + str(textList[i]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 so_service_log_parser.py logname"')
    else:       
        logname = sys.argv[1]

        logParser(logname)