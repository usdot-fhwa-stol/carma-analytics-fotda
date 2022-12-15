#This script is used to convert the test date and times to time since epochs values, which will be used
#for further analysis. The one parameter required is the name of the test log.
import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import shutil

text_directory_path = f'{constants.DATA_DIR}/{constants.RAW_TEXT_DIR}'

#Converts test date and time to datetime object, then converts to epoch time
def timestamp_converter(logName):
    for file in os.listdir(text_directory_path):
        #Find desired log file
        if logName in file:
            fileName = file.split(".")[0]
            timestamp_file = pd.read_csv(f'{text_directory_path}/{file}')

            #create datetime object by concat date and start/end string together
            timestamp_file.dropna(subset=['Start'], how='any',inplace=True)
            timestamp_file["Date_Time_Start"] = timestamp_file['Date'] + " " + timestamp_file['Start']
            timestamp_file["Date_Time_End"] = timestamp_file['Date'] + " " + timestamp_file['End']

            #convert datetime to time since epoch
            timestamp_file['Start_converted'] = pd.to_datetime(timestamp_file['Date_Time_Start']).map(pd.Timestamp.timestamp).astype(float)
            timestamp_file['End_converted'] = pd.to_datetime(timestamp_file['Date_Time_End']).map(pd.Timestamp.timestamp).astype(float)

            timestamp_file.to_csv(f'{text_directory_path}/{fileName}_converted.csv', index=False)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 test_log_parser.py testLogName"')
    else:
        logName = sys.argv[1]
        timestamp_converter(logName)
