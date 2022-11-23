import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import scheduling_service_metrics
import shutil

def fileSplitter():
    output_directory_path = f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}'
    print(splitLine)
    parsedFile = pd.read_csv(f'{output_directory_path}/{logFile}_parsed.csv', skiprows=1)
    
    csvfile = open('import_1458922827.csv', 'r').readlines()
    filename = 1
    for i in range(len(csvfile)):
        if i % 1000 == 0:
            open(str(filename) + '.csv', 'w+').writelines(csvfile[i:i+1000])
            filename += 1


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python3 scheduling_service_parser.py logfileName"')
    else:       
        splitLine = [0]

        fileSplitter()
