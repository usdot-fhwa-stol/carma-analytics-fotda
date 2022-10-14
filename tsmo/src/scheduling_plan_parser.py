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
def kafkaParser(logName):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python scheduling_plan_parser.py log_name"')
    else:
        scheduling_log = sys.argv[1]

        kafkaParser(logName)