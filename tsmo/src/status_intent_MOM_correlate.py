""" This is an analysis script that can be used to correlate the mobility operations messages with
the vehicle status and intent messages. This script requires the status and intent data and mobility operations
message data to be parsed using their respective parser scripts. To establish the time for a unique MOM to be 
associated with a specific vehicle status and intent message, the difference in "CreateTime" between the status 
and intent message and the MOM is calculated. """

## How to use this script:
""" Run with python3 status_intent_MOM_correlate.py parsedMOMLog parsedStatusIntentLog"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import constants
import sys
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


#This function will correlate MOM with status and intent messages.
def plotter(mom_parsed, status_intent_parsed):
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'

    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        if mom_parsed in file:
            mom_data = pd.read_csv(f'{input_directory_path}/{file}')
        elif status_intent_parsed in file:
            status_intent_data = pd.read_csv(f'{input_directory_path}/{file}')
    
    #perform left merge on MOM data
    combined_df = status_intent_data.merge(mom_data, how='left', on='Timestamp(ms)', suffixes=('_SI', '_MOM'))
    combined_df['Processing_Time(ms)'] = combined_df['Create_Time(ms)_SI'] - combined_df['Create_Time(ms)_MOM']
    combined_df.drop(['Vehicle_ID', 'Cur_ds(m)', 'Cur_Speed', 'Cur_Accel', 
    'Cur_lane_id', 'Entry_lane_id', 'Link_lane_id', 'Dest_lane_id', 'Vehicle_state'], axis=1, inplace=True)

    combined_fileName = status_intent_parsed.split("_")[4] + "_" + status_intent_parsed.split("_")[5] + "_" + status_intent_parsed.split("_")[6]
    combined_df.to_csv(f'{output_directory_path}/{combined_fileName}_MOM_SI_correlate.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run with: "python3 status_intent_MOM_correlate.py parsedMOMLog parsedStatusIntentLog"')
    else:       
        mom = sys.argv[1]
        status_intent = sys.argv[2]
        
        plotter(mom, status_intent)
