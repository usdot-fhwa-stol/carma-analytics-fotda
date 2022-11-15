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
# from datetime import datetime
import datetime as dt
import matplotlib.dates as md

pd.options.mode.chained_assignment = None

def modified_spat_plotter():
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PLOT_DIR}'
   
    all_in_filenames = os.listdir(input_directory_path)
    for file in all_in_filenames:
        if "modified_spat" in file and "parsed" in file:
            modified_spat_data = pd.read_csv(f'{input_directory_path}/{file}')
            signal_groups = modified_spat_data['Signal_Group'].unique()

            first_day_epoch = dt.datetime(2022, 1, 1, 0, 0, 0).timestamp() * 1000 #get time since epoch for beggining of year in ms
            modified_spat_data['Epoch_Time(ms)'] = (modified_spat_data['Moy'] * 60000) + modified_spat_data['Timestamp'] + first_day_epoch #convert moy to milliseconds              
            modified_spat_data['Epoch_Time(s)'] = modified_spat_data['Epoch_Time(ms)'] / 1000
            min_time = dt.datetime.fromtimestamp(modified_spat_data['Epoch_Time(s)'].min())
            max_time = dt.datetime.fromtimestamp(modified_spat_data['Epoch_Time(s)'].max())

            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 10)

            #Iterate through every signal group in the parsed file
            for group in signal_groups:
                df = modified_spat_data.copy()
                group_subset = df[df['Signal_Group'] == group]
                dates=[dt.datetime.fromtimestamp(ts) for ts in group_subset["Epoch_Time(s)"]]
                ax1.plot(dates,group_subset['Event_State'], label="Signal Group " + str(group))

            plt.xticks(rotation=75)
            axs=plt.gca()
            xfmt = md.DateFormatter('%H:%M:%S.%f') 
            axs.xaxis.set_major_formatter(xfmt)
            fig.autofmt_xdate()
            plt.xlim(min_time, max_time)
            plt.ylim(0, 12)
            #TODO update plot title/name once log naming convention has been established
            plt.xlabel('Date-Time')
            plt.ylabel('Event State')
            fig.suptitle("Signal Group Event State vs Time")
            plt.legend()
            plotName = "Signal_Groups_Event_State_Vs_Time.png"
            plt.savefig(f'{output_directory_path}/{plotName}')

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python modified_spat_plotter.py"')
    else:          
        modified_spat_plotter()    
