import sys
from csv import writer
import os
import constants
import json
import pandas as pd
import shutil

#parser method to extract necessary fields from intersection model json
def intersectionModelParser(intersection_model):
    input_directory_path = f'{constants.DATA_DIR}/{constants.INTERSECTION_MODEL_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if intersection_model in file:
            f = open(f'{input_directory_path}/{file}')
            intersection_json = json.load(f)

            filename = file.split(".")[0]
            #write data of interest to csv which will be used to produce plots
            with open(f'{input_directory_path}/{filename}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Entry_lane_id", "Length(m)"])

                #extract relevant elements from the json
                for entry_lane in intersection_json['entry_lanelets']:
                    lane_id = entry_lane['id']
                    lanelet_length = entry_lane['length']
                    
                    csv_writer.writerow([lane_id, lanelet_length])

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Run with: "python intersection_model_parser.py intersectionModelJson"')
    else:     
        intersection_model = sys.argv[1]
  
        intersectionModelParser(intersection_model)