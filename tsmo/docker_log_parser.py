import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import shutil

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}')

#parser method to extract necessary fields from raw text file
def dockerLogParser():
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_INPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        fileName = file.split(".")[0]
        if fileName == logFile:
            filename = file.split(".")[0]

            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            #removing empty strings
            textList = [i for i in textList if i]

            with open(f'{output_directory_path}/{filename}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["FLIR_Ped_Time", "V2XHub_Ped_Rx_Time", "V2XHub_PSM_Tx_Time", "ped_id", "PSM_Count", "PSM_Hex"])

                for i in range(0, len(textList)):
                    #search for this string indicating a PSM has been sent to the RSU for broadcast
                    if "Pedestrian Plugin :: Broadcast PSM::" in textList[i]:
                        v2xhub_time = ""
                        flir_ped_create_time = ""
                        flir_ped_rx_dateTime = ""
                        id = ""
                        count = ""
                        hex = ""
                        v2xhub_ped_rx_datetime = ""

                        try:
                            #get time when FLIR created ped data

                            #example: {"log":"[2022-05-23 19:29:35.715] WebSockAsyncClnSession.cpp (190) - INFO   : Received PedestrianPresenceTracking
                            #data at time: 2022-05-23T15:29:35.711-04:00\n","stream":"stdout","time":"2022-05-23T19:29:35.715987218Z"}
                            flir_ped_create_time = textList[i-10].split("at time:")[1].split("T")[0].lstrip() + " " + textList[i-10].split("at time:")[1].split("T")[1].split("-")[0]
                        except:
                            print("Error retrieving flir create time at line: " + str(i))
                            continue

                        try:
                            #get time when V2X Hub received ped info from FLIR
                            #convert from GMT to local time (subtract 4 hours)

                            #example: {"log":"[2022-05-23 19:29:35.715] WebSockAsyncClnSession.cpp (190) - INFO   : Received PedestrianPresenceTracking
                            #data at time: 2022-05-23T15:29:35.711-04:00\n","stream":"stdout","time":"2022-05-23T19:29:35.715987218Z"}
                            v2xhub_datetime = textList[i-10].split(" ")[0] + " " + textList[i-10].split(" ")[1]
                            v2xhub_datetime = v2xhub_datetime.split('{"log":"[')[1].replace("]", "")

                            date = v2xhub_datetime.split(" ")[0]
                            time = v2xhub_datetime.split(" ")[1]
                            hour_converted = int(time.split(":")[0].lstrip()) - 4
                            if hour_converted < 0:
                                hour_converted += 24

                            new_time = str(hour_converted) + ":" + time.split(":")[1] + ":" + time.split(":")[2]

                            v2xhub_ped_rx_datetime = date + " " + new_time
                        except:
                            print("Error retrieving v2xhub time at line: " + str(i))
                            continue
                        try:
                            #get ped ID

                            #example: {"log":"[2022-05-23 19:29:36.716] WebSockAsyncClnSession.cpp (258) - INFO   : Received FLIR camera
                            #data for pedestrian 84423000 at location: (389550271, -771491380), travelling at speed: 98.315,
                            #with heading: 11920 degrees\n","stream":"stdout","time":"2022-05-23T19:29:36.716752509Z"}

                            id = textList[i-6].split("pedestrian")[1].split(" ")[1]
                        except:
                            print("Error retrieving id at line: " + textList[i])
                            continue

                        try:
                            #get PSM count

                            #example: {"log":"[2022-05-23 19:29:33.614] WebSockAsyncClnSession.cpp (261) - INFO   : PSM message count: 13\n",
                            #"stream":"stdout","time":"2022-05-23T19:29:33.614962907Z"}

                            count = textList[i-4].split("PSM message count:")[1].split('","')[0].replace("n", "").strip()
                            count = count[:-1]
                        except:
                            print("Error retrieving count at line: " + str(i))
                            continue

                        try:
                            #get UPER encoded PSM hex

                            #example: {"log":"[2022-05-23 19:29:35.417] n/src/PedestrianPlugin.cpp (274) - INFO   :  Pedestrian Plugin ::
                            #Broadcast PSM:: 00203320000203387e1108c0004cdcf9da3d4dcae8ffffffff02b2ad0480fcfccb77ba0338d693a3fed693a400010000400010000000\n",
                            #"stream":"stdout","time":"2022-05-23T19:29:35.41794713Z"}

                            hex = textList[i].split(" ")[15].split('","')[0]
                            hex = hex[:-2]
                        except:
                            print("Error retrieving hex at line: " + str(i))
                            continue

                        try:
                            #getting time when V2X Hub sent PSM to RSU

                            #example: {"log":"[2022-05-23 19:29:35.417] n/src/PedestrianPlugin.cpp (274) - INFO   :  Pedestrian Plugin ::
                            #Broadcast PSM:: 00203320000203387e1108c0004cdcf9da3d4dcae8ffffffff02b2ad0480fcfccb77ba0338d693a3fed693a400010000400010000000\n",
                            #"stream":"stdout","time":"2022-05-23T19:29:35.41794713Z"}

                            v2xhub_tx_datetime = textList[i].split(" ")[0] + " " + textList[i].split(" ")[1]
                            v2xhub_tx_datetime = v2xhub_tx_datetime.split('{"log":"[')[1].replace("]", "")

                            date = v2xhub_tx_datetime.split(" ")[0]
                            time = v2xhub_tx_datetime.split(" ")[1]
                            hour_converted = int(time.split(":")[0].lstrip()) - 4
                            if hour_converted < 0:
                                hour_converted += 24

                            new_time = str(hour_converted) + ":" + time.split(":")[1] + ":" + time.split(":")[2]

                            final_v2xhub_tx_datetime = date + " " + new_time
                        except:
                            print("Error retrieving v2xhub tx time at line: " + str(i))
                            continue

                        csv_writer.writerow([flir_ped_create_time, v2xhub_ped_rx_datetime, final_v2xhub_tx_datetime, id, count, hex])

def dockerLogSplitter():
    text_directory_path = f'{constants.DATA_DIR}/{constants.RAW_TEXT_DIR}'
    input_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    dateToSearch = logFile.split("_")[2] + "_" + logFile.split("_")[3] + "_" + logFile.split("_")[4]
    for file in all_in_filenames:
        if "parsed" in file and dateToSearch in file:
            parsedFile = file

    print(parsedFile)
    #read parsed docker log file created in above function
    parsedFileData = pd.read_csv(f'{input_directory_path}/{parsedFile}')

    #convert flir rx time and v2x psm tx datetimes to time since epoch
    parsedFileData['V2XHub_Ped_Rx_Time_converted'] = pd.to_datetime(parsedFileData['V2XHub_Ped_Rx_Time'], errors='coerce').map(pd.Timestamp.timestamp, na_action='ignore').astype(float, errors='ignore')
    parsedFileData['V2XHub_PSM_Tx_Time_converted'] = pd.to_datetime(parsedFileData['V2XHub_PSM_Tx_Time'],errors='coerce').map(pd.Timestamp.timestamp, na_action='ignore').astype(float, errors='ignore')

    #calculate v2x hub processing time by calculating difference
    parsedFileData['V2XHub_Processing_Time(s)'] = parsedFileData['V2XHub_PSM_Tx_Time_converted'] - parsedFileData['V2XHub_Ped_Rx_Time_converted']

    test_timestamps = pd.read_csv(f'{text_directory_path}/CP_Test_timestamps_converted.csv')
    tests_day = test_timestamps[test_timestamps['Date'] == date]
    tests = tests_day['Test'].unique().astype(int)

    #iterate through each trial of the different tests and create parsed docker log files for each
    for testnum in tests:
        for i in range(1,9):
            try:
                test_start = tests_day['Start_converted'][(tests_day['Test'] == float(testnum)) & (tests_day['Trial'] == float(i))]
                test_stop = tests_day['End_converted'][(tests_day['Test'] == float(testnum)) & (tests_day['Trial'] == float(i))]
                #split parsed docker log file based on start/end time of each trial
                dockerParsedTestTrialSubset = parsedFileData[(parsedFileData['V2XHub_Ped_Rx_Time_converted'] > float(test_start)) & (parsedFileData['V2XHub_Ped_Rx_Time_converted'] < float(test_stop))]
                dockerParsedTestTrialSubset['Test_Num'] = int(testnum)
                dockerParsedTestTrialSubset['Trial_Num'] = int(i)

                dockerParsedTestTrialSubset = dockerParsedTestTrialSubset[['Test_Num','Trial_Num', 'FLIR_Ped_Time', 'V2XHub_Ped_Rx_Time', 'V2XHub_PSM_Tx_Time','PSM_Count',
                'PSM_Hex','V2XHub_Ped_Rx_Time_converted','V2XHub_PSM_Tx_Time_converted','V2XHub_Processing_Time(s)']]

                dockerParsedTestTrialSubset.to_csv(f'{output_directory_path}/Test_{int(testnum)}_Trial_{i}_docker_log_parsed.csv', index=False)

            except:
                print("No docker log data for test: " + str(testnum) + " trial: " + str(i))
                continue

#will concat all of the individual parsed files into one file
#***should only be run once all days of testing have been analyzed
def concatFiles():
    parsed_output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'

    all_in_filenames = os.listdir(parsed_output_directory_path)
    parsed_files = []

    for file in all_in_filenames:
        if "docker_log_parsed.csv" in file:
            parsed_files.append(file)

    matched_files = (pd.read_csv(f'{parsed_output_directory_path}/{f}') for f in parsed_files if f in os.listdir(parsed_output_directory_path))
    concatenated_df = pd.concat(matched_files, ignore_index=False)
    out_filename = "All_docker_parsed.csv"
    concatenated_df.to_csv(f'{parsed_output_directory_path}/{out_filename}', index=False)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run with docker log name and date: "python docker_log_parser.py" V2X_Hub_5_23_2022 05/23/2022')
    else:
        logFile = sys.argv[1]
        date = sys.argv[2]
        # cleaningDirectories()
        # dockerLogParser()
        # dockerLogSplitter()
        concatFiles()
