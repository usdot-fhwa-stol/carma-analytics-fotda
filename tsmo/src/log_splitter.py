import constants
import pandas as pd
from datetime import datetime
import sys

#input name of scheduling and messaging service log files (without file extension)
scheduling = sys.argv[1]
messaging = sys.argv[2]
testDate = scheduling.split("_")[0] + "-" + scheduling.split("_")[1] + "-" + scheduling.split("_")[2]

smallfile = None
endTimeArr = [0]

input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_INPUT_DIR}'
output_directory_path = f'{constants.DATA_DIR}/{constants.SPLIT_INPUT_DIR}'

#splits schedule log file when there are 20 consecutive null producer payloads and a consumer message has been created
def scheduling_splitter(schedulingFile):
    null_producer_count = 0
    consumer = False
    fileCount = 1

    with open(f'{input_directory_path}/{schedulingFile}.txt') as schedulingFile:
        smallfile = open(f'{output_directory_path}/scheduling_service_run_1.txt', "w")

        for lineno, line in enumerate(schedulingFile):
            #finds a null producer payload
            if "message content:  {" in line and '"payload":null' in line:
                null_producer_count += 1
            #finds valid consumer message
            if '):  {"metadata"' in line:
                null_producer_count = 0
                consumer = True

            #if split conditions are met the file is closed and a new file is created
            if null_producer_count == 20 and consumer == True:
                fileCount += 1
                consumer = False
                if smallfile:
                    smallfile.close()

                end_time = line.split(" ")[1].replace("]", "")
                endTimeArr.append(end_time)
                small_filename = "scheduling_service_run_"+str(fileCount)+"_"+str(end_time.split(".")[0])+".txt"
                smallfile = open(f'{output_directory_path}/{small_filename}', "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    #returns an array that contains all of the times for the end of a test run
    return endTimeArr


#function to convert date time to time since epoch in seconds
def epochConverter(dateTime):
    t1 = datetime.strptime(dateTime, "%Y-%m-%d %H:%M:%S.%f")
    epoch = datetime.utcfromtimestamp(0) # start of epoch time
    delta = t1 - epoch
    #epoch time for stop time to search for in file
    return delta.total_seconds()

#uses the scheduling splitter array to split up messaging service log
def messaging_splitter(stop_times_arr, schedulingFile, messagingFile):
    fileCount = 1
    epoch_stop_times_arr = []

    #convert all test run end times to time since epoch
    for time in stop_times_arr:
        if time == 0:
            combined_time = testDate + " 00:00:00.000"
        else:
            combined_time = testDate + " " + time
        epoch_stop_times_arr.append(epochConverter(combined_time))


    #create split messaging service files
    with open(f'{input_directory_path}/{schedulingFile}.txt') as messagingFile:
        for i in range(0, len(epoch_stop_times_arr)):
            smallfile = open(f'{output_directory_path}/messaging_service_run_{i+1}_{stop_times_arr[i+1].split(".")[0]}.txt', "w")

            time = epoch_stop_times_arr[i]
            messagingFile.seek(0)

            for lineno, line in enumerate(messagingFile):
                #get machine date/time for each line of output
                if "[info]" in line:
                    machine_date = line.split(" ")[0].replace("[", "")
                    machine_time = line.split(" ")[1].replace("]", "")
                    machine_date_time = machine_date + " " + machine_time

                    time_since_epoch2 = epochConverter(machine_date_time)
                #write to file if the machine time is within the desired range
                if time_since_epoch2 > epoch_stop_times_arr[i] and time_since_epoch2 < epoch_stop_times_arr[i+1]:
                    smallfile.write(line)

            fileCount += 1

            if smallfile:
                smallfile.close()


times = scheduling_splitter(scheduling)
messaging_splitter(times, scheduling, messaging)
