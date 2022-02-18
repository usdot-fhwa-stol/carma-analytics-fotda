import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import messaging_service_metrics
import shutil

#name of log to process ex: scheduling_service_01_25_10am
logFile = sys.argv[1]
#vehicle ids to use when plotting
vehicle_id_1 = constants.VEHICLE_ID_1
vehicle_id_2 = constants.VEHICLE_ID_2

splitLine = [0]

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}')

    # if os.path.isdir(f'{constants.DATA_DIR}/{constants.PLOT_DIR}'):
    #     shutil.rmtree(f'{constants.DATA_DIR}/{constants.PLOT_DIR}')
    #     os.makedirs(f'{constants.DATA_DIR}/{constants.PLOT_DIR}')
    # else:
    #     os.makedirs(f'{constants.DATA_DIR}/{constants.PLOT_DIR}')

#parser method to extract necessary fields from raw text file
def outputParser():
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_INPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.MS_PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)
    firstProducerRow = True
    firstConsumerRow = True

    for file in all_in_filenames:
        fileName = file.split(".")[0]
        if fileName == logFile:
            filename = file.split(".")[0]

            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line)

            with open(f'{output_directory_path}/{filename}_MS_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Kafka", "Machine_Date", "Machine_Time", "Message_Timestamp"])

                with open(f'{output_directory_path}/{filename}_MS_producer_parsed.csv', 'w', newline='') as write_obj_producer:
                    csv_writer_producer = writer(write_obj_producer)
                    csv_writer_producer.writerow(["Kafka", "Machine_Date", "Machine_Time", "Message_Timestamp"])

                    with open(f'{output_directory_path}/{filename}_MS_consumer_parsed.csv', 'w', newline='') as write_obj_consumer:
                        csv_writer_consumer = writer(write_obj_consumer)
                        csv_writer_consumer.writerow(["Kafka", "Message_Type", "Vehicle_ID", "Machine_Date", "Machine_Time", "Message_Timestamp",
                        "BSM_MessageCount", "BSM_Accel_Long(m/s^2)", ])

                        for i in range(0, len(textList)):
                            #search for message content --> producer messages
                            if "message content:" in textList[i]:
                                splitter = textList[i].split(" ")

                                #Consumed or produced message
                                try:
                                    kafka = splitter[4].split("#")[1].split("-")[0]

                                    #Read in the json and parse necessary fields
                                    json_start_index = textList[i].find('{"metadata"')
                                    payload_string = textList[i][json_start_index:]
                                    payload_json = json.loads(payload_string)
                                    payload_timestamp = payload_json['metadata']['timestamp']

                                    #check if it is a producer message
                                    if kafka == "producer":
                                        #Get the date and time reported by the computer
                                        machine_date_time = splitter[0] + " " + splitter[1]
                                        machine_date_time_replaced = machine_date_time.replace("[", "").replace("]", "")
                                        machine_date =  machine_date_time_replaced.split(" ")[0]
                                        machine_time =  machine_date_time_replaced.split(" ")[1]

                                        #Only write to csv if the payload is not null
                                        if payload_json['payload'] != None:
                                            #generate a csv row based on the number of vehicles in the json
                                            prod_row = [kafka, machine_date, machine_time, payload_timestamp]
                                            csv_writer.writerow(prod_row)
                                            csv_writer_producer.writerow(prod_row)
                                except:
                                    print("Error at line: " + textList[i] + "in file: " + filename + " at line number: " + str(i))

                            #search for weird string --> consumer messages
                            elif '):  {' in textList[i]:
                                splitter = textList[i].split(" ")

                                #Consumed or produced message
                                kafka = splitter[4].split("#")[1].split("-")[0]
                                if kafka == "consumer":
                                    #Get the date and time reported by the computer
                                    machine_date_time = splitter[0] + " " + splitter[1]
                                    machine_date_time_replaced = machine_date_time.replace("[", "").replace("]", "")
                                    machine_date =  machine_date_time_replaced.split(" ")[0]
                                    machine_time =  machine_date_time_replaced.split(" ")[1]

                                    #Read in the json and parse necessary fields
                                    json_start_index = textList[i].find('):  {')
                                    nextLine = textList[i+1].strip().replace('"', "").split(" ")[0]
                                    bsm_json_str = "{"

                                    msg_count = " "
                                    accel_long = " "
                                    if nextLine == "core_data":
                                        for j in range(1, 41):
                                            bsm_json_str += textList[i+j]

                                        bsm_json = json.loads(bsm_json_str)
                                        message_type = "BSM"
                                        timestamp = " "
                                        accel_long = float(bsm_json['core_data']['accel_set']['long']) * 0.01 #convert to m/s^2
                                        veh_id = bsm_json['core_data']['id']
                                        msg_count = bsm_json['core_data']['msg_count']

                                    elif nextLine == "metadata":
                                        veh_id = textList[i+4].strip().replace('"', "").split(" ")[2].replace(',', "")
                                        timestamp = textList[i+7].strip().replace('"', "").split(" ")[2]
                                        if textList[i+9].strip().replace('"', "").split(" ")[0] == "strategy":
                                            message_type = "MOM"
                                        elif textList[i+9].strip().replace('"', "").split(" ")[0] == "trajectory":
                                            message_type = "MPM"

                                    consumer_row = [kafka, message_type, veh_id, machine_date, machine_time, timestamp,
                                    msg_count, accel_long]
                                    csv_writer.writerow(consumer_row)
                                    csv_writer_consumer.writerow(consumer_row)


cleaningDirectories()
outputParser()
messaging_service_metrics.runner(logFile, vehicle_id_1, vehicle_id_2)
