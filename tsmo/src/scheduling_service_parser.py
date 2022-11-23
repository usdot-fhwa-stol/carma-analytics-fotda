import sys
from csv import writer
from csv import reader
import os
import constants
import json
import pandas as pd
import sys
import shutil
import scheduling_service_metrics

#clean out directories prior to running
def cleaningDirectories():
    if os.path.isdir(f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}'):
        shutil.rmtree(f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}')
        os.makedirs(f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}')
    else:
        os.makedirs(f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}')

#parser method to extract necessary fields from raw text file
def outputParser():
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_INPUT_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.SS_PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)
    firstProducerRow = True
    firstConsumerRow = True

    #find the logfile of interest in the raw text files directory
    for file in all_in_filenames:
        fileName = file.split(".")[0]
        if fileName == logFile:
            filename = file.split(".")[0]

            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line)

            #boolean to check if a consumer message has been received
            consumedMessage = False
            #creating an array to count number of null producer payloads in a row
            nullProducer = []
            csvLineCount = 0

            #write data of interest to csv files, one for producer data, one for consumer data, and one with both
            with open(f'{output_directory_path}/{filename}_SS_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(["Kafka", "Machine_Date", "Machine_Time", "Message_Timestamp", "v_id", "v_length", "cur_speed", "cur_accel",
                "react_t", "max_accel", "max_decel", "min_gap", "depart_pos", "is_allowed", "cur_lane_id", "cur_ds", "entry_lane_id", "dest_lane_id",
                "link_lane_id", "direction", "est_paths_id", "est_paths_ds", "est_paths_ts"])

                with open(f'{output_directory_path}/{filename}_SS_producer_parsed.csv', 'w', newline='') as write_obj_producer:
                    csv_writer_producer = writer(write_obj_producer)
                    csv_writer_producer.writerow(["Kafka", "Machine_Date", "Machine_Time", "Message_Timestamp","v_id",
                    "st", "et", "dt", "dp", "access", "EST", "State"])

                    with open(f'{output_directory_path}/{filename}_SS_consumer_parsed.csv', 'w', newline='') as write_obj_consumer:
                        csv_writer_consumer = writer(write_obj_consumer)

                        #create array to store vehicle ids and ESTs
                        vehicle_est_arr = []
                        for i in range(0, len(textList)):

                            if "Scheduling Class - vehicle" in textList[i]:
                                vehicle = textList[i].split(" ")[7].replace(":", "")
                                est = textList[i].split(" ")[13].replace(",", "")
                                state = textList[i+2].split(" ")[23].replace(",", "")
                                vehicle_est_arr.append([vehicle, est, state])

                            #Need to add this check for change in log format when vehicle changes state to "DV"
                            elif "Scheduling Class Vehicle Info Update" in textList[i]:
                                vehicle = textList[i].split(" ")[17].replace(",", "")
                                est = " "
                                state = textList[i].split(" ")[23].replace(",", "")
                                vehicle_est_arr.append([vehicle, est, state])

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
                                            #iterate through number of payload objects
                                            payload_arr = []
                                            numVehicles = len(payload_json['payload'])

                                            #iterate through number of vehicles in producer message json
                                            for i in range(1, numVehicles+1):
                                                #iterate through vehicle est arr to find matching vehicle id
                                                for j in range(1, len(vehicle_est_arr)+1):
                                                    id = payload_json['payload'][i-1]['v_id']
                                                    #check if vehicle ids match
                                                    if id == vehicle_est_arr[j-1][0]:
                                                        est = vehicle_est_arr[j-1][1]
                                                        state = vehicle_est_arr[j-1][2]

                                                        #generate a csv row for each vehicle in the json
                                                        prod_row = [kafka, machine_date, machine_time, payload_timestamp,
                                                        payload_json['payload'][i-1]['v_id'],
                                                        payload_json['payload'][i-1]['st'],
                                                        payload_json['payload'][i-1]['et'],
                                                        payload_json['payload'][i-1]['dt'],
                                                        payload_json['payload'][i-1]['dp'],
                                                        payload_json['payload'][i-1]['access'],
                                                        est, state]

                                                        csv_writer.writerow(prod_row)
                                                        csv_writer_producer.writerow(prod_row)
                                        else:
                                            #if there is a null payload, write to array
                                            nullProducer.append("null")

                                    #if a consumer message has been receieved and there are more than 20 consecutive null producer payload, a new file should be made
                                    if consumedMessage == True and len(nullProducer) > 20:
                                        consumedMessage = False
                                        nullProducer.clear()
                                        splitLine.append(csvLineCount)

                                    #clear the array because producer message has been parsed
                                    vehicle_est_arr.clear()

                                except:
                                    print("Error at line: " + textList[i] + "in file: " + filename + " at line number: " + str(i))

                            #search for weird string --> consumer messages
                            elif '):  {"metadata"' in textList[i]:
                                consumer_state = " "
                                if "Vehicle Class Vehicle Info Update" in textList[i+1]:
                                    consumer_state = textList[i+1].split(" ")[20].replace(",", "")

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
                                    json_start_index = textList[i].find('{"metadata"')
                                    payload_string = textList[i][json_start_index:]
                                    payload_json = json.loads(payload_string)
                                    payload_timestamp = payload_json['metadata']['timestamp']

                                    v_id = payload_json['payload']['v_id']
                                    v_length = payload_json['payload']['v_length']
                                    cur_speed = payload_json['payload']['cur_speed']
                                    cur_accel = payload_json['payload']['cur_accel']
                                    react_t = payload_json['payload']['react_t']
                                    max_accel = payload_json['payload']['max_accel']
                                    max_decel = payload_json['payload']['max_decel']
                                    min_gap = payload_json['payload']['min_gap']
                                    depart_pos = payload_json['payload']['depart_pos']
                                    is_allowed = payload_json['payload']['is_allowed']
                                    cur_lane_id = payload_json['payload']['cur_lane_id']
                                    cur_ds = payload_json['payload']['cur_ds']
                                    entry_lane_id = payload_json['payload']['entry_lane_id']
                                    dest_lane_id = payload_json['payload']['dest_lane_id']
                                    link_lane_id = payload_json['payload']['link_lane_id']
                                    direction = payload_json['payload']['direction']

                                    #need to iterate through number of est paths
                                    path_arr = []
                                    numPaths = len(payload_json['payload']['est_paths'])

                                    for i in range(1, numPaths+1):
                                        path_arr.append([payload_json['payload']['est_paths'][i-1]['id'], payload_json['payload']['est_paths'][i-1]['ds'],
                                        payload_json['payload']['est_paths'][i-1]['ts']])

                                    consumer_row = [kafka, machine_date, machine_time, payload_timestamp, v_id, v_length, consumer_state, cur_speed, cur_accel, react_t, max_accel, max_decel,
                                    min_gap, depart_pos, is_allowed, cur_lane_id, cur_ds, entry_lane_id, dest_lane_id, link_lane_id, direction]

                                    for i in range(1, numPaths+1):
                                        for item in path_arr[i-1]:
                                            consumer_row.append(item)

                                    #write the header of the consumer row based on the number of est paths
                                    if firstConsumerRow == True:
                                        consumer_header = ["Kafka", "Machine_Date", "Machine_Time", "Message_Timestamp", "v_id", "v_length", "state", "cur_speed", "cur_accel",
                                        "react_t", "max_accel", "max_decel", "min_gap", "depart_pos", "is_allowed", "cur_lane_id", "cur_ds", "entry_lane_id", "dest_lane_id",
                                        "link_lane_id", "direction"]

                                        for i in range(1, numPaths+1):
                                            consumer_header.append("est_paths_id_"+str(i))
                                            consumer_header.append("est_paths_ds_"+str(i))
                                            consumer_header.append("est_paths_ts_"+str(i))

                                        firstConsumerRow = False
                                        csv_writer_consumer.writerow(consumer_header)

                                    csv_writer.writerow(consumer_row)
                                    csv_writer_consumer.writerow(consumer_row)

                                    #set the boolean appropriately and clear the null producer payload array
                                    consumedMessage = True
                                    nullProducer.clear()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 scheduling_service_parser.py logfileName"')
        exit()
    else:       
        #name of log to process ex: scheduling_service_01_25_10am
        logFile = sys.argv[1]

        #vehicle ids to use when plotting
        vehicle_id_1 = constants.VEHICLE_ID_1
        vehicle_id_2 = constants.VEHICLE_ID_2

        cleaningDirectories()
        outputParser()
