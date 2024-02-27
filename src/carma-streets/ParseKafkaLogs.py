import json
import re
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import argparse
from csv import writer


class KafkaLogMessageType(Enum):
    """Enumeration used for indentifying the type of KafkaLogMessage
    """
    TimeSync="time_sync"
    DesiredPhasePlan="desired_phase_plan"
    SPAT="spat"
    TSCConfigState="tsc_config_state"
    BSM="bsm"
    MAP="map"
    MobilityOperation="mobility_operation"
    SchedulingPlan="scheduling_plan"
    SDSM="sdsm"
    DetectedObject="detected_object"
    VehicleStatusIntent="vehicle_status_intent"

@dataclass
class KafkaLogMessage:
    """Class used to store data for each Kafka Log Message
    """
    created: int
    json_message: dict
    msg_type: KafkaLogMessageType
    def __init__(self, created, json_message, msg_type):
        self.created = created
        self.json_message = json_message
        self.msg_type = msg_type

def parse_kafka_logs(input_file_path: Path, message_type: KafkaLogMessageType)-> list:
    """Parse Kafka Topic Logs into a list of KafkaLogMessages

    Args:
        input_file_path (Path): Path to an inputfile
        message_type (KafkaLogMessageType): Type of KafkaLogMessage to parse from input file

    Returns:
        [KafkaLogsMessage]: list of KafkaLogMessages parsed from input file
    """
    if input_file_path.is_file():
        #Convert the text file into an array of lines
        with open(input_file_path, encoding="utf8", errors='ignore') as log_file:
            msgs = []
            skipped_messages = 0
            for line in log_file:
                try:
                    line = line.strip()
                    #get the create time stamped by kafka
                    create_index = line.find("CreateTime")
                    if (create_index != -1):
                        create_time = re.sub("[^0-9]", "", line.split(":")[1])      

                    json_beg_index = line.find("{")
                    kafka_message = line[json_beg_index:]
                    kafka_json = json.loads(kafka_message)
                    msgs.append(KafkaLogMessage(create_time, kafka_json, message_type))
                except json.JSONDecodeError as e:
                    print(f"Error {e} extracting json info for message: {kafka_message}. Skipping message.")
                    skipped_messages += 1
            if skipped_messages > 0 :
                print(f"WARNING: Skipped {skipped_messages} due to JSON decoding errors. Please inspect logs.")
            else:
                print(f"Successfully extracted all {len(msgs)} messages from inputfile.")
            return msgs

def parse_spat_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse SPAT Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    spat_msgs = parse_kafka_logs(inputfile, KafkaLogMessageType.SPAT)
    if not outputfile.exists():
        #write data of interest to csv which will be used to produce plots
        with open(outputfile, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Created Time(ms)", "Intersection Name", "Intersection ID", "Minute of the Year",
                 "Millisecond of the Minute", "Intersection State"])            
            skipped_messages = 0
            #extract relevant elements from the json
            for msg in spat_msgs:
                try:
                    csv_writer.writerow([
                        msg.created, 
                        msg.json_message['intersections'][0]['name'],
                        msg.json_message['intersections'][0]['id'],
                        msg.json_message['intersections'][0]['moy'],
                        msg.json_message['intersections'][0]['time_stamp'],
                        msg.json_message['intersections'][0]['states']])
                except Exception as e:
                    print(f"Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.")
            if skipped_messages == 0 :
                print("Finished writing all entries successfully")
            else:
                print(f"WARNING: Skipped {skipped_messages} due to errors. Please inspect logs")
    else :
        print(f"Output file {outputfile} already exists! Aborting process")

def parse_timesync_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse timesync Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    timesync_msgs = parse_kafka_logs(inputfile, KafkaLogMessageType.TimeSync)
    if not outputfile.exists():
        #write data of interest to csv which will be used to produce plots
        with open(outputfile, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Created Time(ms)", "Epoch Time(ms)", "Sequence Number"])
            skipped_messages = 0
            #extract relevant elements from the json
            for msg in timesync_msgs:
                try:
                    csv_writer.writerow([msg.created, msg.json_message['timestep'], msg.json_message['seq']])
                except Exception as e:
                    print(f"Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.")
            if skipped_messages == 0 :
                print("Finished writing all entries successfully")
            else:
                print(f"WARNING: Skipped {skipped_messages} due to errors. Please inspect logs")
    else :
        print(f"Output file {outputfile} already exists! Aborting process")

def parse_message_types(kafka_log_dir, csv_dir):
    """Parse all Kafka Topic Logs in a provided directory and output csv message data.

    Args:
        kafka_log_dir (Path): String path to directory kafka logs directory.
        csv_dir (Path): _description_
    """
    kafka_log_dir_path = Path(kafka_log_dir)
    csv_dir_path = Path(csv_dir)
    if kafka_log_dir_path.is_dir():
        if csv_dir_path.is_dir():
            print(f"WARNING: {csv_dir} already exists. Contents will be overwritten.")
        csv_dir_path.mkdir(exist_ok=True)
        for kafka_topic_log in kafka_log_dir_path.glob("*.log"):
            if KafkaLogMessageType.TimeSync.value in kafka_topic_log.name :
                print(f"Found TimeSync Kafka topic log {kafka_topic_log}. Parsing log to csv ...")
                parse_timesync_to_csv(kafka_topic_log, Path(f"{csv_dir}/{KafkaLogMessageType.TimeSync.value}.csv"))
            elif KafkaLogMessageType.SPAT.value in kafka_topic_log.name:
                print(f"Found SPAT Kafka topic log {kafka_topic_log}. Parsing log to csv ...")
                parse_spat_to_csv(kafka_topic_log, Path(f"{csv_dir}/{KafkaLogMessageType.SPAT.value}.csv"))

    else:
        print("Please ensure that Kafka Logs Directory exists and CSV Logs directory does not exist")

   
def main():
    parser = argparse.ArgumentParser(description='Script to parse Kafka Topic log files into CSV data')
    parser.add_argument('--kafka-log-dir', help='Directory containing Kafka Log files.', type=str)  # Required argument
    parser.add_argument('--csv-dir', help='Directory to write csv files to.', type=str)  # Required argument


    args = parser.parse_args()
    parse_message_types(args.kafka_log_dir, args.csv_dir)


if __name__ == '__main__':
    main()