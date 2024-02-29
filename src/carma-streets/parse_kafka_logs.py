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
    TimeSync='time_sync'
    DesiredPhasePlan='desired_phase_plan'
    SPAT='spat'
    TSCConfigState='tsc_config_state'
    BSM='bsm'
    MAP='map'
    MobilityOperation='mobility_operation'
    SchedulingPlan='scheduling_plan'
    SDSM='sdsm'
    DetectedObject='detected_object'
    VehicleStatusIntent='vehicle_status_intent'

@dataclass
class KafkaLogMessage:
    """Class used to store data for each Kafka Log Message
    """
    created_time: int
    json_message: dict
    msg_type: KafkaLogMessageType
    
def parse_kafka_logs_as_type(input_file_path: Path, message_type: KafkaLogMessageType)-> list:
    """Parse Kafka Topic Logs into a list of KafkaLogMessages of the type provided.

    Args:
        input_file_path (Path): Path to an inputfile
        message_type (KafkaLogMessageType): Type of KafkaLogMessage to parse message to

    Returns:
        [KafkaLogsMessage]: list of KafkaLogMessages parsed from input file
    """
    if input_file_path.is_file():
        #Convert the text file into an array of lines
        with open(input_file_path, encoding='utf8', errors='ignore') as log_file:
            msgs = []
            skipped_messages = 0
            for line in log_file:
                try:
                    line = line.strip()
                    #get the create time stamped by kafka
                    create_index = line.find('CreateTime')
                    if (create_index != -1):
                        create_time = re.sub('[^0-9]', '', line.split(':')[1])      

                    json_beg_index = line.find('{')
                    kafka_message = line[json_beg_index:]
                    kafka_json = json.loads(kafka_message)
                    msgs.append(KafkaLogMessage(create_time, kafka_json, message_type))
                except json.JSONDecodeError as e:
                    print(f'Error {e} extracting json info for message: {kafka_message}. Skipping message.')
                    skipped_messages += 1
            if skipped_messages > 0 :
                print(f'WARNING: Skipped {skipped_messages} due to JSON decoding errors. Please inspect logs.')
            else:
                print(f'Successfully extracted all {len(msgs)} messages from inputfile.')
            return msgs

def parse_spat_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse SPAT Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    spat_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.SPAT)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['Created Time(ms)', 'Intersection Name', 'Intersection ID', 'Minute of the Year',
                'Millisecond of the Minute', 'Intersection State'])            
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in spat_msgs:
            try:
                csv_writer.writerow([
                    msg.created_time, 
                    msg.json_message['intersections'][0]['name'],
                    msg.json_message['intersections'][0]['id'],
                    msg.json_message['intersections'][0]['moy'],
                    msg.json_message['intersections'][0]['time_stamp'],
                    msg.json_message['intersections'][0]['states']])
            except Exception as e:
                print(f'Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')

def parse_timesync_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse timesync Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    timesync_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.TimeSync)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['Created Time(ms)', 'Epoch Time(ms)', 'Sequence Number'])
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in timesync_msgs:
            try:
                csv_writer.writerow([msg.created_time, msg.json_message['timestep'], msg.json_message['seq']])
            except Exception as e:
                print(f'Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')

def parse_detectedobject_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse timesync Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    intersection_lidar_x_offset = 263.799988
    intersection_lidar_y_offset = 178.050003
    intersection_lidar_z_offset = 0.0

    detectedobject_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.DetectedObject)
    print(f"Extracted {len(detectedobject_msgs)} messages from inputfile. xxx")
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    with open(outputfile, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['Timestamp (ms)', 'Type', 'ObjID', 'Positionx', 'Positiony'])
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in detectedobject_msgs:
            try:
                # csv_writer.writerow([msg.created_time, msg.json_message['timestep'], msg.json_message['seq']])
                csv_writer.writerow([msg.json_message['timestamp'], msg.json_message['type'], msg.json_message['objectId'],  
                                    msg.json_message['position']['x']+intersection_lidar_x_offset, msg.json_message['position']['y']+intersection_lidar_y_offset])
            except Exception as e:
                print(f'Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')


def parse_kafka_log_dir(kafka_log_dir, csv_dir):
    """Parse all Kafka Topic Logs in a provided directory and output csv message data.

    Args:
        kafka_log_dir (Path): String path to directory kafka logs directory.
        csv_dir (Path): _description_
    """
    kafka_log_dir_path = Path(kafka_log_dir)
    csv_dir_path = Path(csv_dir)
    if kafka_log_dir_path.is_dir():
        if csv_dir_path.is_dir():
            print(f'WARNING: {csv_dir} already exists. Contents will be overwritten.')
        csv_dir_path.mkdir(exist_ok=True)
        for kafka_topic_log in kafka_log_dir_path.glob('*.log'):
            if KafkaLogMessageType.TimeSync.value in kafka_topic_log.name :
                print(f'Found TimeSync Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_timesync_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.TimeSync.value}.csv')
            elif KafkaLogMessageType.SPAT.value in kafka_topic_log.name:
                print(f'Found SPAT Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_spat_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.SPAT.value}.csv')
            elif KafkaLogMessageType.DetectedObject.value in kafka_topic_log.name:
                print(f'Found DetectedObject Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_detectedobject_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.DetectedObject.value}.csv')

    else:
        print('ERROR:Please ensure that Kafka Logs Directory exists.')

def main():
    parser = argparse.ArgumentParser(description='Script to parse Kafka Topic log files into CSV data')
    parser.add_argument('--kafka-log-dir', help='Directory containing Kafka Log files.', type=str, required=True) 
    parser.add_argument('--csv-dir', help='Directory to write csv files to.', type=str, required=True)  
    args = parser.parse_args()
    parse_kafka_log_dir(args.kafka_log_dir, args.csv_dir)


if __name__ == '__main__':
    main()