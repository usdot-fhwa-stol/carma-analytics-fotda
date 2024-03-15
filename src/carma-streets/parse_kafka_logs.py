import json
import re
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import argparse
from csv import writer
import datetime
from dateutil import tz



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
def get_create_time(line: str) -> int:
    """Read CreateTime from line in Kafka Log file.

    Args:
        line (str): line in Kafka log file

    Returns:
        int: timestamp
    """
    return re.sub('[^0-9]', '', line.split(':')[1])

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
            kafka_message = str()
            for line in log_file:
                line = line.strip()
                if 'CreateTime' in line:
                    if kafka_message:
                        try:
                            kafka_json = json.loads(kafka_message)
                            kafka_message = ''
                            msgs.append(KafkaLogMessage(create_time, kafka_json, message_type))
                        except json.JSONDecodeError as e:
                            print(f'Error {e.msg} extracting json info for message: {kafka_message}. Skipping message.')
                            skipped_messages += 1
                            kafka_message = ''
                    create_time = get_create_time(line)
                    json_beg_index = line.find('{')
                    kafka_message += line[json_beg_index:]
                else:
                    kafka_message += line

            if skipped_messages > 0 :
                print(f'WARNING: Skipped {skipped_messages} due to JSON decoding errors. Please inspect logs.')
            else:
                print(f'Successfully extracted all {len(msgs)} messages from inputfile.')
            return msgs
def get_spat_timestamp(json_data: dict,test_year: int, simulation: bool = False):
    """Function to extract timestamp data from SPAT message

    Args:
        json_data (dict): SPAT JSON data
        test_year (int): Test year data is within
        simulation (bool): Flag to indicate whether data was collected in simulation environment. Default is False

    Returns:
        int: epoch timestamp in milliseconds
    """
    first_day_epoch = datetime.datetime(test_year, 1, 1, 0, 0, 0, tzinfo=tz.gettz('America/New_York')).timestamp()*1000
    if simulation:
        first_day_epoch = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=tz.UTC).timestamp()*1000


    return json_data['intersections'][0]['moy']*60*1000 + json_data['intersections'][0]['time_stamp'] + first_day_epoch

def parse_spat_to_csv(inputfile: Path, outputfile: Path, simulation: bool=False):
    """Function to parse SPAT Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
        simulation (bool): Flag to indicate whether data was collected in simulation environment (Default = False)
    """
    spat_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.SPAT)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['System Time (ms)', 'Intersection Name', 'Intersection ID', 'Message Time (ms)', 'Intersection State'])
        skipped_messages = 0
        #Need to get time since epoch of first day of year to use with moy and timestamp
        #Our local timezone GMT-5 actually needs to be implemented as GMT+5 with pytz library
        #documentation: https://stackoverflow.com/questions/54842491/printing-datetime-as-pytz-timezoneetc-gmt-5-yields-incorrect-result

        #Get the epoch ms time of the first data of the relavent year
        test_year = datetime.datetime.fromtimestamp(int(spat_msgs[0].created_time)/1000).year
        #extract relevant elements from the json
        for msg in spat_msgs:
            try:
                csv_writer.writerow([
                    msg.created_time,
                    msg.json_message['intersections'][0]['name'],
                    msg.json_message['intersections'][0]['id'],
                    get_spat_timestamp(msg.json_message, test_year,simulation),
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
    with open(outputfile, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['System Time (ms)', 'Message Time (ms)', 'Sequence Number'])
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

def parse_map_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse MAP Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """

    map_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.MAP)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['System Time (ms)', 'Message Time (ms)', 'Map Data'])
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in map_msgs:
            try:
                csv_writer.writerow([msg.created_time, msg.json_message['metadata']['timestamp'], msg.json_message['map_data']])
            except Exception as e:
                print(f'Error {e} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')

def get_sdsm_timestamp(json_data: dict, simulation: bool = False) -> int :
    """Get SDSM timestamp in milliseconds from json SDSM date time

    Args:
        json_data (dict): The sdsm_time_stamp part of the SDSM JSON
        simulation (bool): Flag to indicate whether data was collected in simulation environment. Default is False

    Returns:
        int: epoch millisecond timestamp
    """
    if simulation:
        return datetime.datetime( json_data['year'], \
                    json_data['month'], \
                    json_data['day'], \
                    json_data['hour'], \
                    json_data['minute'], \
                    json_data['second']//1000, \
                    (json_data['second']%1000) * 1000, \
                    tzinfo=tz.UTC).timestamp()*1000
    else:
        return datetime.datetime( json_data['year'], \
                    json_data['month'], \
                    json_data['day'], \
                    json_data['hour'], \
                    json_data['minute'], \
                    json_data['second']//1000, \
                    (json_data['second']%1000) * 1000, \
                    tzinfo=tz.gettz('America/New_York')).timestamp()*1000

def parse_sdsm_to_csv(inputfile: Path, outputfile: Path, simulation: bool = False):
    """Function to parse SDSM Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
        simulation (bool): Flag to indicate whether data was collected in simulation environment. Default is False
    """

    sdsm_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.SDSM)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['System Time (ms)', 'Message Time (ms)',
             'Message Count', 'Source ID', 'Equipement Type', 'Reference Position Longitude',
             'Reference Position Latitude', 'Reference Position Elevation','Objects'])
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in sdsm_msgs:
            try:
                csv_writer.writerow([
                    msg.created_time,
                    get_sdsm_timestamp(msg.json_message['sdsm_time_stamp'],simulation),
                    msg.json_message['msg_cnt'],
                    msg.json_message['source_id'],
                    msg.json_message['equipment_type'],
                    msg.json_message['ref_pos']['long'],
                    msg.json_message['ref_pos']['lat'],
                    msg.json_message['ref_pos']['elevation'],
                    msg.json_message['objects']])
            except Exception as e:
                print(f'Error {e.msgs} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')

def parse_detected_object_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse Detected Object Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """

    detected_object_msgs = parse_kafka_logs_as_type(inputfile, KafkaLogMessageType.DetectedObject)
    if  outputfile.exists():
        print(f'Output file {outputfile} already exists. Overwritting file.')
    #write data of interest to csv which will be used to produce plots
    with open(outputfile, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['System Time (ms)', 'Message Time (ms)',
             'Type', 'Sensor ID', 'Projection String', 'Object ID',
             'Position X(m)', 'Position Y(m)','Position Z(m)'])
        skipped_messages = 0
        #extract relevant elements from the json
        for msg in detected_object_msgs:
            try:
                csv_writer.writerow([
                    msg.created_time,
                    msg.json_message['timestamp'],
                    msg.json_message['type'],
                    msg.json_message['sensorId'],
                    msg.json_message['projString'],
                    msg.json_message['objectId'],
                    msg.json_message['position']['x'],
                    msg.json_message['position']['y'],
                    msg.json_message['position']['z']])
            except Exception as e:
                print(f'Error {e.msgs} occurred while writing csv entry for kafka message {msg.json_message}. Skipping message.')
        if skipped_messages == 0 :
            print('Finished writing all entries successfully')
        else:
            print(f'WARNING: Skipped {skipped_messages} due to errors. Please inspect logs')
def parse_kafka_log_dir(kafka_log_dir:str, csv_dir:str, simulation:bool=False):
    """Parse all Kafka Topic Logs in a provided directory and output csv message data.

    Args:
        kafka_log_dir (str): String path to directory kafka logs directory.
        csv_dir (str): String path to directory to write CSV files to.
        simulation (bool): Flag to indicate whether data was collected in simulation environment. Default is False

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
                parse_spat_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.SPAT.value}.csv', simulation)
            elif KafkaLogMessageType.MAP.value in kafka_topic_log.name:
                print(f'Found MAP Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_map_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.MAP.value}.csv')
            elif KafkaLogMessageType.SDSM.value in kafka_topic_log.name:
                print(f'Found SDSM Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_sdsm_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.SDSM.value}.csv', simulation)
            elif KafkaLogMessageType.DetectedObject.value in kafka_topic_log.name:
                print(f'Found Detected Object Kafka topic log {kafka_topic_log}. Parsing log to csv ...')
                parse_detected_object_to_csv(kafka_topic_log, csv_dir_path/f'{KafkaLogMessageType.DetectedObject.value}.csv')
    else:
        print('ERROR:Please ensure that Kafka Logs Directory exists.')


def main():
    parser = argparse.ArgumentParser(description='Script to parse Kafka Topic log files into CSV data')
    parser.add_argument('--kafka-log-dir', help='Directory containing Kafka Log files.', type=str, required=True)
    parser.add_argument('--csv-dir', help='Directory to write csv files to.', type=str, required=True)
    parser.add_argument('--simulation', help='Flag indicating data is from simulation', action='store_true')
    args = parser.parse_args()
    parse_kafka_log_dir(args.kafka_log_dir, args.csv_dir, args.simulation)


if __name__ == '__main__':
    main()
