from csv import writer
from pathlib import Path
from enum import Enum
import argparse
import ParseKafkaLog
from KafkaLogMessage import KafkaLogMessage, KafkaLogMessageType


def parseSpatToCsv(inputfile: Path, outputfile: Path):
    """Function to parse SPAT Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    spat_msgs = ParseKafkaLog.parse_kafka_logs(inputfile, KafkaLogMessageType.TimeSync)
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
