from csv import writer
from pathlib import Path
from enum import Enum
import argparse
import ParseKafkaLog
from KafkaLogMessage import KafkaLogMessage, KafkaLogMessageType


def parse_timesync_to_csv(inputfile: Path, outputfile: Path):
    """Function to parse timesync Kafka Topic log file and generate csv data of all time sync messages

    Args:
        inputfile (Path): Path to Kafka Topic log file
        outputfile (Path): File name (excluding file extension) of desired csv file
    """
    timesync_msgs = ParseKafkaLog.parse_kafka_logs(inputfile, KafkaLogMessageType.TimeSync)
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
