from csv import writer
from pathlib import Path
from enum import Enum
import argparse
import ParseKafkaLog
import ParseTimeSyncToCSV
import ParseSPATToCSV
from KafkaLogMessage import KafkaLogMessage, KafkaLogMessageType


def parse_message_types(kafka_log_dir, csv_dir):
    """Parse all Kafka Topic Logs in a provided directory and output csv message data.

    Args:
        kafka_log_dir (Path): String path to directory kafka logs directory.
        csv_dir (Path): _description_
    """
    kafka_log_dir_path = Path(kafka_log_dir)
    csv_dir_path = Path(csv_dir)
    if kafka_log_dir_path.is_dir() and not csv_dir_path.is_dir():
        csv_dir_path.mkdir(exist_ok=False)
        for kafka_topic_log in kafka_log_dir_path.glob("*.log"):
            if KafkaLogMessageType.TimeSync.value in kafka_topic_log.name :
                print(f"Found TimeSync Kafka topic log {kafka_topic_log}. Parsing log to csv ...")
                ParseTimeSyncToCSV.parse_timesync_to_csv(kafka_topic_log, Path(f"{csv_dir}/{KafkaLogMessageType.TimeSync.value}.csv"))
            elif KafkaLogMessageType.SPAT.value in kafka_topic_log.name:
                print(f"Found SPAT Kafka topic log {kafka_topic_log}. Parsing log to csv ...")
                ParseSPATToCSV.parse_spat_to_csv(kafka_topic_log, Path(f"{csv_dir}/{KafkaLogMessageType.SPAT.value}.csv"))

    else:
        print("Please ensure that Kafka Logs Directory exists and CSV Logs directory does not exist")

   
def main():
    parser = argparse.ArgumentParser(description='Parse Time Sync data from Kafka logs and generate csv output')
    parser.add_argument('--kafka-log-dir', help='Directory containing Kafka Log files.', type=str)  # Required argument
    parser.add_argument('--csv-dir', help='Output Directory to write csv files to.', type=str)  # Required argument
    ## TODO
    #parser.add_argument('--messageTypes', help='Output Directory to write csv files to.', type=str)  # Required argument

    args = parser.parse_args()
    parse_message_types(args.kafka_log_dir, args.csv_dir)


if __name__ == '__main__':
    main()
