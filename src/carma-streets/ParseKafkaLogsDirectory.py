from csv import writer
from pathlib import Path
from enum import Enum
import argparse
import ParseKafkaLog
import ParseTimeSyncToCSV
from KafkaLogMessage import KafkaLogMessage, KafkaLogMessageType


def parse_message_types(kafkaLogDir, csvDir):
    """Parse all Kafka Topic Logs in a provided directory and output csv message data.

    Args:
        kafkaLogDir (Path): String path to directory kafka logs directory.
        csvDir (Path): _description_
    """
    kafkaLogDir_path = Path(kafkaLogDir)
    csvDir_path = Path(csvDir)
    if kafkaLogDir_path.is_dir() and not csvDir_path.is_dir():
        csvDir_path.mkdir(exist_ok=False)
        for kafka_topic_log in kafkaLogDir_path.glob("*.log"):
            if not kafka_topic_log.name.find(KafkaLogMessageType.TimeSync.value) == -1 :
                print(f"Found TimeSync Kafka topic log {kafka_topic_log}. Parsing log to csv ...")
                ParseTimeSyncToCSV.parse_timesync_log(kafka_topic_log, Path(f"{csvDir}/{KafkaLogMessageType.TimeSync.value}.csv"))
    else:
        print("Please ensure that Kafka Logs Directory exists and CSV Logs directory does not exist")

   
def main():
    parser = argparse.ArgumentParser(description='Parse Time Sync data from Kafka logs and generate csv output')
    parser.add_argument('--kafkaLogDir', help='Directory containing Kafka Log files.', type=str)  # Required argument
    parser.add_argument('--csvDir', help='Output Directory to write csv files to.', type=str)  # Required argument
    ## TODO
    #parser.add_argument('--messageTypes', help='Output Directory to write csv files to.', type=str)  # Required argument

    args = parser.parse_args()
    parse_message_types(args.kafkaLogDir, args.csvDir)


if __name__ == '__main__':
    main()
