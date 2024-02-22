import json
import re
from pathlib import Path
from enum import Enum
from KafkaLogMessage import KafkaLogMessage, KafkaLogMessageType


def parse_kafka_logs(inputfile: Path, message_type: KafkaLogMessageType)-> list:
    """Parse Kafka Topic Logs into a list of KafkaLogMessages

    Args:
        inputfile (String): Path to an inputfile
        message_type (KafkaLogMessageType): Type of KafkaLogMessage to parse from input file

    Returns:
        [KafkaLogsMessage]: list of KafkaLogMessages parsed from input file
    """
    if inputfile.is_file():
        #Convert the text file into an array of lines
        with open(inputfile, encoding="utf8", errors='ignore') as textFile:
            textList = []
            for line in textFile:
                textList.append(line.strip())
            msgs = []
            skipped_messages = 0
            for i in range(0, len(textList)):
                try:
                    #get the create time stamped by kafka
                    create_index = textList[i].find("CreateTime")
                    if (create_index != -1):
                        create_time = re.sub("[^0-9]", "", textList[i].split(":")[1])      

                    json_beg_index = textList[i].find("{")
                    kafka_message = textList[i][json_beg_index:]
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

