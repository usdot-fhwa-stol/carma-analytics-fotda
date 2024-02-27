# CARMA Streets Data Analysis
## Introduction
This Package contains several Python Modules useful for parsing Kafka Topic logs into csv format for plotting and data analysis.
## Collecting Kafka Logs
Documentation for data collection for **CARMA Streets** can be found [here](https://github.com/usdot-fhwa-stol/carma-streets/tree/release/lavida?tab=readme-ov-file#data-collection). To collect Kafka Topic log data, the Kafka docker image must still be running. Using the `collect_kafka_logs.py` script you can collect all the messages on each topic. Each topic will generate a log file with the json messages exchandeg on the topic and a timestamp for when Kafka "Created" the message. This "Created" timestamp can be treated as the time at which **CARMA Streets** received the message.
## Parsing Kafka logs to CSV 
Using the scripts in this package we should be able to ingest all Kafka Topic Logs and produce CSV data from the resulting messages for data analysis. The `ParseKafkaLogsDirectory.py` is intended for use with to collected Kafka logs using the `collect_kafka_losg.py` script. It will search a provided directory for a log file for each supported Kafka Message type and output a CSV file containing the message data.
```
usage: ParseKafkaLogs.py [-h] [--kafka-log-dir KAFKA_LOG_DIR] [--csv-dir CSV_DIR]

Script to parse Kafka Topic log files into CSV data

options:
  -h, --help            show this help message and exit
  --kafka-log-dir KAFKA_LOG_DIR
                        Directory containing Kafka Log files.
  --csv-dir CSV_DIR     Directory to write csv files to.
```