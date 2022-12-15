This directory contains the necessary scripts for data analysis on the TSMO Cooperative Perception Use Case. The raw
data from the various test events must be placed in the "data" directory prior to runnning the scripts. The order to
run the scripts is below:

1. First, run the "test_log_parser.py" script with the csv containing the start and end date/times for all test cases.
    Ex: python3 test_log_parser.py CP_Test_timestamps.csv

2. Next, run the "rviz_psm_parser.py".
    Ex: python3 rviz_psm_parser.py

3. Next, run the "docker_log_parser.py" script with the name of the Docker log, the date of testing, and the name of the timestamp
file generated in step 1. This script extracts necessary data from the docker logs and writes them to a csv. Run this script for every day of testing data collected. 
    Ex: python3 docker_log_parser.py V2X_Hub_5_23_2022.log 05/23/22 CP_Test_timestamps_converted.csv

4. Next, run the "pcap_parser.py" script with the date of testing and the name of the timestamp file generated in step 1. This script extracts timestamps and J2735 payloads from the pcap files. It then merges the RSU and OBU files based on payload to get the communication between the two devices. Run this script for every day of testing data collected.  
    Ex: python3 pcap_parser.py 05/23/22 CP_Test_timestamps_converted.csv

5. Next, run the "final_combiner.py" script with the date of testing and the name of the timestamp file generated in step 1. 
    Ex: python3 final_combiner.py 05/23/22 CP_Test_timestamps_converted.csv

6. Finally, run the "plotter.py" with the test number interested in plotting
    Ex: python3 plotter.py 1
