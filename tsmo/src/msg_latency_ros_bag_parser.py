import sys
import csv
import matplotlib.pyplot as plt
import rospy
import rosbag # To import this, run the following command: "pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag rospkg"
import datetime
import constants
import os
from csv import writer
from datetime import datetime, timezone
import pytz





def messageLatencyRosBagParser(logname, vehicle_static_id):
    input_directory_path = f'{constants.DATA_DIR}/{constants.RAW_LOG_DIR}'
    output_directory_path = f'{constants.DATA_DIR}/{constants.PARSED_OUTPUT_DIR}'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logname in file:
            fileName = "".join(file.split("_")[1:])
    bag = rosbag.Bag(f'{input_directory_path}/{logname}')
    #write data of interest to csv which will be used to produce plots
    print("Creating ", fileName + '_schedule_msg_latency.csv' )
    with open(f'{output_directory_path}/{fileName}_schedule_msg_latency.csv', 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["Message Created (ms)", "Message Received by CARMA-Platform ROS (ms)", "Message Latency (ms)"])
        for topic, msg, t in bag.read_messages(topics = ['/message/incoming_mobility_operation']):
            if msg.m_header.recipient_id == vehicle_static_id and msg.strategy_params != "null" :
                csv_writer.writerow([msg.m_header.timestamp, t.to_sec()*1000.0, t.to_sec()*1000.0-msg.m_header.timestamp])
    print("Creating ", fileName + '_spat_msg_latency.csv' )
    with open(f'{output_directory_path}/{fileName}_spat_msg_latency.csv', 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["Message Created (ms)", "Message Received by CARMA-Platform ROS (ms)", "Message Latency (ms)"])
        naive = datetime(int(datetime.today().year), 1, 1, 0, 0, 0)
        utc = pytz.utc
        gmt5 = pytz.timezone('Etc/GMT+5')
        first_day_epoch = utc.localize(naive).astimezone(gmt5).timestamp()*1000
        for topic, msg, t in bag.read_messages(topics = ['/message/incoming_spat']):
            if msg.intersection_state_list[0].time_stamp_exists :
                moy = msg.intersection_state_list[0].moy
                timestamp = msg.intersection_state_list[0].time_stamp * 1000.0
                epoch_ms = (moy* 60000) + timestamp + first_day_epoch #convert moy to milliseconds              
                csv_writer.writerow([epoch_ms, t.to_sec()*1000.0, t.to_sec()*1000.0-epoch_ms])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run with: "python3 msg_latency_ros_bag_parser.py bagfile.bag "DOT-454244"')
    else:       
        logname = sys.argv[1]
        vehicle_static_id = sys.argv[2]
        messageLatencyRosBagParser(logname, vehicle_static_id)