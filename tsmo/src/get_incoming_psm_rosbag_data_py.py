from pydoc_data.topics import topics
import sys
import csv
import matplotlib.pyplot as plt
import rospy
import rosbag
import datetime
import math
import os
import re

# Usage:
# python3.7 get_incoming_psm_rosbag_data in folder containing rosbags and csv of Test data
# 1. The rosbags need to be labelled with yyyy-mm--dd--Test--Testnums (2022-05-24-Test-5-2-8.bag)
# The rosbag file names need to be add to the script main as well
# The Test number represents the run/case. Since this script was initially used for multiple runs for specific cases, 
# the cases recorded in the rosbag help filter the data being looked for
# 2. The csv log file records the Test start and stop timings.
# Since the initial test analysis comprised of multiple Tests in one rosbag, this helps specify the duration to check within the rosbag
# The csv log file needs to be labelled Test_Timestamps.csv
# The format of entries in the csv needs to be TestNum, TrialNum, Start time, End time with the timings at utc time
# (example : Test1, Trial 1,10:46:26.40,10:46:51.30)


def format_timestamp_nanos(nanos):
    # Method converts timestamp in nanoseconds to datetime
    dt = datetime.datetime.fromtimestamp(nanos / 1e9)
    # return '{}{:03.0f}'.format(dt.strftime('%Y-%m-%d %H:%M:%S.%f'), nanos % 1e3)
    return dt

# Function to get timestamp data for Test
def get_test_duration(test_num, yy_mm_dd):
    
    # Get starting and ending timestamps for each trial
    file = open("Test_Timestamps.csv")
    csv_reader = csv.reader(file)
    # Read csv row by row till we have all trials for test_num
    test_durations = []
    
    for row in csv_reader:
        if row[0] == "Test " + str(test_num):
            trial_duration = []
            format = "%Y-%m-%d %H:%M:%S.%f"
            trial_start_time = row[2]
            start_time = datetime.datetime.strptime((yy_mm_dd + " " + trial_start_time), format).timestamp()
            trial_duration.append(start_time)
            trial_end_time = row[3]
            end_time = datetime.datetime.strptime((yy_mm_dd + " " + trial_end_time), format).timestamp()
            trial_duration.append(end_time)
            test_durations.append(trial_duration)

    return test_durations

class Object:
    def __init__(self, id, velocity, t, encoded_timestamp):
        self.id = id
        self.velocity = velocity
        self.t = t
        self.encoded_timestamp = encoded_timestamp


            
def get_external_object_timestamp(bag, test_num, test_trials):
    print("Getting external object timestamp for test ", test_num)

    # Define csv
    csv_results_filename = "Result_timestamps_Test" + str(test_num) + ".csv"
    csv_results_writer = csv.writer(open(csv_results_filename,'w'))
    csv_results_writer.writerow(["Test Case", "Trial", "Msg ID", "Incoming psm timestamp", "Encoded timestamp", "Speed", "External Object timestamp", "ext_obj_encoded_timestamp"])
    
    
    
    trial_num = 1
    for trial in test_trials:
        unique_psm_ids = []
        psm_objects = []
        for topic, msg, t in bag.read_messages(topics = ["/message/incoming_psm"], start_time = rospy.Time(trial[0], 0), end_time = rospy.Time(trial[1],0)):
            
            # Convert psm id to int
            psm_id = int.from_bytes(msg.id.id, byteorder='big', signed=False)
            if not psm_id in unique_psm_ids:
                unique_psm_ids.append(psm_id)
            
            # Get velocty
            velocity = msg.speed.velocity

            # Get encoded timestamp
            year = msg.path_history.initial_position.utc_time.year.year
            month = msg.path_history.initial_position.utc_time.month.month
            day = msg.path_history.initial_position.utc_time.day.day
            hour = msg.path_history.initial_position.utc_time.hour.hour
            minute = msg.path_history.initial_position.utc_time.minute.minute
            second = round(msg.path_history.initial_position.utc_time.second.millisecond / 60999)
            microsecond = (msg.path_history.initial_position.utc_time.second.millisecond - second) * 1000
            encoded_time = datetime.datetime(year, month, day, hour, minute, second, microsecond)

            # Create object for psm
            psm_object = Object (psm_id, velocity, t, encoded_time)
            psm_objects.append(psm_object)


    # Once we have the unique psm ids, get the associated timestamps from external_object_predictions
    
        curr_trial_psm_timestamps = []
        external_object_predictions = []
        for topic, msg, t in bag.read_messages(topics = ["/environment/external_object_predictions"], start_time = rospy.Time(trial[0], 0), end_time = rospy.Time(trial[1],0)):
            
            if msg.objects: # If objects exist
                for obj in msg.objects:
                    if obj.bsm_id:
                        object_id = int.from_bytes(obj.bsm_id, byteorder='big', signed=False)
                        if object_id in unique_psm_ids:
                            curr_trial_psm_timestamps.append(t)
                            # Convert t to datetime
                            date_time = format_timestamp_nanos(t.to_nsec())
                            ######
                            object_speed = obj.velocity.twist.linear.x
                            header_timestamp = rospy.Time(obj.header.stamp.secs, obj.header.stamp.nsecs)
                            header_datetime = format_timestamp_nanos(header_timestamp.to_nsec())
                            
                            
                            external_object_prediction = Object(object_id, object_speed, t, header_datetime)
                            external_object_predictions.append(external_object_prediction)
                            # csv_results_writer.writerow([test_num, trial_num, object_id, date_time, header_datetime, object_speed])
                    

                        
        
        for psm in psm_objects:
            # Find external object prediction closest to psm
            psm_date_time = format_timestamp_nanos(psm.t.to_nsec())
            found_match = False
            for obj in external_object_predictions:
                if obj.id == psm.id:
                    if abs(obj.velocity - psm.velocity) < 0.0000000001:
                        
                        # if (psm.encoded_timestamp -  obj.encoded_timestamp).total_seconds() * 1000 < datetime.datetime(0,0,0,0,0,0,20).total_seconds * 1000:
                        if abs(psm.encoded_timestamp.timestamp() * 1000 - obj.encoded_timestamp.timestamp() * 1000) < (240 + 1*1.67e-5)* 60000:
                            # Additional check to see if psm_message is behind external_object timestamp
                            if (obj.t.to_nsec() - psm.t.to_nsec()):
                        # diff = (psm.encoded_timestamp - obj.encoded_timestamp)
                                obj_date_time = format_timestamp_nanos(obj.t.to_nsec())
                                # "Test Case", "Trial", "Msg ID", "Incoming psm timestamp", "Encoded timestamp", "Speed", "External Object timestamp"]
                                csv_results_writer.writerow([test_num, trial_num, psm.id, psm_date_time, psm.encoded_timestamp, psm.velocity, obj_date_time, obj.encoded_timestamp])
                                found_match = True
                                break
                
            if found_match is False:
                csv_results_writer.writerow([test_num, trial_num, psm.id, psm_date_time, psm.encoded_timestamp, psm.velocity, "", ""]) 
   
                
        trial_num += 1
        csv_results_writer.writerow(["","","",""])


def main(): 
    # Redirect the output of print() to a specified .txt file
    orig_stdout = sys.stdout
    # current_time = datetime.datetime.now()
    # text_log_filename = "Results_" + str(current_time) + ".txt"
    # text_log_file_writer = open(text_log_filename, 'w')
    # sys.stdout = text_log_file_writer

    bag_files = []
    # List rosbag file names
    bag_files.append("2022-05-24-Test-1-4.bag")
    bag_files.append("2022-05-24-Test-5-2-8.bag")
    bag_files.append("2022-05-23-Test-3-6-9.bag")
    bag_files.append("2022-05-26-Test-7.bag")

    for bag_filename in bag_files:

        # sys.stdout = text_log_file_writer
        print("*****************************************************************")
        print("Processing Bag file: ", bag_filename)

        try:
            print("Starting to process bag at " + str(datetime.datetime.now()))
        except:
            print("Skipping" + bag_filename + ", unable to open or process bag file")
        
        # For each bag file get the name of the tests within it
        yy_mm_dd = bag_filename[0:(bag_filename.find('Test') - 1)]
        sys.stdout = orig_stdout
        start = bag_filename.find('Test') + 5
        end = bag_filename.find('.bag', start)
        tests_string = bag_filename[start:end]
        
        tests_in_bag = re.findall(r'\d+', tests_string)

        for test in tests_in_bag:
            test_duration = get_test_duration(int(test),yy_mm_dd)
            get_external_object_timestamp(rosbag.Bag(bag_filename), int(test), test_duration)
            
        
        print(tests_in_bag)
    


    sys.stdout = orig_stdout
    # text_log_file_writer.close()
    return


if __name__ == "__main__":
    main()