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

## This file generates a plot of lidar detected external_objects and incoming_psm objects 

# Usage:
# python3.7 get_lidar_object_plot in folder containing rosbags and csv of Test data
# 1. The rosbags need to be labelled with yyyy-mm--dd--Test--Testnums (2022-05-24-Test-5-2-8.bag)
# The rosbag file names need to be add to the script main as well
# The Test number represents the run/case. Since this script was initially used for multiple runs for specific cases, 
# the cases recorded in the rosbag help filter the data being looked for
# 2. The csv log file records the Test start and stop timings.
## Since the initial test analysis comprised of multiple Tests in one rosbag, this helps specify the duration to check within the rosbag
# The csv log file needs to be labelled Test_Timestamps.csv
# The format of entries in the csv needs to be TestNum, TrialNum, Start time, End time with the timings at utc time
# (example : Test1, Trial 1,10:46:26.40,10:46:51.30)

def format_timestamp_nanos(nanos):
    # Method converts timestamp in nanoseconds to datetime
    dt = datetime.datetime.fromtimestamp(nanos / 1e9)
    return '{}{:03.0f}'.format(dt.strftime('%Y-%m-%d %H:%M:%S.%f'), nanos % 1e3)

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

class Dynamic_obj:
    def __init__(self, x, y, t, size_x, size_y):
        self.x = x
        self.y = y
        self.t = t
        self.size_x = size_x
        self.size_y = size_y

def euclideanDistance(point1_x, point1_y, point2_x, point2_y):
    
    distance = ((point1_x - point2_x)**2 + (point1_y - point2_y)**2)**0.5
    return distance

def get_psm_object_pose(bag, test_num, test_trials):
    ###### Test Approximation ####
        ### Categorize an object as a pedestrian based on following criteria ###
    #1. object should be a dynamic object
    #2. minumum size of external_object : (size.x * size.y)
    min_size = 0.01
    #3. maximum size of object
    max_size = 0.25
    #4 min speed
    min_average_speed = 0.5
    #5 Max distance from vehicle (in meters)
    max_distance_from_vehicle = 100.0

    trial_num = 1
    for trial in test_trials:
        # Plot the path for dynamic objects
        dynamic_obj_unique_ids = []
        dynamic_objs_in_trial = {} #Dictionary with object list associated with each unique id
        psm_obj_unique_ids = []
        psm_objs_in_trial = {}

        for topic, msg, t in bag.read_messages(topics = ["/environment/external_objects"], start_time = rospy.Time(trial[0], 0), end_time = rospy.Time(trial[1],0)):
            for obj in msg.objects:

                if obj.dynamic_obj is True:
                    
                    # If object speed is 0, skip
                    average_velocity = ((obj.velocity.twist.linear.x)**2 + (obj.velocity.twist.linear.y)**2 + (obj.velocity.twist.linear.z)**2)**0.5
                    if average_velocity < min_average_speed:
                        continue
                    
                    if obj.id not in dynamic_obj_unique_ids: 
                        #If object not already in list, add it to the dictionary
                        dynamic_obj_unique_ids.append(obj.id)
                        dynamic_object = Dynamic_obj(obj.pose.pose.position.x, obj.pose.pose.position.y, t, obj.size.x, obj.size.y)
                        dynamic_objs_in_trial[obj.id] = [dynamic_object]

                    else:
                        #If object already in dictionary, add it to the list for the key
                        dynamic_object = Dynamic_obj(obj.pose.pose.position.x, obj.pose.pose.position.y, t, obj.size.x, obj.size.y)
                        dynamic_objs_in_trial[obj.id] += [dynamic_object]


        for topic, msg, t in bag.read_messages(topics = ["/environment/external_object_predictions"], start_time = rospy.Time(trial[0], 0), end_time = rospy.Time(trial[1],0)):
            for obj in msg.objects:
                if obj.bsm_id: #If object has a bsm id, it is most likely a psm object
                        if obj.id not in psm_obj_unique_ids:
                            psm_obj_unique_ids.append(obj.id)
                            psm_obj = Dynamic_obj(obj.pose.pose.position.x, obj.pose.pose.position.y, t, 0.0, 0.0)
                            psm_objs_in_trial[obj.id] = [psm_obj]
                        else:
                            psm_obj = Dynamic_obj(obj.pose.pose.position.x, obj.pose.pose.position.y, t, 0.0, 0.0)
                            psm_objs_in_trial[obj.id] += [psm_obj]
        
        # get constant position of vehicle 
        vehicle_pose_x = 0.0
        vehicle_pose_y = 0.0
        for topic, msg, t in bag.read_messages(topics = ["/localization/current_pose"], start_time = rospy.Time(trial[0], 0), end_time = rospy.Time(trial[1],0)):
            vehicle_pose_x = msg.pose.position.x
            vehicle_pose_y = msg.pose.position.y

            break

        # Trial reading ends
        print("Number of unique dynamic objects: ", len(dynamic_obj_unique_ids))

        # Plot Dynamic Objects
        qualifying_objects_num = 0 
        for i in dynamic_obj_unique_ids:
            x_coord = []
            y_coord = []
            object_disqualified = False
            for point in dynamic_objs_in_trial[i]:
                point_size = point.size_x * point.size_y
                if point_size < min_size or point_size > max_size:
                    object_disqualified = True
                    break

                x_coord.append(point.x)
                y_coord.append(point.y)

                distance_from_vehicle = euclideanDistance(point.x, point.y, vehicle_pose_x, vehicle_pose_y)
                if distance_from_vehicle > max_distance_from_vehicle :
                    object_disqualified = True
                    break
            
            if object_disqualified is False:
                qualifying_objects_num += 1
                plt.plot(x_coord, y_coord, label = i)

        # Plot PSM Objects
        print("PSM object size: ", len(psm_obj_unique_ids))
        
        for i in psm_obj_unique_ids:
            x_coord = []
            y_coord = []
            for point in psm_objs_in_trial[i]:
                x_coord.append(point.x)
                y_coord.append(point.y)
            # plt.plot(x_coord, y_coord, label = "psm_obj_"+str(i), color = 'red', linestyle = 'dotted')
            plt.scatter(x_coord, y_coord, label = "psm_obj_"+str(i), s = 5)


        print("Test", test_num, " Trial", trial_num)
        print("Number of qualifying objects: ", qualifying_objects_num)
        plt.legend(title = "object id")
        plt_title = "Test" + str(test_num) + ", trial" +  str(trial_num) + " Dynamic Object path min_size: "+ str(min_size)+" max_size: " + str(max_size)
        plt.title(plt_title, fontsize = 12)
        # plt.xlabel('timestamp (nsecs)', fontsize = 9)
        # plt.ylabel('distance from origin (map_frame)', fontsize = 9)
        plt.xlabel('x coordinate (map_frame)', fontsize = 9)
        plt.ylabel('y_coordinate (map_frame)', fontsize = 9)
        # plt.plot(timestamp, distance_from_origin, label = i)
        plt.plot(vehicle_pose_x, vehicle_pose_y, marker = '*', label = 'Vehicle')

        plt.show()

        trial_num += 1
        


def main(): 
    
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
        start = bag_filename.find('Test') + 5
        end = bag_filename.find('.bag', start)
        tests_string = bag_filename[start:end]
        
        tests_in_bag = re.findall(r'\d+', tests_string)

        for test in tests_in_bag:
            if int(test) == 7:
                test_duration = get_test_duration(int(test),yy_mm_dd)
                get_psm_object_pose(rosbag.Bag(bag_filename), int(test), test_duration)
            
        
            
        print(tests_in_bag)
    

    return

if __name__ == "__main__":
    main()