#!/usr/bin/python3

#  Copyright (C) 2021 LEIDOS.
# 
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not
#  use this file except in compliance with the License. You may obtain a copy of
#  the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations under
#  the License.

from inspect import TPFLAGS_IS_ABSTRACT
import sys
import csv
import matplotlib.pyplot as plt
import rospy
import rosbag # To import this, run the following command: "pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag rospkg"
import datetime
import math

# HOW TO USE SCRIPT:
# Run the following in a terminal to download dependencies:
#   sudo add-apt-repository ppa:deadsnakes/ppa
#   sudo apt-get update
#   sudo apt install python3.7
#   python3.7 -m pip install --upgrade pip
#   python3.7 -m pip install matplotlib
#   python3.7 -m pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag rospkg
#   python3.7 -m pip install lz4
#   python3.7 -m pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
# In terminal, navigate to the directory that contains this python script and run the following:
#   python3.7 analyze_fer_rosbags.py <path to folder containing Freight Emergency Response Use Case .bag files>

def generate_speed_plot(bag):
    # Get the vehicle speed and plot it
    speed_received_first_msg = 0.0
    first = True
    times = []
    speeds = []
    crosstracks = []
    downtracks = []
    for topic, msg, t in bag.read_messages(topics=['/hardware_interface/vehicle_status']):
    #for topic, msg, t in bag.read_messages(topics=['/guidance/route_state']):
        if first:
            time_start = t
            first = False
            continue
        
        times.append((t-time_start).to_sec())
        speeds.append(msg.speed * 0.621371) # Conversion from kph to mph
        #crosstracks.append(msg.cross_track)
        #downtracks.append(msg.down_track)
    
    plt.plot(times,speeds)
    #plt.plot(times,crosstracks)
    #plt.plot(times,downtracks)
    plt.show()

    return



# Helper Function: Get the original speed limit for the lanelets within the vehicle's route
# Note: Assumes that all lanelets in the route share the same speed limit prior to the first geofence CARMA Cloud message being processed.#TODO
def get_route_original_speed_limit(bag, bag_file, time_test_start_engagement):
    # Initialize the return variable
    original_speed_limit = 0.0

    # Find the speed limit associated with the first lanelet when the system first becomes engaged
    for topic, msg, t in bag.read_messages(topics=['/guidance/route_state'], start_time = time_test_start_engagement):
        original_speed_limit = msg.speed_limit
        break
    
    if (original_speed_limit is 0.0):
        print("FER-X: (WARNING): No route_state topic detected! for file " + bag_file)

    return original_speed_limit

# Helper Function: Get start and end times of the period of engagement#TODO
def get_test_case_engagement_times(bag):
    # Initialize system engagement start and end times
    time_start_engagement = rospy.Time()
    time_stop_engagement = rospy.Time()

    # Loop through /guidance/state messages to determine start and end times of engagement that include the in-geofence section
    is_engaged = False
    found_engagement_times = False
    for topic, msg, t in bag.read_messages(topics=['/guidance/state']):
        # If entering engagement, track this start time
        if (msg.state == 4 and not is_engaged):
            time_start_engagement = t
            is_engaged = True
        
        # If exiting engagement, check whether this period of engagement included the geofence entrance and exit times
        elif (msg.state != 4 and is_engaged):
            is_engaged = False
            time_stop_engagement = t
            found_engagement_times = True
            break
    
    # Sanity check
    if (time_stop_engagement <= time_start_engagement or not found_engagement_times):
        found_engagement_times = False
        print("Unable to find engagement times")
    
    return time_start_engagement, time_stop_engagement, found_engagement_times

# Helper Function: Print out the times associated with the vehicle entering each new lanelet according to /guidance/route_state#TODO unused function, maybe useful?
def print_lanelet_entrance_times(bag, time_start_engagement):
    # Print out time vehicle enters each lanelet according to /guidance/route_state
    id = 0
    print("/guidance/route_state lanelet change times:")
    for topic, msg, t in bag.read_messages(topics=['/guidance/route_state'], start_time = time_start_engagement):
        if msg.lanelet_id != id:
            print("Time: " + str(t.to_sec()) + "; Lanelet: " + str(msg.lanelet_id) + "; Speed Limit: " + str(msg.speed_limit))
            id = msg.lanelet_id
    
    return


###########################################################################################################
# Freight Emergency Response FER-4: The CMV receives the ERV's BSM without Part II information before the ERV activates its lights and siren.
# Freight Emergency Response FER-5: For any 5-second window, the CMV receives the ERV's BSM without Part II information at a rate >= 8 messages/sec.
# Freight Emergency Response FER-8: The CMV receives the ERV's BSM with Part II information once the ERV is located within the CMV's communication range after it activates its lights and siren.
# Freight Emergency Response FER-9: The CMV receives the ERV's BSM with Part II information once the ERV is located within the CMV's communication range after it activates its lights and siren.
###########################################################################################################
def check_cmv_bsm_related(bag, time_start_engagement, bag_file_name, opposing_direction_bags_set):
    
    ##########
    ## FER-4
    #########
    fer_4 = False
    bsm_topic_name = '/message/incoming_bsm' #TODO
    received_bsm = False
    for topic, msg, t in bag.read_messages(topics=[bsm_topic_name], start_time = time_start_engagement): #TODO
        received_bsm = True
        if (len(list(msg.part_ii)) is 0):
            fer_4 = True
            print("FER-4 Success; Received the ERV's BSM without Part II information before the ERV activates its lights and siren.")
        else:
            print("FER-4 Failure; Received the ERV's BSM WITH Part II information before the ERV activates its lights and siren.")
        break
    
    if (not received_bsm):
        print("FER-4 and FER-5 Failure; CMV never received BSM after engaging.")
        return False, False, False, False, False, False, False
    
    ##########
    ## FER-5 & 8
    #########
    fer_5 = True
    fer_8 = True

    threshold_frequency = 8
    window_duration_to_check = 5
    msgs = bag.read_messages(topics=[bsm_topic_name], start_time = time_start_engagement)
    idx = 0
    msgs_list = list(msgs)

    start_time = None
    end_time = None
    frequency = None
    
    start_idx = -1
    duration = 0

    for topic, msg, t in bag.read_messages(topics=[bsm_topic_name], start_time = time_start_engagement): #TODO
        # if part ii exists, its sirens and lights need to be active
        if (len(list(msg.part_ii)) is not 0 and not fer_8):
            if (msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.lights_use.lightbar_in_use is not 2 and
                msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.siren_use.siren_in_use  is not 2):
                print(f'FER-6 Failed" Detected a BSM part_ii without siren or lightbar in use')
                fer_8 = False

        num_messages = idx - start_idx

        if (start_time is None or duration >= window_duration_to_check):

            start_time = msgs_list[start_idx + 1][2]
            start_idx = start_idx + 1
            duration = 0
        end_time = t

        duration = (end_time - start_time).to_sec()
        print(f'DEBUG idx: {idx}, start_idx: {start_idx}, num_messages: {num_messages}, duration: {duration:.2f}, and from {start_time.to_sec():.1f} to {end_time.to_sec():.1f}')

        if (duration > window_duration_to_check):

            frequency = num_messages / duration
            print(f'DEBUG idx: {idx}, start_idx: {start_idx}, num_messages: {num_messages}, frequency: {frequency:.2f}, duration: {duration:.2f}, and from {start_time.to_sec():.1f} to {end_time.to_sec():.1f}')

            if (frequency < threshold_frequency - 0.05): # accounting for error
                fer_5 = False
                print(f'FER-5 Failure; when after detecting frequency in a 5s window of msgs where frequency is {frequency:.2f} Hz (as opposed to >={threshold_frequency} Hz from {start_time.to_sec():.1f} to {end_time.to_sec():.1f}')
                break
        idx+=1

    total_start_time = msgs_list[0][2]
    total_end_time = msgs_list[len(msgs_list) - 1][2]
    overall_frequency = (len(msgs_list) - 1) / (total_end_time - total_start_time).to_sec()
    print(f'FER-5 (DEBUG): CMV overall frequency of topic {bsm_topic_name} is {overall_frequency:.2f} Hz with total total_num_messages: {len(msgs_list)}, and from {total_start_time.to_sec():.1f} to {total_end_time.to_sec():.1f}')
    if (fer_5):
        print("FER-5 Success; CMV received the ERV's BSM at a rate >= 8 messages/sec.")
    if (fer_8):
        print("FER-8 Success; The CMV receives the ERV's BSM with Part II information once the ERV is located within the CMV's communication range after it activates its lights and siren.")
    
    ##########
    ## FER-9: NOTE: Interpreted as once CMV starts receiving BSM with siren and lightbar in use, it will get that type of BSM at 8 Hz, not just general BSM without part_ii.
    #########
    fer_9 = False
    first = True
    first_part_ii_time = rospy.Time()
    last_part_ii_time = rospy.Time()
    part_ii_msgs = 0
    threshold_frequency = 8
    part_ii_frequency = 0
    for topic, msg, t in bag.read_messages(topics=[bsm_topic_name], start_time = time_start_engagement): #TODO
        if (len(list(msg.part_ii)) is 0):
            continue
            
        if (msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.lights_use.lightbar_in_use is 2 and
                msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.siren_use.siren_in_use  is 2):
            part_ii_msgs+=1
            last_part_ii_time = t

            if (first):
                first_part_ii_time = t
                first = False
            
    if (last_part_ii_time != first_part_ii_time):
        part_ii_frequency = (part_ii_msgs - 1) / (last_part_ii_time - first_part_ii_time).to_sec()
    print(f'FER-9 (DEBUG): CMV part_ii frequency of topic {bsm_topic_name} is {part_ii_frequency:.2f} Hz')
    if (part_ii_frequency < 8):
        print(f'FER-9 Failure; CMV part_ii frequency of topic {bsm_topic_name} is below {threshold_frequency} at {part_ii_frequency:.2f} Hz')
        fer_9 = False
    else:
        print(f'FER-9 Success; CMV part_ii frequency of topic {bsm_topic_name} is above {threshold_frequency} at {part_ii_frequency:.2f} Hz')
        fer_9 = True

    ##########
    ## FER-11:
    #########
    fer_11 = False
    detected_erv = False
    erv_status_topic_name = "/guidance/approaching_erv_status"
    detected_erv = False
    for topic, msg, t in bag.read_messages(topics=[erv_status_topic_name]):
        status_string = msg.msg
        detected_erv, time_until_passing, action = convert_erv_status_string(status_string)

        if detected_erv and bag_file_name not in opposing_direction_bags_set:
            fer_11 = True
            break
    
    if not detected_erv and bag_file_name in opposing_direction_bags_set:
        fer_11 = True

    if (fer_11):
        print(f'FER-11 Success; CMV received the ERVs BSM, it identified whether the emergency vehicle is on the same route as the CMVs.')
    else:
        print(f'FER-11 Failure; CMV did not generate status message indicative of any ERV is approaching. Please check if opposing direction scenario bags are chosen correctly')


    ##########
    ## FER-10 and 12:
    #########
    fer_10 = False
    fer_12 = False
    max_processing_threshold = 1.0

    time_erv_detected = rospy.Time()
    time_bsm_detected = rospy.Time()
    detected_erv = False
    detected_bsm = False
    processing_duration = 0.1

    if (bag_file_name not in opposing_direction_bags_set):

        for topic, msg, t in bag.read_messages(topics=[erv_status_topic_name], start_time = time_start_engagement): #TODO
            status_string = msg.msg
            detected_erv, time_until_passing, action = convert_erv_status_string(status_string)

            if time_until_passing > 0.0:
                time_erv_detected = t
                detected_erv = True
                break
        
        for topic, msg, t in bag.read_messages(topics=[bsm_topic_name], start_time = time_start_engagement): #TODO
            if (len(list(msg.part_ii)) is 0):
                continue
                
            if (msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.lights_use.lightbar_in_use is 2 and
                    msg.part_ii[0].special_vehicle_extensions.vehicle_alerts.siren_use.siren_in_use  is 2):
                time_bsm_detected = t
                detected_bsm = True
                break
        
        if (detected_erv and detected_bsm):
            print(f'FER-10 Success; CMV detected bsm with part ii and detected the ERV and displaying info about it')

            fer_10 = True
            processing_duration = abs((time_erv_detected - time_bsm_detected).to_sec())

    else:
        fer_10 = True
        fer_12 = True
        detected_erv = True
        detected_bsm = True
    
    if (fer_10):
        if (processing_duration <= max_processing_threshold ):
            print(f'FER-12 Success; CMV identifies ERV and displays info under {processing_duration:.5f}, where max threshold is: {max_processing_threshold:.2f}.')
        else:
            print(f'FER-12 Failure; CMV identifies ERV and displays info under {processing_duration:.5f}, where max threshold is: {max_processing_threshold:.2f}.')

    if (not detected_bsm and not detected_erv):
        fer_10 = True
        fer_12 = True
        print(f'FER-10 Success; BSM with part ii was never received so ERV was not detected and info was shown accordingly.')
        print(f'FER-12 Success; BSM with part ii was never received so ERV was not detected and info was shown accordingly.')
    else:
        if (not detected_bsm):
            print(f'FER-10 Failure; CMV did not detect bsm with lights and siren')
            fer_10 = False
        if (not detected_erv):
            print(f'FER-10 Failure; CMV did not detect erv')
            fer_10 = False

    return fer_4, fer_5, fer_8, fer_9, fer_10, fer_11, fer_12

###########################################################################################################
# Freight Emergency Response FER-1: While the ERV is active, for any 5-second window, the ERV shall broadcast BSMs at a rate >= 8 Hz
###########################################################################################################
def check_erv_metrics(bag):
    fer_1 = True
    fer_6 = True
    fer_7 = True
    
    ##########
    ## FER-1
    #########
    topic_name = '/bsm_outbound' 
    threshold_frequency = 8
    window_duration_to_check = 5
    msgs = bag.read_messages(topics=[topic_name])
    idx = 0
    msgs_list = list(msgs)
    total_num_messages = 0

    if (len(msgs_list) is 0):
        print(f'FER-1 (DEBUG): ERV of topic {topic_name} is empty')
    else:
        start_time = None
        end_time = None
        frequency = None
        
        start_idx = -1
        duration = 0

        for topic, msg, t in bag.read_messages(topics=[topic_name]):    #TODO use engagement?
            total_num_messages+=1
            num_messages = idx - start_idx

            if (start_time is None or duration >= window_duration_to_check):

                start_time = msgs_list[start_idx + 1][2]
                start_idx = start_idx + 1
                duration = 0
            end_time = t

            duration = (end_time - start_time).to_sec()
            if (duration > window_duration_to_check):

                frequency = num_messages / duration

                if (frequency < threshold_frequency - 0.05): # accounting for error
                    fer_1 = False
                    print(f'FER-1 Failure; when after detecting frequency in a 5s window of msgs where frequency is {frequency:.2f} Hz from {start_time.to_sec():.1f} to {end_time.to_sec():.1f}')
                    break
            idx+=1

        print(f'FER-1 (DEBUG): ERV broadcasted total of {total_num_messages} msgs')

        total_start_time = msgs_list[0][2]
        total_end_time = msgs_list[len(msgs_list) - 1][2]
        overall_frequency = (total_num_messages - 1) / (total_end_time - total_start_time).to_sec()

        if (fer_1):
            print(f'FER-1 Success; ERV overall frequency of topic {topic_name} is {overall_frequency:.2f} Hz, which is >=8 Hz')

    ##########
    ## FER-34
    #########
    incoming_response_topic = "/incoming_emergency_vehicle_response"
    time_incoming_response_detected = rospy.Time()
    detected_incoming_response = False

    for topic, msg, t in bag.read_messages(topics=[incoming_response_topic]):
        if (not msg.can_change_lanes):
            time_incoming_response_detected = t
            detected_incoming_response = True
            break
    
    time_ack_detected = rospy.Time()
    detected_ack = False
    ack_topic = "/outgoing_emergency_vehicle_ack"
    duration = 0.0
    max_duration = 1.0
    for topic, msg, t in bag.read_messages(topics=[ack_topic]):
        if (msg.acknowledgement):
            time_ack_detected = t
            detected_ack = True
            break
    
    if (detected_incoming_response):
        if (time_ack_detected):
            duration = (time_incoming_response_detected - time_ack_detected).to_sec()
            if (duration <= max_duration):
                print(f'FER-34 Success; ERV broadcasted in {duration:.2f} seconds after receiving the emergency response msg from CMV')
            else:
                print(f'FER-34 Failure; ERV broadcasted in {duration:.2f} seconds after receiving the emergency response msg from CMV')
        else:
            print(f'FER-34 Failure; ERV never broadcasted acknowledgement')
    else:
        fer_34 = True
        print(f'FER-34 Failure; ERV did not get emergency CMV')


    return fer_1, fer_6, fer_7, fer_34

###########################################################################################################
# Freight Emergency Response FER-2: Amount of time that the vehicle is going at steady state (e.g. same lane, constant speed) 
#                before it receives the first approaching message. (> 5 seconds)
# Freight Emergency Response FER-3: The CMV maintains a speed within 2 mph of the speed limit at steady-state
###########################################################################################################
def check_steady_state_before_first_received_message(bag, time_start_engagement, time_first_erv_detected, original_speed_limit):
    fer_2 = False
    fer_3 = True
    
    # (m/s) Threshold offset of vehicle speed to speed limit to be considered at steady state
    threshold_speed_limit_offset = 1.78816 # 1.78816 m/s is 2 mph
    min_steady_state_speed = original_speed_limit - threshold_speed_limit_offset
    max_steady_state_speed = original_speed_limit + threshold_speed_limit_offset

    # (seconds) Minimum time between vehicle reaching steady state and first TIM MobilityOperation message being received
    min_time_between_steady_state_and_erv_detection = 5.0

    # Get the time that the vehicle reaches within the set offset of the speed limit (while system is engaged)
    time_start_steady_state = rospy.Time()
    has_reached_steady_state = False
    for topic, msg, t in bag.read_messages(topics=['/hardware_interface/vehicle/twist'], start_time = time_start_engagement):
        current_speed = msg.twist.linear.x # Current vehicle speed in m/s
        if (not has_reached_steady_state and max_steady_state_speed >= current_speed >= min_steady_state_speed):
            has_reached_steady_state = True
            time_start_steady_state = t

        if (has_reached_steady_state and t <= time_start_steady_state + rospy.Duration(min_time_between_steady_state_and_erv_detection)):
            if (max_steady_state_speed < current_speed or current_speed < min_steady_state_speed):
                print("FER-3 Failure; At steady state, current speed deviates from the speed limit for " + str(abs(current_speed - original_speed_limit)) + " m/s")
                fer_3 = False
                break
    
    # Check if the time the vehicle reaches steady state is more than 'min_time_between_steady_state_and_erv_detection' seconds before the first ERV detection within "approaching" range
    if (has_reached_steady_state):
        time_between_steady_state_and_erv_detection = (time_first_erv_detected - time_start_steady_state).to_sec()
        if (time_between_steady_state_and_erv_detection >= min_time_between_steady_state_and_erv_detection):
            fer_2 = True
            print("FER-2(2) Success; reached steady state " + str(time_between_steady_state_and_erv_detection) + " seconds before receiving first ERV detection.")
        else:
            fer_2 = False
            if (time_between_steady_state_and_erv_detection > 0):
                print("FER-2(2) Failure; reached steady state " + str(time_between_steady_state_and_erv_detection) + " seconds before receiving first ERV detection.")
            else:
                print("FER-2(2) Failure; reached steady state " + str(-time_between_steady_state_and_erv_detection) + " seconds after receiving first ERV detection.")
    else:
        print("FER-2(2) and FER-3 Failure; vehicle never reached steady state during rosbag recording.")
        fer_2 = False
        fer_3 = False

    if (15.63 > original_speed_limit or original_speed_limit > 15.65):
        fer_2 = False
        print(f'FER-2(1) Failure; Speed limit is not 35 mph, but: {(original_speed_limit * 2.23694):.2f} mph')
    else:
        print(f'FER-2(1) Success; Speed limit is 35 mph')
    
    if (fer_3):
        print("FER-3 Success; The CMV maintains a speed within 2 mph of the speed limit at steady-state.")

    return fer_2, fer_3

###########################################################################################################
# Helper function to convert status string into useful variables
###########################################################################################################
def convert_erv_status_string(status_string):
    status = status_string.split(",")
    detected_erv = True if status[0].split(":")[1] is "1" else False
    time_until_passing = float(status[1].split(":")[1])
    action = status[2].split(":")[1]

    return detected_erv, time_until_passing, action

###########################################################################################################
# Helper function to get first time ERV is detected
###########################################################################################################
def get_time_first_erv_detected(bag):

    time_first_msg_received = rospy.Time()
    first = True
    detected_erv = False
    time_until_passing = 0.0
    action = ''
    topic_name = "/guidance/approaching_erv_status"

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        status_string = msg.msg
        detected_erv, time_until_passing, action = convert_erv_status_string(status_string)

        if first and detected_erv:
            time_first_msg_received = t
            first = False
            print(f'FER-X (DEBUG): time_first_msg_received: {time_first_msg_received.to_sec():.2f}')
        
    return time_first_msg_received

##TODO maybe useful?
def euclidean_distance(p1, p2):
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)
    return distance

###########################################################################################################
# Freight Emergency Response FER-14: Once the CMV determines that the ERV is on the same route, and that the CMV will eventually be passed by the ERV, the CMV will determine that it needs to move over.
# Freight Emergency Response FER-15: Once the CMV determines that the ERV is on the same route, and that the CMV will eventually be passed by the ERV, the CMV will determine that it needs to move over to the rightmost lane within 0.1 seconds.
# NOTE: Only expects to process rosbags that includes scenarios where cmd can lane change 
###########################################################################################################
def check_cmv_lane_change_related(bag, bag_file, time_test_start_engagement, time_end_engagement, original_speed_limit):
    print("--- CHECKING LANE CHANGE RELATED TEST CASES, PLEASE MAKE SURE ONLY RELEVANT BAG FILES ARE INCLUDED --- ")

    ##########
    # FER-14
    ##########
    fer_14 = False
    detected_lanechange_needed = rospy.Time()
    topic_name = '/guidance/approaching_erv_status'
    for topic, msg, t in bag.read_messages(topics=[topic_name], start_time = time_test_start_engagement): #TODO
        status_string = msg.msg
        erv_detected, time_until_passing, action = convert_erv_status_string(status_string)
        for s in action.split(' '):
            # if any status contains Attempting string, lanechange is possible
            if ("Attempting" in s):
                fer_14 = True
                detected_lanechange_needed = t
                break
    
    if (fer_14):
        print("FER-14 Success; Once the CMV determines that the ERV is on the same route, and that the CMV will eventually be passed by the ERV, the CMV will determine that it needs to move over")
    else:
        print("FER-14 Failure; CMV did not attempt to lanechange when it should have. Please make sure input bag is correct")

    ##########
    # FER-15
    ##########
    fer_15 = False
    detected_lanechange_maneuver = False
    first_planned_lanechange = rospy.Time()
    # Only run FER-15 if lanechange was detected to be necessary
    if (fer_14):
        maneuver_topic_name = '/guidance/final_maneuver_plan'
        for topic, msg, t in bag.read_messages(topics=[maneuver_topic_name] , start_time = detected_lanechange_needed): #TODO
            for maneuver in msg.maneuvers:
                if (maneuver.type is 1): # 1 is LANE_CHANGE
                    detected_lanechange_maneuver = True
                    first_planned_lanechange = t
                    break
            
    if not detected_lanechange_maneuver:
        print("FER-15 Failure; CMV did not plan lanechange when it should have. Please make sure input bag is correct")

    plan_duration = (first_planned_lanechange - detected_lanechange_needed).to_sec()
    if (detected_lanechange_maneuver and plan_duration <= 1.05): # Less than 1 second
        fer_15 = True
        print(f'FER-15 Success; CMV planned lanechange in: {plan_duration:.2f} seconds')
    elif (detected_lanechange_maneuver and plan_duration > 1.05):
        print(f'FER-15 Failure; CMV planned lanechange in: {plan_duration:.2f} seconds')


    ##########
    # FER-16
    ##########
    fer_16 = False
    detected_lanechange_trajectory = False
    first_planned_lanechange_traj = rospy.Time()

    # Only run FER-15 if lanechange was detected to be necessary
    if (detected_lanechange_maneuver):
        traj_topic_name = '/guidance/trajectory_plan'
        for topic, msg, t in bag.read_messages(topics=[traj_topic_name] , start_time = first_planned_lanechange): #TODO
            for point in msg.trajectory_points:
                if (point.planner_plugin_name is ''): # Only cooperative_lanechange plugin doesn't fill its name
                    detected_lanechange_trajectory = True
                    first_planned_lanechange_traj = t
                    break
            
    if not detected_lanechange_trajectory:
        print("FER-16 Failure; CMV did not generate lanechange trajectory when it should have. Please make sure input bag is correct")

    traj_generation_duration = (first_planned_lanechange_traj - first_planned_lanechange).to_sec()
    if (detected_lanechange_trajectory and traj_generation_duration <= 0.25): # Less than 0.25 
        fer_16 = True
        print(f'FER-16 Success; CMV generated lanechange trajectory in: {traj_generation_duration:.2f} seconds')
    elif (detected_lanechange_trajectory and traj_generation_duration > 0.25):
        print(f'FER-16 Failure; CMV generated lanechange trajectory in: {traj_generation_duration:.2f} seconds')


    ##########
    # FER-22
    ##########
    rightmost_lanes = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 1151, 87879, 17225, 60187, 1335, 
                       26105, 47183, 2036799, 20367, 58713, 26645, 83139, 17970, 14329, 65787, 24943, 76157, 33730, 72139}
    middle_lanes = {200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 2151, 87427, 17010, 60257, 1338, 
                    20153, 58153, 26781, 83551, 18120, 15344, 66681, 25047, 75903, 33521}

    first_lane = '' #MID OR RIGHT
    first_check = True
    first_lanelet = 0
    second_lanelet = 0
    fer_22 = False
    fer_14 = True #TODO

    route_state = '/guidance/route_state'
    route_state_msgs = list(bag.read_messages(topics=[route_state] , start_time = time_test_start_engagement))
    if (len(route_state_msgs) is 0):
        print("FER-22: (WARNING): No route_state topic detected! for file " + bag_file)

    if (fer_14):

        for topic, msg, t in bag.read_messages(topics=[route_state] , start_time = time_test_start_engagement): #TODO
            current_lane = ''
            
            if msg.lanelet_id in rightmost_lanes:
                current_lane = 'RIGHT'
            elif msg.lanelet_id in middle_lanes:
                current_lane = 'MID'
            else:
                print(f'FER-22: (DEBUG): Detected unrecognized lanelet: {msg.lanelet_id}, skipping so we may detect valid lanelet next')
                continue

            if (first_check):
                first_lane = current_lane
                first_check = False
            
            if current_lane is 'MID' and first_lane is 'RIGHT':
                fer_22 = True
                print(f'FER-22 Success; Detected left lanechange from lanelet: {first_lanelet} to {msg.lanelet_id}')

            elif current_lane is 'RIGHT' and first_lane is 'MID':
                fer_22 = True
                print(f'FER-22 Success; Detected right lanechange from lanelet: {first_lanelet} to {msg.lanelet_id}')
            
            if (current_lane is first_lane):
                first_lanelet = msg.lanelet_id
            else:
                second_lanelet = msg.lanelet_id
                break
    
    if (not fer_22):
        print(f'FER-22 Failure; CMV did not perform lanechange')


    ##########
    # FER-23
    ##########
    lanechange_start_time = rospy.Time()
    lanechange_end_time = rospy.Time()
    detected_first = True
    detected_second = False
    detected_lanechange_times = False
    original_crosstrack = 0
    end_crosstrack = 0
    min_lateral_lanechange_speed = 0.5
    max_lateral_lanechange_speed = 1.25
    lateral_speed = 0.0
    for topic, msg, t in bag.read_messages(topics=[route_state] , start_time = time_test_start_engagement): #TODO
        if msg.lanelet_id is first_lanelet and detected_first:
            original_crosstrack = msg.cross_track
            lanechange_start_time = t
            first = False
            continue
        if msg.lanelet_id is second_lanelet:
            detected_second = True
            continue
        if msg.lanelet_id is not second_lanelet and detected_second:    #first lanelet after detecting 2nd lanelet of a lanechange
            lanechange_end_time = t
            detected_lanechange_times = True
            end_crosstrack = msg.cross_track
            break
    
    if detected_lanechange_times:
        cross_track_change = end_crosstrack - original_crosstrack
        time_change = (lanechange_end_time - lanechange_start_time).to_sec()
        lateral_speed = abs(cross_track_change) / time_change
    
    if (min_lateral_lanechange_speed <= lateral_speed <= max_lateral_lanechange_speed):
        print(f'FER-23 Success; Detected lanechange with speed {lateral_speed:.2f} m/s')
        fer_23 = True
    else:
        print(f'FER-23 Failure; Detected lanechange with speed {lateral_speed:.2f} m/s')
        fer_23 = False
    
    ##########
    # FER-26
    ##########

    fer_26 = False
    twist_topic = '/hardware_interface/vehicle/twist'
    lowest_speed = 6.7056 #15mph in m/s which is 20mph lower than 35mph 
    num_consecutive_lowest_speed_required = 10 # Arbitrarily select that topic must show vehicle slowing down for this many consecutive messages to be considered "steady"
    num_consecutive = 0
    lowest_detected_speed = 99.0
    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = lanechange_end_time, end_time = time_end_engagement):
        
        if (msg.twist.linear.x < lowest_speed + 1.1176): #(2.5 mph) buffer
            num_consecutive +=1
            lowest_detected_speed = min(msg.twist.linear.x, lowest_detected_speed)
        else:
            num_consecutive = 0
        if (num_consecutive == num_consecutive_lowest_speed_required):
            fer_26 = True
            break

    if (fer_26):
        print(f'FER-26 Success; Detected reduced speed to {lowest_speed:.2f} m/s')
    else:
        print(f'FER-26 Failure; Detected reduced speed to {lowest_detected_speed:.2f} m/s')


    ##########
    # FER-27
    ##########
    fer_27 = False
    max_avg_deceleration = 1.0 # m/s^2
    end_decel_speed = lowest_speed + 1.1176 # Add 1.1176 m/s (2.5 mph) buffer to the expected end-of-deceleration speed
    speed_start_decel_ms = 0.0
    speed_end_decel_ms = 0.0

    # Obtain timestamp associated with the end of the deceleration section 
    # Note: This is the moment when the vehicle's speed reaches the advisory speed limit    #TODO this next logic seems to be dependent on lanechange logic, not sure if it doesnt need to be
    first = True
    time_end_decel = rospy.Time()
    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = lanechange_end_time, end_time = time_end_engagement):
        # Obtain the speed at the start of the deceleration section
        if first:
            speed_start_decel_ms = msg.twist.linear.x
            first = False

            # Print Debug Line
            speed_start_decel_mph = msg.twist.linear.x * 2.2639 # 2.2369 mph is 1 m/s
            print(f'FER-27(2) (DEBUG): Speed at start of deceleration section: {speed_start_decel_mph:.2f} mph ({speed_start_decel_ms:.2f} m/s)')
            continue

        current_speed_ms = msg.twist.linear.x
        if (current_speed_ms <= end_decel_speed):
            time_end_decel = t
            speed_end_decel_ms = current_speed_ms

            # Print Debug Line
            speed_end_decel_mph = speed_end_decel_ms * 2.2369 # 2.2369 mph is 1 m/s
            print(f'FER-27(2) (DEBUG): Speed at end of deceleration section: {speed_end_decel_mph:.2f} mph ({speed_end_decel_ms:.2f} m/s)')
            break


    # Calculate the average deceleration across the full deceleration section
    print(f'FER-27(2) (DEBUG): Duration between start and end of deceleration: {(time_end_decel-lanechange_end_time).to_sec():.2f})')
    if (speed_start_decel_ms is speed_end_decel_ms):
        total_average_decel = 0.0
    else:
        total_average_decel = (speed_start_decel_ms - speed_end_decel_ms) / (time_end_decel - lanechange_end_time).to_sec()
    
    if total_average_decel <= max_avg_deceleration + 0.05:  # accounting for error
        print(f'FER-27(2) Success; average deceleration after lanechanging was {total_average_decel:.2f} m/s^2')
        fer_27 = True
    else:
        print(f'FER-27(2) Failure; average deceleration after lanechanging was {total_average_decel:.2f} m/s^2')

    threshold_deceleration = 2.0
    window_duration_to_check = 1
    twist_msgs = bag.read_messages(topics=[twist_topic])
    idx = 0
    twist_msgs_list = list(twist_msgs)

    decel_start_time = rospy.Time()
    decel_end_time = rospy.Time()
    decel_rate = 0
    
    start_idx = -1
    duration = 0
   
    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = lanechange_end_time, end_time = time_end_engagement):

        speed_difference = abs(twist_msgs_list[idx][1].twist.linear.x - twist_msgs_list[start_idx][1].twist.linear.x)

        if (decel_start_time is None or duration >= window_duration_to_check):
            decel_start_time = twist_msgs_list[start_idx + 1][2]
            start_idx = start_idx + 1
            duration = 0

        decel_end_time = t

        duration = (decel_end_time - decel_start_time).to_sec()
        if (duration > window_duration_to_check):

            decel_rate = speed_difference / duration

            if (decel_rate > threshold_deceleration + 0.05): # accounting for error
                fer_27 = False
                print(f'FER-27(1) Failure; when after detecting decel_rate in a 1s window of twist_msgs where decel_rate is {decel_rate:.2f} m/s^2 from {decel_start_time.to_sec():.1f} to {decel_end_time.to_sec():.1f}')
                break
        idx+=1
    
    if fer_27:
        print(f'FER-27(1) Success; when after detecting decel_rate in a 1s window of twist_msgs where decel_rate is {decel_rate:.2f} m/s^2 from {decel_start_time.to_sec():.1f} to {decel_end_time.to_sec():.1f}')


    ##########
    # FER-28
    ##########
    fer_28 = False
    # After decel end time that was previously calculated, the ERV should have more than 10s until passing (arbitrary but close to passing_threshold of 13s)
    erv_status_topic = "/guidance/approaching_erv_status"
    time_until_passing = 0.0
    for topic, msg, t in bag.read_messages(topics=[erv_status_topic], start_time = lanechange_end_time, end_time = time_end_engagement):
        if (t >= decel_end_time):
            status_string = msg.msg
            erv_detected, time_until_passing, action = convert_erv_status_string(status_string)
            if (time_until_passing > 10.0):
                fer_28 = True
                break

    if fer_28:
        print(f'FER-28 Success; when CMV finished decelerating, time_until_passing was {time_until_passing:.2f}s compared to 10s')
    else:
        print(f'FER-28 Failure; when CMV finished decelerating, time_until_passing was {time_until_passing:.2f}s compared to 10s')

    
    #########
    # FER-29
    ##########
    fer_29 = False

    detected_left_lanechange_maneuver = False
    detected_right_lanechange_maneuver = False

    first_planned_lanechange = rospy.Time()
    # Only run FER-29 if lanechange was detected to be necessary
    
    if (fer_14):
        maneuver_topic_name = '/guidance/final_maneuver_plan'
        for topic, msg, t in bag.read_messages(topics=[maneuver_topic_name] , start_time = detected_lanechange_needed):#TODO
            for maneuver in msg.maneuvers:
                if (maneuver.type is 1 and
                    int(maneuver.lane_change_maneuver.starting_lane_id) in middle_lanes and
                    int(maneuver.lane_change_maneuver.ending_lane_id) in rightmost_lanes): # 1 is LANE_CHANGE
                        detected_right_lanechange_maneuver = True
                if (maneuver.type is 1 and
                    int(maneuver.lane_change_maneuver.starting_lane_id) in rightmost_lanes and
                    int(maneuver.lane_change_maneuver.ending_lane_id) in middle_lanes): # 1 is LANE_CHANGE
                        detected_left_lanechange_maneuver = True
                
                if (detected_left_lanechange_maneuver and detected_right_lanechange_maneuver):
                    fer_29 = True
                    break
    
    if fer_29:
        print(f'FER-29 Success; detected two lanechanges total meaning the vehicle returned to its original lane.')
    else:
        print(f'FER-29 Failure; did not detect two lanechanges')


    #########
    # FER-30(2) average acceleration must be below max_avg_acceleration
    ##########
    fer_30 = False
    max_avg_acceleration = 1.0 # m/s^2
    end_accel_speed = original_speed_limit - 1.1176 # Add 1.1176 m/s (2.5 mph) buffer to the expected end-of-acceleration speed
    speed_start_accel_ms = 0.0
    speed_end_accel_ms = 0.0
    num_consecutive_accel_required = 20 # Arbitrarily select that topic must show vehicle speeding up for this many consecutive messages to be considered the start of acceleration
    time_begin_acceleration_after_decel = rospy.Time()

     # Obtain timestamp associated with start of the acceleration section 
    first = True
    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = decel_end_time, end_time = time_end_engagement): # time_start_engagement+time_duration):
        if first:
            prev_speed = msg.twist.linear.x
            first = False
            continue

        if msg.twist.linear.x - prev_speed > 0:
            num_consecutive_accel +=1
            if num_consecutive_accel == num_consecutive_accel_required:
                time_begin_acceleration_after_decel = t
                break
            
        else:
            num_consecutive_accel = 0

        prev_speed = msg.twist.linear.x

    # Obtain timestamp associated with end of the acceleration section 
    time_end_accel = rospy.Time()
    first = True
    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = time_begin_acceleration_after_decel, end_time = time_end_engagement): # rospy.Duration(1) for error
        
        # Obtain the speed at the start of the acceleration section
        if first:
            speed_start_accel_ms = msg.twist.linear.x
            first = False

            # Print Debug Line
            speed_start_accel_mph = msg.twist.linear.x * 2.2639 # 2.2369 mph is 1 m/s
            print(f'FER-30(2) (DEBUG): Speed at start of acceleration section: {speed_start_accel_mph:.2f} mph ({speed_start_accel_ms:.2f} m/s)')
            continue

        current_speed_ms = msg.twist.linear.x
        if (current_speed_ms >= end_accel_speed):
            time_end_accel = t
            speed_end_accel_ms = current_speed_ms

            # Print Debug Line
            speed_end_accel_mph = speed_end_accel_ms * 2.2369 # 2.2369 mph is 1 m/s
            print(f'FER-30(2) (DEBUG): Speed at end of acceleration section: {speed_end_accel_mph:.2f} mph ({speed_end_accel_ms:.2f} m/s)')
            break


    # Calculate the average acceleration across the full acceleration section
    print(f'FER-30(2) (DEBUG): Duration between start and end of acceleration: {(time_end_accel-time_begin_acceleration_after_decel).to_sec():.2f})')
    if (speed_start_accel_ms is speed_end_accel_ms):
        total_average_accel = 0.0
    else:
        total_average_accel = (speed_start_accel_ms - speed_end_accel_ms) / (time_end_accel - time_begin_acceleration_after_decel).to_sec()
    
    if total_average_accel <= max_avg_acceleration + 0.05:  # accounting for error
        print(f'FER-30(2) Success; average acceleration after lanechanging was {total_average_accel:.2f} m/s^2')
        fer_30 = True
    else:
        print(f'FER-30(2) Failure; average acceleration after lanechanging was {total_average_accel:.2f} m/s^2')

    #########
    # FER-30(1) any 1 second window must have acceleration rate lower than 2m/s^2
    ##########

    threshold_acceleration = 2.0
    window_duration_to_check = 1
    twist_msgs = bag.read_messages(topics=[twist_topic])
    idx = 0
    twist_msgs_list = list(twist_msgs)

    accel_start_time = rospy.Time()
    accel_end_time = rospy.Time()
    accel_rate = 0
    start_idx = -1
    duration = 0

    for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time  = time_begin_acceleration_after_decel, end_time = time_end_accel):

        speed_difference = abs(twist_msgs_list[idx][1].twist.linear.x - twist_msgs_list[start_idx][1].twist.linear.x)

        if (accel_start_time is None or duration >= window_duration_to_check):
            accel_start_time = twist_msgs_list[start_idx + 1][2]
            start_idx = start_idx + 1
            duration = 0

        accel_end_time = t

        duration = (accel_end_time - accel_start_time).to_sec()
        if (duration > window_duration_to_check):

            accel_rate = speed_difference / duration

            if (accel_rate > threshold_acceleration + 0.05): # accounting for error
                fer_30 = False
                print(f'FER-30(1) Failure; when after detecting accel_rate in a 1s window of twist_msgs where accel_rate is {accel_rate:.2f} m/s^2 from {accel_start_time.to_sec():.1f} to {accel_end_time.to_sec():.1f}')
                break
        idx+=1

    if fer_30:  # accounting for error
        print(f'FER-30(1) Success; when after detecting accel_rate in a 1s window of twist_msgs where accel_rate is {accel_rate:.2f} m/s^2 from {accel_start_time.to_sec():.1f} to {accel_end_time.to_sec():.1f}')

    return fer_14, fer_15, fer_16, fer_22, fer_23, fer_26, fer_27, fer_28, fer_29, fer_30


def check_cmv_hazard_related(bag, time_test_start_engagement, time_test_end_engagement, original_speed_limit):
    
    print("--- CHECKING CMV CANNOT LANE CHANGE / HAZARD SCENARIO TEST CASES, PLEASE MAKE SURE ONLY RELEVANT BAG FILES ARE CONSIDERED ---")

    #########
    # FER-32
    ##########

    fer_32 = False
    topic_name_approaching = "/guidance/approaching_erv_status"
    topic_name_warning = "/message/outgoing_emergency_vehicle_response"
    response_duration = 0.0
    detected_hazard = False
    time_first_detected_hazard = rospy.Time()

    for topic, msg, t in bag.read_messages(topics=[topic_name_approaching], start_time = time_test_start_engagement, end_time = time_test_end_engagement):#TODO
        status_string = msg.msg
        detected_erv, time_until_passing, action = convert_erv_status_string(status_string)
        if 'possible.' in action.split(' '): #word not possible is only in every status if lanechange is not possible
            time_first_detected_hazard = t
            break

    for topic, msg, t in bag.read_messages(topics=[topic_name_warning], start_time = time_first_detected_hazard, end_time = time_test_end_engagement):
        if (not msg.can_change_lanes):
            response_duration = abs((t - time_first_detected_hazard).to_sec())
            detected_hazard = True
            break
    
    if (detected_hazard):
        if (response_duration <= 0.1):
            fer_32 = True
            print(f'FER-32 Success; when after hazard response duration was {response_duration:.2f}s which should be less than 0.1s')
        else:
            fer_32 = False
            print(f'FER-32 Failure; when after hazard response duration was {response_duration:.2f}s which should be less than 0.1s')
    else:
        fer_32 = False
        print(f'FER-32 Failure; the CMV never broadcasted cannot lane change warning to the ERV')

    #########
    # FER-33
    ##########

    fer_33 = False
    twist_topic = '/hardware_interface/vehicle/twist'
    lowest_speed = 6.7056 #15mph in m/s which is 20mph lower than 35mph 
    num_consecutive_lowest_speed_required = 10 # Arbitrarily select that topic must show vehicle slowing down for this many consecutive messages to be considered "steady"
    num_consecutive = 0
    lowest_detected_speed = 99.0
    
    if (detected_hazard):
        for topic, msg, t in bag.read_messages(topics=[twist_topic], start_time = time_first_detected_hazard, end_time = time_test_end_engagement):
            lowest_detected_speed = min(msg.twist.linear.x, lowest_detected_speed)
            if (msg.twist.linear.x < lowest_speed + 1.1176): #(2.5 mph) buffer
                num_consecutive +=1
            else:
                num_consecutive = 0

            if (num_consecutive == num_consecutive_lowest_speed_required):
                fer_33 = True
                break

    if (fer_33):
        print(f'FER-33 Success; Detected reduced speed to {lowest_speed:.2f} m/s')
    else:
        print(f'FER-33 Failure; Detected lowest reduced speed to {lowest_detected_speed:.2f} m/s where it should have been {lowest_speed:.2f} m/s')

    #########
    # FER-35
    ##########
    fer_35 = False
    topic_name_ack = "/message/incoming_emergency_vehicle_ack"
    detected_ack_time = rospy.Time()
    detected_ack = False
    max_number_of_warnings = 10
    num_of_warnings = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name_ack], start_time = time_test_start_engagement, end_time = time_test_end_engagement):#TODO
        
        if (msg.acknowledgement):
            detected_ack_time = t
            detected_ack = True
            break
        
    last_warning_time = rospy.Time()

    for topic, msg, t in bag.read_messages(topics=[topic_name_warning], start_time = time_test_start_engagement, end_time = time_test_end_engagement):
        if (not msg.can_change_lanes):
            num_of_warnings += 1
            last_warning_time = t

    if (detected_ack and last_warning_time <= detected_ack_time + rospy.Duration(0.1) and 
        num_of_warnings <= max_number_of_warnings):
        fer_35 = True
    
    if (fer_35):
        print(f'FER-35 Success; Detected number of warnings: {max_number_of_warnings}, and last_warning_time: {last_warning_time.to_sec():.2f}, where detected_ack_time was: {detected_ack_time.to_sec():.2f}')
    else:
        print(f'FER-35 Failure; Detected number of warnings: {max_number_of_warnings}, and last_warning_time: {last_warning_time.to_sec():.2f}, where detected_ack_time was: {detected_ack_time.to_sec():.2f}')

    return fer_32, fer_33, fer_35


# Main Function; run all tests from here
def main():  
    if len(sys.argv) < 2:
        print("Need 1 arguments: process_bag.py <path to source folder with .bag files> ")
        exit()
    
    source_folder = sys.argv[1]

    # Re-direct the output of print() to a specified .txt file:
    orig_stdout = sys.stdout
    current_time = datetime.datetime.now()

    # Create list of White
    white_truck_bag_files = [
                                "WT_8.4.1_R1.bag",
                                "WT_8.4.2_R1.bag",
                                "WT_8.4.3_R1.bag",
                                "WT_8.4.4_R1.bag",
                                "WT_8.5.3_R1.bag",
                                "WT_8.6.10_R1.bag",
                                "WT_8.6.1_R1.bag",
                                "WT_8.6.1_R2.bag",
                                "WT_8.6.1_R3.bag",
                                "WT_8.6.1_R4.bag",
                                "WT_8.6.1_R5.bag",
                                "WT_8.6.2_R1.bag",
                                "WT_8.6.2_R2.bag",
                                "WT_8.6.2_R3.bag",
                                "WT_8.6.2_R4.bag",
                                "WT_8.6.2_R5.bag",
                                "WT_8.6.3_R1.bag",
                                "WT_8.6.3_R2.bag",
                                "WT_8.6.3_R3.bag",
                                "WT_8.6.3_R4.bag",
                                "WT_8.6.4_R1.bag",
                                "WT_8.6.4_R2.bag",
                                "WT_8.6.4_R3.bag",
                                "WT_8.6.4_R4.bag",
                                "WT_8.6.4_R5.bag",
                                "WT_8.6.4_R6.bag",
                                "WT_8.6.4_R7.bag",
                                "WT_8.6.6_R1.bag",
                                "WT_8.6.6_R2.bag",
                                "WT_8.6.6_R3.bag",
                                "WT_8.6.6_R4.bag",
                                "WT_8.6.6_R5.bag",
                                "WT_8.6.6_R6.bag",
                                "WT_8.6.7_R1.bag",
                                "WT_8.6.7_R2.bag",
                                "WT_8.6.7_R3.bag",
                                "WT_8.6.7_R4.bag"
                                ]
     

    # Create list of Silver
    silver_truck_bag_files = []
    
    # Create list of CMV
    cmv_bag_files = white_truck_bag_files + silver_truck_bag_files

    # Bags where ERV should NOT be detected by CMV, like opposing lane
    cmv_should_not_detect_erv_files = []

    # Bags where ERV should NOT lanechange such as when not lanechange is possible
    cmv_should_not_lanechange_files = []
    
    # Bags where ERV should be detected by CMV
    cmv_should_lanechange_files = list(set(cmv_bag_files) - set(cmv_should_not_lanechange_files) -  set(cmv_should_not_detect_erv_files))

    # Create list of ERV
    erv_bag_files = ["TA_8.6.10_R1.bag"
"TA_8.6.11_R1.bag",
"TA_8.6.11_R2.bag",
"TA_8.6.11_R3.bag",
"TA_8.6.11_R4.bag",
"TA_8.6.11_R5.bag",
"TA_8.6.11_R6.bag",
"TA_8.6.1_R1.bag",
"TA_8.6.1_R2.bag",
"TA_8.6.1_R3.bag",
"TA_8.6.1_R4.bag",
"TA_8.6.1_R5.bag",
"TA_8.6.2_R2.bag",
"TA_8.6.2_R3.bag",
"TA_8.6.2_R4.bag",
"TA_8.6.2_R5.bag",
"TA_8.6.3_R1.bag",
"TA_8.6.3_R2.bag",
"TA_8.6.3_R3.bag",
"TA_8.6.3_R4.bag",
"TA_8.6.3_R5.bag",
"TA_8.6.3_R6.bag",
"TA_8.6.3_R7.bag",
"TA_8.6.3_R8.bag",
"TA_8.6.4_R1.bag",
"TA_8.6.4_R2.bag",
"TA_8.6.4_R3.bag",
"TA_8.6.4_R4.bag",
"TA_8.6.4_R5.bag",
"TA_8.6.4_R6.bag",
"TA_8.6.4_R7.bag",
"TA_8.6.5_R1.bag",
"TA_8.6.5_R2.bag",
"TA_8.6.6_R1.bag",
"TA_8.6.6_R2.bag",
"TA_8.6.6_R4.bag",
"TA_8.6.6_R5.bag",
"TA_8.6.6_R6.bag",
"TA_8.6.7_R1.bag",
"TA_8.6.7_R3.bag",
"TA_8.6.7_R4.bag",
"TA_8.6.9_R1.bag",
"TA_8.6.9_R2.bag",
"TA_8.6.9_R3.bag",
"TA_8.6.9_R4.bag",
"TA_8.6.9_R5.bag"

                    ]
    


    # Concatenate all Basic Travel bag files into one list
    fer_bag_files = white_truck_bag_files + silver_truck_bag_files + erv_bag_files

    # Loop to conduct data analysis on each bag file:
    for bag_file in fer_bag_files:
        print("*****************************************************************")
        print("Processing new bag: " + str(bag_file))
        if bag_file in white_truck_bag_files:
            print("White Truck CMV Test Case")
        elif bag_file in silver_truck_bag_files:
            print("Silver Truck CMV Test Case")
        elif bag_file in erv_bag_files:
            print("ERV Test Case")
        else:
            print("Unknown bag file being processed.")
            
        # Print processing progress to terminal (all other print statements are re-directed to outputted .txt file):
        print("Processing bag file " + str(bag_file) + " (" + str(fer_bag_files.index(bag_file) + 1) + " of " + str(len(fer_bag_files)) + ")")

        # Process bag file if it exists and can be processed, otherwise skip and proceed to next bag file
        try:
            print("Starting To Process Bag at " + str(datetime.datetime.now()))
            bag_file_path = str(source_folder) + "/" + bag_file
            bag = rosbag.Bag(bag_file_path)
            print("Finished Processing Bag at " + str(datetime.datetime.now()))
        except:
            print("Skipping " + str(bag_file) +", unable to open or process bag file.")
            continue
        
        if (bag_file not in erv_bag_files):
            # Get the rosbag times associated with the starting engagement and ending engagement for the Basic Travel use case test
            print("Getting engagement times at " + str(datetime.datetime.now()))
            time_test_start_engagement, time_test_end_engagement, found_test_times = get_test_case_engagement_times(bag)
            print("Got engagement times at " + str(datetime.datetime.now()))
            if (not found_test_times):
                print("Could not find test case engagement start and end times in bag file.")
                #continue#TODO

            original_speed_limit = get_route_original_speed_limit(bag, bag_file, time_test_start_engagement) # Units: m/s
            print("Original Speed Limit is " + str(original_speed_limit) + " m/s")

        print("--------------------------TEST CASES---------------------------------")

        # Initialize results 
        fer_1_result = None
        fer_2_result = None
        fer_3_result = None
        fer_4_result = None
        fer_5_result = None
        fer_6_result = None #skipped
        fer_7_result = None #skipped
        fer_8_result = None
        fer_9_result = None
        fer_10_result = None
        fer_11_result = None
        fer_12_result = None
        fer_13_result = None
        fer_14_result = None
        fer_15_result = None
        fer_16_result = None
        fer_17_result = None
        fer_18_result = None
        fer_19_result = None
        fer_20_result = None
        fer_21_result = None
        fer_22_result = None
        fer_23_result = None
        fer_24_result = None
        fer_25_result = None
        fer_26_result = None
        fer_27_result = None
        fer_28_result = None
        fer_29_result = None
        fer_30_result = None
        fer_31_result = None
        fer_32_result = None
        fer_33_result = None
        fer_34_result = None
        fer_35_result = None

        if (bag_file in cmv_bag_files):
            time_first_erv_detected = get_time_first_erv_detected(bag)
            fer_4_result, fer_5_result, fer_8_result, fer_9_result, fer_10_result, fer_11_result, fer_12_result = check_cmv_bsm_related(bag, time_test_start_engagement, bag_file, set(cmv_should_not_detect_erv_files))
            fer_2_result, fer_3_result = check_steady_state_before_first_received_message(bag, time_test_start_engagement, time_first_erv_detected, original_speed_limit)

        if (bag_file in cmv_should_lanechange_files):
            fer_14_result, fer_15_result, fer_16_result, fer_22_result, fer_23_result, fer_26_result, fer_27_result, fer_28_result, fer_29_result, fer_30_result = check_cmv_lane_change_related(bag, bag_file, time_test_start_engagement, time_test_end_engagement, original_speed_limit)

        if (bag_file in cmv_should_not_lanechange_files):
            fer_32_result, fer_33_result, fer_35_result = check_cmv_hazard_related(bag, time_test_start_engagement, time_test_end_engagement, original_speed_limit)

        if bag_file in erv_bag_files:
            fer_1_result, fer_6_result, fer_7_result, fer_34_result = check_erv_metrics(bag)
        
    return

if __name__ == "__main__":
    main()