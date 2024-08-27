import numpy as np
import argparse, argcomplete
import unittest
import os
import sys
from tqdm import tqdm
from rclpy.logging import set_logger_level, get_logging_severity_from_string
from functools import partialmethod

from check_distance_to_arrival import check_distance_to_arrival
from check_message_published import check_message_published
from check_message_timing import check_message_timing
from check_port_drayage_ack import check_port_drayage_ack
from plot_crosstrack_error import plot_crosstrack_error
from plot_vehicle_speed import plot_vehicle_speed


class C1TMetricAnalysis(unittest.TestCase):

    def setUp(self):
        # Save the passed in bag directory, disable tqdm and rosbag2_storage logging
        self.bag_dir = sys.argv[1]
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        set_logger_level("rosbag2_storage", get_logging_severity_from_string("FATAL"))
        

    def test_C1T_01_generates_route_from_MOM(self):
        # The C1T CMV is able to receive destinations as Mobility Operation messages from the infrastructure and generate a route
        self.assertTrue(check_message_published(self.bag_dir, "/plan"), "Message was not received on /plan")

    def test_C1T_02_generates_route_from_rviz_timing(self):
        # The C1T CMV will generate a route within 3 seconds of receiving a goal pose from Rviz
        time_between_messages = check_message_timing(self.bag_dir, "/goal_pose", "/plan")
        self.assertTrue(np.all(time_between_messages < 3.0), "Time between messages on /goal_pose and /plan exceeded 3 seconds")

    def test_C1T_03_generates_route_from_MOM_timing(self):
        # The C1T CMV will generate a route within 3 seconds of receiving an incoming mobility operation message
        time_between_messages = check_message_timing(self.bag_dir, "/incoming_mobility_operation", "/plan")
        self.assertTrue(np.all(time_between_messages < 3.0), "Time between messages on /incoming_mobility_operation and /plan exceeded 3 seconds")

    def test_C1T_04_follow_route(self):
        # The C1T CMV will follow its generate routed to the destination with less than 0.4 m of crosstrack error 
        _, crosstrack_errors = plot_crosstrack_error(self.bag_dir, "/plan", show_plots=False)
        self.assertTrue(np.all(crosstrack_errors < 0.4), "Crosstrack error from planned route exceeded 0.4 m")

    def test_C1T_05_follows_road_network(self):
        # The C1T CMV will drive along the centerline of the defined road network with less than 0.2 m of crosstrack error
        _, crosstrack_errors = plot_crosstrack_error(self.bag_dir, "/route_graph", show_plots=False)
        self.assertTrue(np.all(crosstrack_errors < 0.2), "Crosstrack error from road network exceeded 0.2 m")

    def test_C1T_06_maintains_speed_on_straights(self):
        # The C1T CMV achieves and maintains its target speed with a tolerance of 0.2 m/s (excluding turns)
        velocities, target_velocities = plot_vehicle_speed(self.bag_dir, show_plots=False)
        max_target_speed = np.max(target_velocities)
        wait_for_vehicle_to_accelerate = 0
        relevant_velocity_idxs = []
        # Add a buffer of ~2 seconds (odom is 50 Hz) to allow the vehicle to accelerate before it is expected to reach its maximum speed
        for velocity in target_velocities:
            if velocity == max_target_speed:
                if wait_for_vehicle_to_accelerate > 100:
                    relevant_velocity_idxs.append(True)
                else:
                    relevant_velocity_idxs.append(False)
                wait_for_vehicle_to_accelerate += 1
            else:
                relevant_velocity_idxs.append(False)
                wait_for_vehicle_to_accelerate = 0
        speeds_on_straights = velocities[relevant_velocity_idxs]
        self.assertTrue(np.all(np.abs(max_target_speed - speeds_on_straights) < 0.2), "Vehicle deviated more than 0.2 m/s from target speed on straight")

    def test_C1T_07_slowdown_on_turns(self):
        # The C1T CMV will not slowdown less than 50% of the target speed during turns
        velocities, target_velocities = plot_vehicle_speed(self.bag_dir, show_plots=False)
        max_target_speed = np.max(target_velocities)
        # Determine which slowdowns are due to turns and which slowdowns are due to entering/exiting goals
        stop_detected = True  # Vehicle will originally be stopped
        buffer = 0  # Buffer used to count previous velocities that may or may not be due to entering/exiting a goal
        relevant_velocity_idxs = []
        for velocity in target_velocities:
            if velocity == max_target_speed:
                relevant_velocity_idxs += (buffer + 1) * [False]
                buffer = 0
                stop_detected = False
            elif stop_detected:
                relevant_velocity_idxs.append(False)    
            elif velocity == 0.0:
                relevant_velocity_idxs += (buffer + 1) * [False]
                buffer = 0
                stop_detected = True
            else:
                buffer += 1
        speeds_on_turns = velocities[relevant_velocity_idxs]
        self.assertTrue(np.all(speeds_on_turns > 0.5 * max_target_speed), "Vehicle speed reduced more than 50% on turn")

    def test_C1T_08_stops_close_to_destination(self):
        # The C1T CMV stops within 0.2 m of the goal destinations
        arrival_distances = check_distance_to_arrival(self.bag_dir)
        self.assertTrue(np.all(arrival_distances < 0.2), "Vehicle did not stop within 0.2 m of goal destination")

    def test_C1T_09_acks_with_MOM(self):
        # The C1T CMV sends Mobility Operation message acknowledging when a goal is reached
        self.assertTrue(check_message_published(self.bag_dir, "/outgoing_mobility_operation"), "Message not received on /outgoing_mobility_operation")

    def test_C1T_10_infrastructure_response_to_ack_timing(self):
        # Infrastructure responds within a new goal destination within 1.5 seconds of receiving an ack from the C1T CMV
        time_between_messages = check_message_timing(self.bag_dir, "/outgoing_mobility_operation", "/incoming_mobility_operation")
        self.assertTrue(np.all(time_between_messages < 1.5), "Time between ack and new goal exceed 1.5 seconds")

    def test_C1T_12_communicate_load_and_unload(self):
        # Infrastructure can communicate with the C1T CMV that the container is loaded and unloaded
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "PICKUP"), "Vehicle did not ack PICKUP correctly")
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "DROPOFF"), "Vehicle did not ack DROPOFF correctly")

    def test_C1T_13_communicate_inspection(self):
        # Infrastructure can communicate with the C1T CMV that the container is inspected
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "HOLDING_AREA"), "Vehicle did not ack HOLDING_AREA corectly")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run C1T metric analysis")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load and analyze")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    sys.argv[1] = os.path.normpath(os.path.abspath(argdict["bag_in"]))
    unittest.main(verbosity=2, argv=[sys.argv[0]])
