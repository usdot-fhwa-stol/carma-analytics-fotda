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
        self.bag_dir = sys.argv[1]
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        set_logger_level("rosbag2_storage", get_logging_severity_from_string("FATAL"))
        

    def test_C1T_01(self):
        self.assertTrue(check_message_published(self.bag_dir, "/plan"))

    def test_C1T_02(self):
        time_between_messages = check_message_timing(self.bag_dir, "/goal_pose", "/plan")
        self.assertTrue(np.all(time_between_messages < 3.0))

    def test_C1T_03(self):
        time_between_messages = check_message_timing(self.bag_dir, "/incoming_mobility_operation", "/plan")
        self.assertTrue(np.all(time_between_messages < 3.0))

    def test_C1T_04(self):
        _, crosstrack_errors = plot_crosstrack_error(self.bag_dir, "/plan", show_plots=False)
        self.assertTrue(np.all(crosstrack_errors < 0.4))

    def test_C1T_05(self):
        _, crosstrack_errors = plot_crosstrack_error(self.bag_dir, "/route_graph", show_plots=False)
        self.assertTrue(np.all(crosstrack_errors < 0.2))

    def test_C1T_06(self):
        velocities, target_velocities = plot_vehicle_speed(self.bag_dir, show_plots=False)
        max_target_speed = np.max(target_velocities)
        speeds_on_straights = velocities[target_velocities == max_target_speed]
        self.assertTrue(np.all(np.abs(max_target_speed - speeds_on_straights) < 0.2))

    def test_C1T_07(self):
        velocities, target_velocities = plot_vehicle_speed(self.bag_dir, show_plots=False)
        max_target_speed = np.max(target_velocities)
        # Determine which slowdowns are due to turns and which slowdowns are due to entering/exiting goals
        stop_detected = True  # Vehicle will originally be stopped
        buffer = 0  # Buffer used to count previous velocities that may or may not be due to entering/exiting a goal
        wait_for_vehicle_to_accelerate = 0
        relevant_velocity_idxs = []
        for velocity in target_velocities:
            if velocity == max_target_speed:
                if wait_for_vehicle_to_accelerate > 5:
                    relevant_velocity_idxs += (buffer + 1) * [True]
                else:
                    relevant_velocity_idxs += (buffer + 1) * [False]
                wait_for_vehicle_to_accelerate += 1
                buffer = 0
                stop_detected = False
            elif stop_detected:
                relevant_velocity_idxs.append(False)    
            elif velocity == 0.0:
                relevant_velocity_idxs += (buffer + 1) * [False]
                buffer = 0
                stop_detected = True
                wait_for_vehicle_to_accelerate = 0
            else:
                buffer += 1
        speeds_on_turns = velocities[relevant_velocity_idxs]
        self.assertTrue(np.all(speeds_on_turns > 0.5 * max_target_speed))

    def test_C1T_08(self):
        arrival_distances = check_distance_to_arrival(self.bag_dir)
        self.assertTrue(np.all(arrival_distances < 0.2))

    def test_C1T_09(self):
        self.assertTrue(check_message_published(self.bag_dir, "/outgoing_mobility_operation"))

    def test_C1T_10(self):
        time_between_messages = check_message_timing(self.bag_dir, "/outgoing_mobility_operation", "/incoming_mobility_operation")
        self.assertTrue(np.all(time_between_messages < 1.5))

    def test_C1T_12(self):
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "PICKUP"))
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "DROPOFF"))
        pass

    def test_C1T_13(self):
        self.assertTrue(check_port_drayage_ack(self.bag_dir, "HOLDING_AREA"))
        pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run C1T metric analysis")
    parser.add_argument("bag_in", type=str, help="Directory of bag to load and analyze")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    sys.argv[1] = os.path.normpath(os.path.abspath(argdict["bag_in"]))
    unittest.main(verbosity=2, argv=[sys.argv[0]])
