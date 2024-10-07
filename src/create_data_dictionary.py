#!/usr/bin/env python3

# This python script automates the process for creating data dictionaries given a ROS1 bag file or a directory containing multiple ROS1 bag files
# By default, the script will create entries for all topics present in all ROS1 bag files
# If desired, a csv detailing which topics should be in the data dictionary can also be provided
# The script will output an Excel file containing the topic name and message spec for each topic, as well as the name and type of each data element defined in the message
# A description for each data element is the only component that will need to be entered manually

import sys
import rosbag
from roslib.message import get_message_class
import argparse, argcomplete
import os
from tqdm import tqdm
import pandas as pd

class DataDictionaryCreator:
    def __init__(self, bag_path, topics_path):
        # Get relevant topics
        if topics_path != "":
            self.topics_list = set(pd.read_csv(topics_path).values.flatten())
        else:
            self.topics_list = set()
        self.topics_processed = set()
        self.dictionary = {}
        self.dictionary["Topic Name"] = []
        self.dictionary["Message Spec"] = []
        self.dictionary["Data Element Name"] = []
        self.dictionary["Data Element Type"] = []
        if os.path.isdir(bag_path):
            bags_list = self.get_bags_recursive(bag_path)
        else:
            bags_list = [bag_path]
        for bag_file in tqdm(bags_list, total=len(bags_list), desc="Bags Processed"):
            bag = rosbag.Bag(bag_file, 'r')
            self.populate_dictionary(bag)
        self.save_dictionary()

    def populate_dictionary(self, bag):
        # Loop through messages in bag
        
        total_messages = bag.get_message_count()
        for topic, msg, _ in tqdm(bag, total=total_messages, desc="Messages Processed", leave=False):
            if (not len(self.topics_list) or topic in self.topics_list) and topic not in self.topics_processed:
                self.topics_processed.add(topic)
                msg_spec = msg._type
                fields = []
                for field_name in msg.__slots__:
                    # Field name and type are in the _slot_types and __slots__ attributes
                    field_type = msg._slot_types[msg.__slots__.index(field_name)]
                    fields.append((field_name, field_type))
                self.dictionary["Topic Name"].append(topic)
                self.dictionary["Message Spec"].append(msg_spec)
                if len(fields):
                    self.dictionary["Data Element Name"].append(fields[0][0])
                    self.dictionary["Data Element Type"].append(fields[0][1])
                    for i in range(1, len(fields)):
                        self.dictionary["Topic Name"].append("")
                        self.dictionary["Message Spec"].append("")
                        self.dictionary["Data Element Name"].append(fields[i][0])
                        self.dictionary["Data Element Type"].append(fields[i][1])
                else:
                    self.dictionary["Data Element Name"].append("")
                    self.dictionary["Data Element Type"].append("")

    def save_dictionary(self):
        pd.DataFrame(self.dictionary).to_excel("dictionary.xlsx", index=False)


    def get_bags_recursive(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".bag"):
                    files.append(os.path.join(root, filename))
        return files

def main(argdict):
    
    DataDictionaryCreator(os.path.normpath(os.path.abspath(argdict["bag_in"])), argdict["topics_list"])

    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data for model identification")
    parser.add_argument("bag_in", type=str, help="Bag to load, or directory of bags")
    parser.add_argument("--topics_list", type=str, default = "", help="Path to csv file containing desired topics")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    main(argdict)
