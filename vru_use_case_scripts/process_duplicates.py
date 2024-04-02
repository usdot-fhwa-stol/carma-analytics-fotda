import rosbag
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from enum import Enum


class ObjectType(Enum):
    UNKNOWN = 0
    SMALL_VEHICLE = 1
    LARGE_VEHICLE = 2
    MOTORCYCLE = 3
    PEDESTRIAN = 4
    def __str__(self):
        return self.name


def plot_duplicates(input_file, plots_dir):
    # plot duplicate durations of each Object over time
    df = pd.read_csv(input_file)
    grouped = df.groupby('ObjID')

    # Plot duration over Message Time for each ObjID
    for obj_id, group in grouped:
        obj_type = group['ObjType'].iloc[0]  # Get the ObjType for the current ObjIDMain
        fig, ax = plt.subplots()
        ax.stem(group['Message Time (ms)'], group['Duration'], basefmt=" ", markerfmt=' ', use_line_collection=True)
        ax.set_xlim(left=0)  # Set x-axis to start at zero
        ax.set_ylim(bottom=0, top=1100)  # Set y-axis to start at zero
        plt.axhline(y=1000, color='r', linestyle='--', label='Cutoff')
        plt.xlabel('Message Time (ms)')
        plt.ylabel('Duration')
        plt.title(f'Duration over Message Time for ObjID: {obj_id} with type: {obj_type}')
        plt.legend()
        plt.savefig(f"{plots_dir}-{obj_id}.png")

def get_duplicate_duration(input_file, output_file):
    # Calculate the duration each duplicate lasts
    df = pd.read_csv(input_file)
    df.sort_values(by=['ObjID', 'Message Time (ms)'], inplace=True)
    df['TimeDiff'] = df.groupby('ObjID')['Message Time (ms)'].diff()
    # Create a mask to identify interruptions in the ObjIDMain sequence
    mask = (df['TimeDiff'] <= 100)
    df['Mask'] = mask
    sum_timediff = 0
    # calculate the duration each duplicate lasts
    for index, row in df.iterrows():
        if row['Mask']:
            sum_timediff += row['TimeDiff']
        else:
            sum_timediff = 0
        df.at[index, 'Duration'] = sum_timediff

    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)


def remove_main_obj(input_file, output_file):
    # The main detections are removed from the detected objects by removing the first instance of each object ID
    df = pd.read_csv(input_file)
    # Group by 'Message Time (ms)' and 'ObjID' drop the first row for each group
    df = df.groupby(['Message Time (ms)', 'ObjID']).apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    df = df.drop_duplicates() # this line removes the exact duplicates (
    # Write the output to a new CSV file
    df.to_csv(output_file, index=False)


def get_detected_objects_id_and_type(ros_bag_file, output_file):
    # Extracting all detected objects from ROS bag
    detected_obj_topic_name = '/environment/fused_external_objects'
    messages = []
    with rosbag.Bag(ros_bag_file, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=[detected_obj_topic_name]):
            messages.extend(msg.objects)

    if output_file.exists():
        print(f'Output file {output_file} already exists. Overwriting file.')
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Message Time (ms)', 'ObjID', 'ObjType'])
        for message in messages:
            msg_timestamp_ms = math.floor (message.header.stamp.to_sec() * 1000)
            writer.writerow([msg_timestamp_ms, message.id%1000, ObjectType(message.object_type)])

    print(f"Sorted data from topic '{detected_obj_topic_name}' saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Script to parse ROS Bags into CSV data')
    parser.add_argument('--ros-bag-file', help='ROS Bags File.', type=str, required=True)
    parser.add_argument('--csv-dir', help='Directory to write csv file to.', type=Path, required=True)
    parser.add_argument('--plots-dir', type=Path, default=Path("plots"))
    args = parser.parse_args()

    #args.plots_dir.mkdir(exist_ok=True)

    get_detected_objects_id_and_type(args.ros_bag_file, args.csv_dir/'all_detected_objects.csv')
    remove_main_obj(args.csv_dir/'all_detected_objects.csv', args.csv_dir/'duplicates.csv')
    get_duplicate_duration(args.csv_dir/'duplicates.csv', args.csv_dir/'duplicates_durations.csv')
    plot_duplicates(args.csv_dir/'duplicates_durations.csv', args.plots_dir)


if __name__ == '__main__':
    main()
