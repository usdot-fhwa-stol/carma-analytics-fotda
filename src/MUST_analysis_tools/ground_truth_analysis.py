import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import copy
from datetime import datetime, timedelta
import math
from mpl_toolkits.basemap import Basemap
from matplotlib.image import imread
import os

# Intersection image
intersection_image = imread('C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\must_sensor_intersection_1.png')
# Set the bounds of the map (in lat/lon coordinates)
lower_left_longitude = -122.143605  # Lower-left corner longitude
lower_left_latitude = 47.627545     # Lower-left corner latitude
upper_right_longitude = -122.142310 # Upper-right corner longitude
upper_right_latitude = 47.628340    # Upper-right corner latitude


def datetime_to_unix_ts(datetime_in):
    seconds = math.floor(datetime_in.timestamp()) + 3600*3
    nanoseconds = (datetime_in.timestamp() % 1)
    return seconds + nanoseconds


def str_to_datetime_unformat(datetime_str, timezone=None):
    if timezone is not None:
        return datetime.strptime(datetime_str, "%Y_%m_%d_%H_%M_%S.%f").replace(tzinfo=timezone)
    return datetime.strptime(datetime_str, "%Y_%m_%d_%H_%M_%S.%f")


def str_to_unix_ts_pst(datetime_str):
    return datetime_to_unix_ts(str_to_datetime_unformat(datetime_str))


# Haversine function to calculate distance between two lat-long points
def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c * 1000  # Convert to meters
    return abs(distance)


def find_sub_tracks(time, lat, lon):
    break_time = 5.0 # seconds
    break_dist = 5 # meters
    break_indices = []
    for i in range(1, len(time)):
        if haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i]) > break_dist or time[i] - time[i-1] > break_time:
            break_indices.append(i)
    break_indices = [0] + break_indices + [len(time)]
    return break_indices


def select_sub_track_user_input(must_data, gps_data, test_name):

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a Basemap instance
    map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax)

    # Show the image as the background
    map.imshow(intersection_image, origin='upper')
    gps_x, gps_y = map(gps_data['longitude'], gps_data['latitude'])
    map.plot(gps_x, gps_y, marker='o', markersize=5, label=f'GPS track')

    break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(), must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())

    for i in range(len(break_indices) - 1):
        sub_track_indices = range(break_indices[i], break_indices[i + 1])
        # Convert lat/lon to map projection coordinates
        must_x, must_y = map(must_data['longitude'].iloc[sub_track_indices], must_data['latitude'].iloc[sub_track_indices])

        # Plot the data points
        map.plot(must_x, must_y, marker='o', markersize=5, label=f'Track {i}')

    # Add labels and a legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title(f'Vehicle tracks {test_name}')
    plt.tight_layout()
    plt.show()


def load_metadata(test_name, test_log_fname, gps_folder):
    header = ['Test numbering', 'Test Case ID', 'Run #', 'Corrected?', 'Original',
              'TimeStamp', 'Description', 'Vehicle Speed', 'Vehicle Lane', 'Date', 'Pass/Fail',
              'Vehicle ID', 'Track Index', 'UDP file name', 'Video file name', 'Novatel file name', 'Notes']
    test_log = pd.read_csv(test_log_fname, names=header, skiprows=2)
    test_log = test_log[test_log['Test Case ID'] == test_name.split('_')[0]].reset_index(drop=True)
    test_log['Run #'] = pd.to_numeric(test_log['Run #'], downcast='integer')
    test_log = test_log[test_log['Run #'] == int(test_name.split('_')[1])].reset_index(drop=True)
    if len(test_log) != 1:
        print(f'Unable to get test {test_name}, {len(test_log)} matches found')
        print(test_log.head(5))
        exit()

    GPS_VEHICLE_ID = int(test_log['Vehicle ID'][0])
    track_index = test_log['Track Index'][0]
    if math.isnan(track_index):
        track_index = None
    else:
        track_index = int(track_index)
    must_filename = test_log['UDP file name'][0]
    filenames = os.listdir(gps_folder)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(gps_folder, f))]
    for filename in filenames:
        check_str = f'{test_name.split('_')[0]}_R-{test_name.split('_')[1]}'
        if filename[:len(check_str)] == check_str:
            gps_filename = filename
    if 'gps_filename' not in locals():
        print(f'Unable to get vehicle file for test {test_name}')
        exit()
    return GPS_VEHICLE_ID, track_index, must_filename, gps_filename


def generate_plots(test_name, test_log, gps_folder, must_folder):

    GPS_VEHICLE_ID, track_index, must_filename, gps_filename = load_metadata(test_name, test_log, gps_folder)

    must_header = ['server time', 'frame id', 'class id', 'vehicle id', 'image_x', 'image_y', 'image_width', 'image_height', 'latitude', 'longitude', 'speed', 'heading']
    must_data = pd.read_csv(f'{must_folder}\\{must_filename}', sep='\\s+', names=must_header)
    # Convert to unix timestamp (epoch time) in UTC
    must_data['epoch_time'] = must_data['server time'].apply(str_to_unix_ts_pst)
    must_data.sort_values('epoch_time')
    must_data = must_data[must_data['vehicle id'] == GPS_VEHICLE_ID].reset_index(drop=True)
    must_data = must_data.drop(['server time'], axis=1)
    # Speed is in mph -> m/s
    must_data['speed'] = must_data['speed'] / 2.23694
    # Heading is North, positive West -> North, positive East
    must_data['heading'] = (-must_data['heading']) % 360

    gps_header = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'latitude stdev', 'longitude stdev', 'altitude stdev', 'heading error', 'speed error']
    gps_data = pd.read_csv(f'{gps_folder}\\{gps_filename}', names=gps_header, skiprows=1)
    gps_data['epoch_time'] = gps_data['timestamp']

    if 'track_index' not in locals() or track_index is None:
        return
        select_sub_track_user_input(must_data, gps_data, test_name)
        return
    # else:
    #     return

    break_indices = break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(),
                                                    must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())
    valid_data_indices = range(break_indices[track_index], break_indices[track_index + 1])
    must_data = must_data.iloc[valid_data_indices].reset_index(drop=True)

    gps_time_offset = -117.5
    gps_data['sim time'] = gps_data['epoch_time'] - must_data['epoch_time'][0] + gps_time_offset
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0]
    # Set up the figure and axes
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax_map = fig.add_subplot(gs[:, :2])

    # Create a Basemap instance
    map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax_map)

    # Show the image as the background
    map.imshow(intersection_image, origin='upper')

    # Convert lat/lon to map projection coordinates
    must_x, must_y = map(must_data['longitude'], must_data['latitude'])
    gps_x, gps_y = map(gps_data['longitude'], gps_data['latitude'])

    # Plot the data points
    map.plot(must_x, must_y, marker='o', markersize=5, label=f'MUST track')
    map.plot(gps_x, gps_y, marker='o', markersize=5, label=f'GPS track')

    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax2.plot(must_data['sim time'], must_data['speed'], label='MUST speed')
    ax2.plot(gps_data['sim time'], gps_data['speed'], label='GPS speed')
    ax2.legend()
    ax3.plot(must_data['sim time'], must_data['heading'], label='MUST heading')
    ax3.plot(gps_data['sim time'], gps_data['heading'], label='GPS heading')

    ax2.set_title('speed vs. time')
    ax2.set_xlabel('time (seconds)')
    ax2.set_ylabel('speed (m/s)')
    # ax2.set_xlim(40, 60)
    ax3.set_title('heading vs. time')
    ax3.set_xlabel('time (seconds)')
    ax3.set_ylabel('heading (degrees)')
    ax3.set_ylim(0, 360)
    # ax3.set_xlim(40, 60)

    fig.suptitle(f'Test {test_name}')
    fig = plt.gcf()
    plt.legend()
    output_folder = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\Analysis'
    plt.savefig(f'{output_folder}\\{test_name}_plots.png', dpi=100)
    # plt.show()
    # plt.clf()

    # Metric 1: position accuracy (90% <30cm error)
    # find matching start/end times
    #   create same-ts interpolated track at a fixed frequency/start time?
    #   find any points super close in time?
    # compute haversine distance
    # compute mean/stdev of error and print

    # Metric 2: speed accuracy (90% <3mph error)
    # find matching start/end times
    #   create same-ts interpolated track at a fixed frequency/start time?
    #   find any points super close in time?
    # compute mean/stdev of error and print


def main(args):
    test_log = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\CARMA-Freight-MUST Test plan log sheet.xlsx - Test Log.csv'
    novatel_folder = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\Novatel Data'
    udp_folder = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\MUST UDP Data'
    test_names = ['MUST-NR_1', 'MUST-NR_2', 'MUST-NR_3', 'MUST-NR_4',
                  'MUST-NS_1', 'MUST-NS_4', 'MUST-NS_5',
                  'MUST-NL_2', 'MUST-NL_3', # 'MUST-NL_1',
                  # 'MUST-ES_1', 'MUST-ES_2', 'MUST-ES_3',
                  'MUST-ER_1', 'MUST-ER_2', # 'MUST-ER_3',
                  'MUST-EL_2', 'MUST-EL_3', #  'MUST-EL_1',
                  'MUST-SS_1', 'MUST-SS_2', # 'MUST-SS_3',
                  'MUST-SR_1', 'MUST-SR_2', 'MUST-SR_3',
                  'MUST-SL_2', 'MUST-SL_3', # 'MUST-SL_1',
                  # 'MUST-WS_1', 'MUST-WS_2', # 'MUST-WS_3',
                  'MUST-WR_1', 'MUST-WR_3', # 'MUST-WR_2',
                  'MUST-WL_3'] # 'MUST-WL_2', # 'MUST-WL_1',
    # test_names = ['MUST-NR_1', 'MUST-NR_2', 'MUST-NR_3', 'MUST-NR_4']
    for test_name in test_names:
        generate_plots(test_name, test_log, novatel_folder, udp_folder)


if __name__ == "__main__":
    main(sys.argv)
