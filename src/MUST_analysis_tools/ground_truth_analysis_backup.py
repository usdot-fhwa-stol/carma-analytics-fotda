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

# GPS stationary noise accounting - Make speed zero for values below this number
MUST_STATIONARY_NOISE = 0.1 #meter/sec
GPS_STATIONARY_NOISE = 0.1 #meter/sec

# lat_zero = 38.954859
# lon_zero = 38.954859
# TFHRC_M_PER_DEGREE_LAT = 111014.602  # meters per degree of latitude change at TFHRC
# TFHRC_M_PER_DEGREE_LON = 86680.832  # meters per degree of longitude change at TFHRC
# ZERO_PT = np.array([38.954859, -77.149175, 50, 0, 0, 0, 0])
# 47.6283465, -122.1436022  top left
# 47.6283346, -122.1422984  top right
# 47.6275475, -122.1436072  bottom left
# 47.6275401, -122.1423235  bottom right

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
# , pytz.timezone('US/Pacific')


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


# def latlon_series_distance(df1, df2, t_offset=0):
#     df2_offset = df2.copy()
#     df2_offset['time'] = df2['time'].to_numpy() + t_offset
#     min_overlap_time = 2.0
#     time_overlap = min(df1['time'][-1], df2['time'][-1] + t_offset) - max(df1['time'][0], df2['time'][0] + t_offset)
#     if time_overlap <= min_overlap_time:
#         return np.inf
#
#     df2_reindexed = df2.reindex(df1.index).interpolate(method='time').reindex(drop=True)
#     print()
#
#
# def find_time_offset(df1, df2, t_guess):
#     t_guess = np.floor(t_guess / 0.2) * 0.2
#     i_guess = np.argmin(df2['time'].to_numpy() - t_guess)
#     min_index = i_guess
#
#     # check above
#     min_distance = latlon_series_distance(df1, df2, t_guess)
#     for t_offset in np.arange(t_guess, df2['time'][-1], 0.2):
#         test_distance = latlon_series_distance(df1, df2, t_offset)


def find_sub_tracks(time, lat, lon):
    break_time = 5.0 # seconds
    break_dist = 5 # meters
    break_indices = []
    for i in range(1, len(time)):
        if haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i]) > break_dist or time[i] - time[i-1] > break_time:
            break_indices.append(i)
    break_indices = [0] + break_indices + [len(time)]
    return break_indices


def select_sub_track_user_input(must_data, gps_data):

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
    plt.title(f'Vehicle tracks')
    plt.tight_layout()
    plt.show()


def load_metadata(test_name, test_log_fname, gps_folder):
    header = ['Test numbering', 'Test Case ID', 'Run #', 'Corrected?',
              'TimeStamp', 'Description', 'Vehicle Speed', 'Vehicle Lane', 'Date', 'Pass/Fail',
              'Vehicle ID', 'Track Index', 'UDP file name', 'Video file name', 'Novatel file name', 'Notes']
    test_log = pd.read_csv(test_log_fname, names=header, skiprows=4)
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

    # test_name = 'NS R-4'
    # if test_name == 'NS R-4':
    #     GPS_VEHICLE_ID = 21
    #     track_index = 11
    #     must_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\2024_08_28_11_29_00_results_converted_sort_speed.txt'
    #     gps_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\MUST-NS_R-4_2024-08-28-18-31-38_814235.csv'
    # elif test_name == 'NR R-1':
    #     # NR-1 looks like maybe the novatel started at 56:25, but according to the
    #     GPS_VEHICLE_ID = 25
    #     track_index = 1
    #     must_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\2024_08_28_10_56_00_results_converted_sort_speed.txt'
    #     gps_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\MUST-NR_R-1_2024-08-28-17-56-25_702953.csv'
    # elif test_name == 'SR R-2':
    #     # NR-1 looks like maybe the novatel started at 56:25, but according to the
    #     GPS_VEHICLE_ID = 4
    #     track_index = 0
    #     must_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\2024_08_28_12_27_01_results_converted_sort_speed.txt'
    #     gps_filename = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\MUST-SR_R-2_2024-08-28-19-28-15_858949.csv'
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
    must_data['heading'] = 360 - must_data['heading']

    # # This block computes an interpolated velocity column
    # # Calculate distance between consecutive pedestrian points
    # distances = [0] + [haversine_distance(must_data['latitude'].iloc[i],
    #                                       must_data['longitude'].iloc[i],
    #                                       must_data['latitude'].iloc[i + 1],
    #                                       must_data['longitude'].iloc[i + 1])
    #                    for i in range(len(must_data) - 1)]
    # # Calculate time differences between consecutive points
    # time_diffs = [0] + [(must_data['epoch_time'].iloc[i+1] - must_data['epoch_time'].iloc[i])
    #                     for i in range(len(must_data)-1)]
    # must_data['speed_interpolated'] = [dist/time if time != 0 else 0 for dist, time in zip(distances, time_diffs)]
    # must_data['speed_interpolated'] = must_data['speed_interpolated'].round(2)
    # must_data.loc[must_data['speed_interpolated'] < MUST_STATIONARY_NOISE, 'speed_interpolated'] = 0

    gps_header = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'latitude stdev', 'longitude stdev', 'altitude stdev', 'heading error', 'speed error']
    gps_data = pd.read_csv(f'{gps_folder}\\{gps_filename}', names=gps_header, skiprows=1)
    gps_data['epoch_time'] = gps_data['timestamp']

    if 'track_index' not in locals() or track_index is None:
        select_sub_track_user_input(must_data, gps_data)
        exit()

    break_indices = break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(),
                                                    must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())
    valid_data_indices = range(break_indices[track_index], break_indices[track_index + 1])
    must_data = must_data.iloc[valid_data_indices].reset_index(drop=True)

    # gps_data['datetime'] = pd.to_datetime(gps_data['epoch_time'], unit='s')
    # gps_data = gps_data.set_index('datetime')
    # gps_data.index = gps_data.index.to_series().dt.round('5ms')
    # gps_data = gps_data.resample('25ms')
    # gps_data = gps_data.interpolate(method='linear')
    # must_data['datetime'] = pd.to_datetime(must_data['epoch_time'], unit='s')
    # must_data = must_data.set_index('datetime')
    # must_data.index = must_data.index.to_series().dt.round('5ms')
    # must_data = must_data.resample('25ms')
    # must_data = must_data.interpolate(method='linear')
    # must_data = must_data.dropna()
    # must_data = must_data.reindex(gps_data.index).interpolate(method='time')

    # example_point = [47.6278859, -122.1431239]
    # must_distances = [haversine_distance(example_point[0], example_point[1], must_data['latitude'][i], must_data['longitude'][i])
    #              for i in range(len(must_data['latitude']))]
    # must_min_index = np.argmin(must_distances)
    # gps_distances = [haversine_distance(example_point[0], example_point[1], gps_data['latitude'][i], gps_data['longitude'][i])
    #              for i in range(len(gps_data['latitude']))]
    # gps_min_index = np.argmin(gps_distances)
    # must_min_epoch = must_data['epoch_time'][must_min_index]
    # gps_min_epoch = gps_data['epoch_time'][gps_min_index]
    gps_time_offset = -117.5 # must_min_epoch - gps_min_epoch
    print(f'GPS time offset: {gps_time_offset}')
    gps_data['sim time'] = gps_data['epoch_time'] - must_data['epoch_time'][0] + gps_time_offset
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0]

    # Set up the figure and axes
    fig = plt.figure(figsize=(20, 10), dpi=100)
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

    # Add labels and a legend
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()
    # plt.title(f'Vehicle tracks Test {test_name}')
    # plt.show()

    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax2.plot(must_data['sim time'], must_data['speed_interpolated'], label='MUST interpolated speed')
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
    # ax3.set_xlim(40, 60)

    fig.suptitle(f'Test {test_name}')
    fig = plt.gcf()
    # fig.set_size_inches(8, 8)
    plt.legend()
    # plt.savefig(f'{output_folder}/{camera_names[i]}_histograms.png', dpi=100)
    plt.show()
    plt.clf()

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
    # test_names = ['MUST-NR_1', 'MUST-NR_2', 'MUST-NR_3', 'MUST-NR_4',
    #               'MUST-NS_1', 'MUST-NS_4', 'MUST-NS_5',
    #               'MUST-NL_1', 'MUST-NL_2', 'MUST-NL_3',
    #               'MUST-ES_1', 'MUST-ES_2', 'MUST-ES_3',
    #               'MUST-ER_1', 'MUST-ER_2', 'MUST-ER_3',
    #               'MUST-EL_1', 'MUST-EL_2', 'MUST-EL_3',
    #               'MUST-SS_1', 'MUST-SS_2', 'MUST-SS_3',
    #               'MUST-SR_1', 'MUST-SR_2', 'MUST-SR_3',
    #               'MUST-SL_1', 'MUST-SL_2', 'MUST-SL_3',
    #               'MUST-WS_1', 'MUST-WS_2', 'MUST-WS_3',
    #               'MUST-WR_1', 'MUST-WR_2', 'MUST-WR_3',
    #               'MUST-WL_1', 'MUST-WL_2', 'MUST-WL_3']
    test_names = ['MUST-NR_1', 'MUST-NR_2', 'MUST-NR_3', 'MUST-NR_4']
    for test_name in test_names:
        generate_plots(test_name, test_log, novatel_folder, udp_folder)




if __name__ == "__main__":
    main(sys.argv)
