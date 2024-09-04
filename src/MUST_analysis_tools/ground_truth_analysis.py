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
import pyproj

# GPS stationary noise accounting - Make speed zero for values below this number
MUST_STATIONARY_NOISE = 0.1 #meter/sec
GPS_STATIONARY_NOISE = 0.1 #meter/sec

intersection_center = [47.6278859, -122.1431239]
lon_to_x = 6371000 * np.cos(intersection_center[0] * np.pi / 180)
lat_to_y = 6371000
x_to_lon = (1 / 1111111.0) / np.cos(intersection_center[0] * np.pi / 180)
y_to_lat = (1 / 1111111.0)


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


def latlon_angle(lat1, lon1, lat2, lon2):
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


# trims x1 and offsets x2 to align, then computes the distance error
def compute_distance_error_haversine(lat1_orig, lon1_orig, lat2, lon2, offset):
    length = len(lat2)
    lat1 = lat1_orig[offset:offset + length]
    lon1 = lon1_orig[offset:offset + length]

    distance = 0
    for i in range(length):
        distance += haversine_distance(lat1[i], lon1[i], lat2[i], lon2[i]) / length

    return distance


def resample_df(df_original, period_ms):
    columns = ['latitude', 'longitude', 'speed', 'heading']
    start_time = df_original['epoch_time'][0]
    end_time = df_original['epoch_time'][len(df_original)-1]
    start_time_smoothed = period_ms * np.ceil(start_time / period_ms)
    end_time_smoothed = period_ms * np.floor(end_time / period_ms)
    df_resampled = pd.DataFrame()
    df_resampled['epoch_time'] = np.arange(start_time_smoothed, end_time_smoothed, period_ms)
    for column in columns:
        df_resampled[column] = resample_series(df_resampled['epoch_time'].to_numpy(), df_original['epoch_time'].to_numpy(), df_original[column].to_numpy())
    return df_resampled


# from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest_index(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


# Assumes that main_time spans series_time, slightly further on both sides
def resample_series(main_time, series_time, series_data):
    overlap_length = len(main_time)

    series_data_main_times = np.zeros(overlap_length)
    for i in range(len(main_time)):
        t = main_time[i]
        if np.min(np.abs(series_time - t)) < 0.001:
            series_index = np.argmin(np.abs(series_time - t))
            series_data_main_times[i] = series_data[series_index]
            continue
        delta = series_time - t
        delta[delta < 0] = np.inf
        i_before = max(np.argmin(delta) - 1, 0)
        slope = ((series_data[i_before+1] - series_data[i_before]) / (series_time[i_before+1] - series_time[i_before]))
        time_delta_before = series_time[i_before] - t
        series_data_main_times[i] = series_data[i_before] + slope * time_delta_before
    return series_data_main_times


# Assumes that df1 spans df2 before and after the 'optimal' time alignment.
# Assumes x1/y1 and x2/y2
# Steps df2 over each piece of df1 computing the minimum distance error for each time shift
# returns the index of that optimal time shift
def find_optimal_time_shift(lat1, lon1, lat2, lon2):
    overlap_region = len(lat1) - len(lat2)
    dist_errors = np.zeros(overlap_region)
    for i in range(overlap_region):
        dist_errors[i] = compute_distance_error_haversine(lat1, lon1, lat2, lon2, i)
    min_error_index = np.argmin(dist_errors)
    return min_error_index


def find_sub_tracks(time, lat, lon):
    break_time = 5.0 # seconds
    break_dist = 5 # meters
    break_indices = []
    for i in range(1, len(time)):
        if haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i]) > break_dist or time[i] - time[i-1] > break_time:
            break_indices.append(i)
    break_indices = [0] + break_indices + [len(time)]
    return break_indices


def select_sub_track_user_input(must_data, gps_data, test_name, intersection_image_path):

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a Basemap instance
    intersection_image = imread(intersection_image_path)
    # Set the bounds of the map (in lat/lon coordinates)
    lower_left_longitude = -122.143605  # Lower-left corner longitude
    lower_left_latitude = 47.627545  # Lower-left corner latitude
    upper_right_longitude = -122.142310  # Upper-right corner longitude
    upper_right_latitude = 47.628340  # Upper-right corner latitude
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


def generate_plots(test_name, test_log, intersection_image_path, gps_folder, must_folder):

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
    must_data['heading'] = (must_data['heading']) % 360

    gps_header = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'latitude stdev', 'longitude stdev', 'altitude stdev', 'heading error', 'speed error']
    gps_data = pd.read_csv(f'{gps_folder}\\{gps_filename}', names=gps_header, skiprows=1)
    gps_data['epoch_time'] = gps_data['timestamp']

    if 'track_index' not in locals() or track_index is None:
        return
        select_sub_track_user_input(must_data, gps_data, test_name, intersection_image_path)
        return
    # else:
    #     return

    break_indices = break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(),
                                                    must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())
    valid_data_indices = range(break_indices[track_index], break_indices[track_index + 1])
    must_data = must_data.iloc[valid_data_indices[1:]].reset_index(drop=True)

    # gps_data['datetime'] = pd.to_datetime(gps_data['epoch_time'], unit='s')
    # gps_data = gps_data.set_index('datetime')
    # gps_data.index = gps_data.index.to_series().dt.round('1ms')
    # must_data['datetime'] = pd.to_datetime(must_data['epoch_time'], unit='s')
    # must_data = must_data.set_index('datetime')
    # must_data.index = must_data.index.to_series().dt.round('1ms')
    # must_data = must_data.reindex(gps_data.index).interpolate(method='time')
    gps_data_resampled = resample_df(gps_data, 0.02)
    must_data_resampled = resample_df(must_data, 0.02)

    # example_point = [47.6278859, -122.1431239]
    # must_distances = [haversine_distance(example_point[0], example_point[1], must_data['latitude'][i], must_data['longitude'][i])
    #              for i in range(len(must_data['latitude']))]
    # must_min_index = np.argmin(must_distances)
    # gps_distances = [haversine_distance(example_point[0], example_point[1], gps_data['latitude'][i], gps_data['longitude'][i])
    #              for i in range(len(gps_data['latitude']))]
    # gps_min_index = np.argmin(gps_distances)
    # must_min_epoch = must_data['epoch_time'][must_min_index]
    # gps_min_epoch = gps_data['epoch_time'][gps_min_index]
    # gps_time_offset = must_min_epoch - gps_min_epoch
    # gps_time_offset = -117.5
    time_offset_index = find_optimal_time_shift(gps_data_resampled['latitude'].to_numpy(), gps_data_resampled['longitude'].to_numpy(),
                                                must_data_resampled['latitude'].to_numpy(), must_data_resampled['longitude'].to_numpy())
    time_offset = gps_data_resampled['epoch_time'][time_offset_index] - gps_data_resampled['epoch_time'][0]
    print(f'time offset for test {test_name}: {time_offset}, static: {-gps_data['epoch_time'][0] + must_data['epoch_time'][0] + 117.5}')
    gps_data['sim time'] = gps_data['epoch_time'] - gps_data['epoch_time'][0]#  + (gps_data['epoch_time'][0] - must_data['epoch_time'][0] - 117.5)
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0] + time_offset
    # Set up the figure and axes
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax_map = fig.add_subplot(gs[:, :2])

    # This block computes an interpolated velocity column
    # Calculate distance between consecutive pedestrian points
    distances = [0] + [haversine_distance(must_data['latitude'].iloc[i],
                                          must_data['longitude'].iloc[i],
                                          must_data['latitude'].iloc[i + 1],
                                          must_data['longitude'].iloc[i + 1])
                       for i in range(len(must_data) - 1)]
    # Calculate time differences between consecutive points
    time_diffs = [0] + [(must_data['epoch_time'].iloc[i+1] - must_data['epoch_time'].iloc[i])
                        for i in range(len(must_data)-1)]
    must_data['speed_interpolated'] = [dist/time if time != 0 else 0 for dist, time in zip(distances, time_diffs)]
    must_data['speed_interpolated'] = must_data['speed_interpolated'].round(2)
    must_data.loc[must_data['speed_interpolated'] < MUST_STATIONARY_NOISE, 'speed_interpolated'] = 0
    N = 1
    must_data['speed_interpolated'] = np.convolve(must_data['speed_interpolated'].to_numpy(), np.ones(N) / N, mode='same')

    must_data['heading_interpolated'] = [0] + [latlon_angle(must_data['latitude'].iloc[i],
                                                    must_data['longitude'].iloc[i],
                                                    must_data['latitude'].iloc[i + 1],
                                                    must_data['longitude'].iloc[i + 1])
                                for i in range(len(must_data) - 1)]
    must_data['heading_interpolated'] = (np.convolve(must_data['heading_interpolated'].to_numpy(), np.ones(N) / N, mode='same') % 360)

    shadow_data = gps_data.copy()
    vehicle_height = 0.5
    camera_height = 5
    location = [47.6277065, -122.1432879]
    # shadow_data['camera_dist'] = [haversine_distance(gps_data['latitude'].iloc[i],
    #                                                 gps_data['longitude'].iloc[i],
    #                                                 location[0],
    #                                                 location[1])
    #                             for i in range(len(gps_data))]
    # shadow_data['angles'] = [latlon_angle(gps_data['latitude'].iloc[i],
    #                                                 gps_data['longitude'].iloc[i],
    #                                                 location[0],
    #                                                 location[1])
    #                             for i in range(len(gps_data))]
    # # shadow_add = distance * (camera_height / (camera_height - vehicle_height) - 1)
    # # shadow_add = distance / 4
    # # shadow_lat = old_lat + shadow_add * sin(angle)
    # shadow_data['lat_shadow'] = [shadow_data['latitude'].iloc[i] +
    #                              (shadow_data['camera_dist'].iloc[i] / 4.0 * np.sin(shadow_data['angles'].iloc[i]*np.pi/180))
    #                             for i in range(len(gps_data))]
    # shadow_data['lon_shadow'] = [shadow_data['longitude'].iloc[i] +
    #                              (shadow_data['camera_dist'].iloc[i] / 4.0 * np.cos(shadow_data['angles'].iloc[i]*np.pi/180))
    #                             for i in range(len(gps_data))]
    shadow_data['lat_shadow'] = [shadow_data['latitude'].iloc[i] + (shadow_data['latitude'].iloc[i] - location[0]) *
                                 (camera_height / (camera_height - vehicle_height) - 1)
                                for i in range(len(gps_data))]
    shadow_data['lon_shadow'] = [shadow_data['longitude'].iloc[i] + (shadow_data['longitude'].iloc[i] - location[1]) *
                                 (camera_height / (camera_height - vehicle_height) - 1)
                                for i in range(len(gps_data))]

    # Create a Basemap instance
    intersection_image = imread(intersection_image_path)
    # Set the bounds of the map (in lat/lon coordinates)
    lower_left_longitude = -122.143605  # Lower-left corner longitude
    lower_left_latitude = 47.627545  # Lower-left corner latitude
    upper_right_longitude = -122.142310  # Upper-right corner longitude
    upper_right_latitude = 47.628340  # Upper-right corner latitude
    map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax_map)

    # Show the image as the background
    map.imshow(intersection_image, origin='upper')

    # Convert lat/lon to map projection coordinates
    must_x, must_y = map(must_data['longitude'], must_data['latitude'])
    gps_x, gps_y = map(gps_data['longitude'], gps_data['latitude'])
    gps_x_e1, gps_y_e1 = map(gps_data['longitude'] + 3*x_to_lon, gps_data['latitude'] + 3*y_to_lat)
    gps_x_e2, gps_y_e2 = map(gps_data['longitude'] - 3*x_to_lon, gps_data['latitude'] - 3*y_to_lat)
    shadow_x, shadow_y = map(shadow_data['lon_shadow'], shadow_data['lat_shadow'])

    # Plot the data points
    map.plot(must_x, must_y, markersize=5, label=f'MUST track')
    map.plot(gps_x, gps_y, markersize=5, label=f'GPS track')
    map.plot(gps_x_e1, gps_y_e1, markersize=5, label=f'GPS error bars')
    map.plot(gps_x_e2, gps_y_e2, markersize=5, label=f'GPS error bars')
    # map.plot(shadow_x, shadow_y, markersize=5, label=f'GPS shadow track')
    ax_map.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax2.plot(must_data['sim time'], must_data['speed'], label='MUST speed')
    ax2.plot(gps_data['sim time'], gps_data['speed'], label='GPS speed')
    ax2.plot(must_data['sim time'], must_data['speed_interpolated'], label='MUST speed (interpolated)')
    ax2.legend()
    ax3.plot(must_data['sim time'], must_data['heading'], label='MUST heading')
    ax3.plot(gps_data['sim time'], gps_data['heading'], label='GPS heading')
    ax3.plot(must_data['sim time'], must_data['heading_interpolated'], label='MUST heading (interpolated)')

    ax2.set_title('speed vs. time')
    ax2.set_xlabel('time (seconds)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_xlim(must_data['sim time'][0], must_data['sim time'][len(must_data) - 1])
    ax3.set_title('heading vs. time')
    ax3.set_xlabel('time (seconds)')
    ax3.set_ylabel('heading (degrees)')
    ax3.set_ylim(0, 360)
    ax3.set_xlim(must_data['sim time'][0], must_data['sim time'][len(must_data) - 1])

    fig.suptitle(f'Test {test_name}')
    fig = plt.gcf()
    plt.legend()
    output_folder = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\Analysis'
    # plt.savefig(f'{output_folder}\\{test_name}_plots.png', dpi=100)
    plt.show()
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
    intersection_image = 'C:\\Users\\annika\\OneDrive\\Documents\\freight_cp\\must_sensor_intersection_1.png'
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
    # test_names = ['MUST-NR_2', 'MUST-NS_5']
    for test_name in test_names:
        generate_plots(test_name, test_log, intersection_image, novatel_folder, udp_folder)


if __name__ == "__main__":
    main(sys.argv)
