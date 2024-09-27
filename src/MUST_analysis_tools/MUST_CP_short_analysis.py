import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import math
from mpl_toolkits.basemap import Basemap
from matplotlib.image import imread
import os
import pyproj
from pathlib import Path
from enum import Enum
import csv


class Object_class(Enum):
    person = 0
    bicycle = 1
    car = 2
    motorcycle = 3
    bus = 4
    truck = 7
    traffic_light = 9
    other = 10

    @classmethod
    def _missing_(cls, value):
        # If the input value is a string that matches a name, return the corresponding enum
        if isinstance(value, str) and value in cls.__members__:
            return cls.__members__[value]
        # If the value doesn't match any member, return `Object_class.other`
        return cls.other

# GPS stationary noise accounting - Make speed zero for values below this number
MUST_STATIONARY_NOISE = 0.1 #meter/sec
GPS_STATIONARY_NOISE = 0.1 #meter/sec

zero_pos = np.array([-122.143246, 47.627705])
MUST_sensor_loc = [-122.143246, 47.627705]
intersection_center = [-122.1431239, 47.6278859]
lon_to_x = 111111.0 * np.cos(intersection_center[1] * np.pi / 180)
lat_to_y = 111111.0
x_to_lon = (1 / 111111.0) / np.cos(intersection_center[1] * np.pi / 180)
y_to_lat = (1 / 111111.0)
# Set the bounds of the map (in lat/lon coordinates)
lower_left_longitude = -122.1436116  # Lower-left corner longitude
lower_left_latitude = 47.6275314  # Lower-left corner latitude
upper_right_longitude = -122.1423166  # Upper-right corner longitude
upper_right_latitude = 47.6283264  # Upper-right corner latitude


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


# Local angle from lat/lon coords
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


# Resample pandas dataframe to specified period
# Creates a standard timeseries slightly inside of the min/max timestamps of the input data, then interpolates the data to match those timestamps
def resample_df(df_original, period_ms):
    columns = ['latitude', 'longitude', 'speed']
    start_time = df_original['epoch_time'][0]
    end_time = df_original['epoch_time'][len(df_original)-1]
    start_time_smoothed = period_ms * np.ceil(start_time / period_ms)
    end_time_smoothed = period_ms * np.floor(end_time / period_ms)
    df_resampled = pd.DataFrame()
    df_resampled['epoch_time'] = np.arange(start_time_smoothed, end_time_smoothed, period_ms)
    for column in columns:
        df_resampled[column] = resample_series(df_resampled['epoch_time'].to_numpy(), df_original['epoch_time'].to_numpy(), df_original[column].to_numpy())
    heading_unwrapped = np.unwrap(df_original['heading'].to_numpy(), period=360, discont=270)
    df_resampled['heading'] = resample_series(df_resampled['epoch_time'].to_numpy(), df_original['epoch_time'].to_numpy(), heading_unwrapped)
    df_resampled['heading'] = df_resampled['heading'] % 360
    return df_resampled


# Interpolate the data in a given series to match the given target timestamps
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
    if np.isinf(np.min(dist_errors)):
        print(f'Optimal timeshift not found')
        exit()
    min_error_index = np.argmin(dist_errors)
    return min_error_index


# Finds sequential points that differ by more than the break_time or break_distance
# Return a list of those break points, [[0], [break_1_index], ..., [-1]]
def find_sub_tracks(time, lat, lon):
    break_time = 5.0 # seconds
    break_dist = 5 # meters
    break_indices = []
    for i in range(1, len(time)):
        if haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i]) > break_dist or time[i] - time[i-1] > break_time:
            break_indices.append(i)
    break_indices = [0] + break_indices + [len(time)]
    return break_indices


# Show subtracks to the user to get them to update the test log with an index, in the rare case that our vehicle is not the first subtrack
def select_sub_track_user_input(must_data, gps_data, test_name, intersection_image_path, output_folder):

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a Basemap instance
    intersection_image = imread(intersection_image_path)
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
    # plt.savefig(os.path.join(output_folder, f'{test_name}_tracks_by_vehicle_id.png'), dpi=100)
    plt.show()
    plt.clf()


# Loads filenames/similar from the test log
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
    gprmc_filename = None
    for filename in filenames:
        gps_check_str = f'{test_name.split("_")[0]}_{test_name.split("_")[1]}_2'
        gprmc_check_str = f'{test_name.split("_")[0]}_{test_name.split("_")[1]}_gprmc'
        if filename[:len(gps_check_str)] == gps_check_str:
            gps_filename = filename
        if filename[:len(gprmc_check_str)] == gprmc_check_str:
            gprmc_filename = filename
    if 'gps_filename' not in locals():
        print(f'Unable to get vehicle file for test {test_name}')
        exit()
    return GPS_VEHICLE_ID, track_index, must_filename, gps_filename, gprmc_filename


# Convert MUST sensor output local x/y in meters to lat/lon
def lat_lon_from_x_y_must(x, y):
    longitude = MUST_sensor_loc[0] + x * x_to_lon
    latitude = MUST_sensor_loc[1] + y * y_to_lat
    return latitude, longitude


# Computing UTC time from the GPRMC message
def recover_timestamp_from_gps_time(date_str_arr, seconds_in_day_arr, message_timestamps):
    # Create a combined datetime array
    datetime_array = pd.to_datetime(date_str_arr) + pd.to_timedelta(seconds_in_day_arr, unit='s')

    # Convert to UTC
    datetime_utc = datetime_array.tz_localize('UTC')

    # Convert to Pacific Time
    datetime_pacific = datetime_utc.tz_convert('America/Los_Angeles')
    timestamps_pacific = np.array(datetime_pacific.astype('int64') / 1e9)

    return timestamps_pacific


# Do the analysis and plot the results
def generate_plots(test_name, test_log, intersection_image_path, gps_folder, must_folder, output_folder):

    ## Load data from test log
    GPS_VEHICLE_ID, track_index, must_filename, gps_filename, gprmc_filename = load_metadata(test_name, test_log, gps_folder)
    plot_results = True

    ## Load MUST sensor data
    must_header = ['class str', 'x', 'y', 'heading', 'speed', 'size', 'confidence', 'vehicle id', 'epoch_time']
    must_data = pd.read_csv(str(os.path.join(must_folder, must_filename)), names=must_header)
    must_data['class id'] = must_data['class str'].apply(lambda x: Object_class(x).value)
    must_data['vehicle id'] = must_data['vehicle id'].astype(np.int32)
    must_data = must_data[must_data['vehicle id'] == GPS_VEHICLE_ID].reset_index(drop=True)
    # Remove first line because the heading and speed are always zero
    must_data = must_data.tail(-1)
    # Get lat/lon from local x/y coordinates
    must_data['latitude'], must_data['longitude'] = lat_lon_from_x_y_must(must_data['x'].to_numpy(), must_data['y'].to_numpy())
    must_data.sort_values('epoch_time')
    # must_data['heading'] = (must_data['heading'] + 180) % 360

    ## Load GPS and GPRMC data
    gps_header = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'latitude stdev', 'longitude stdev', 'altitude stdev', 'heading error', 'speed error']
    gps_data = pd.read_csv(str(os.path.join(gps_folder, gps_filename)), names=gps_header, skiprows=1)
    gps_data['epoch_time'] = gps_data['timestamp']
    # gps_data['heading'] = (gps_data['heading'] + 180) % 360
    gprmc_header = ['message_timestamp', 'date', 'seconds_in_day', 'latitude', 'longitude', 'heading', 'speed']
    gprmc_data = pd.read_csv(str(os.path.join(gps_folder, gprmc_filename)), names=gprmc_header, skiprows=1)
    gprmc_data['epoch_time'] = recover_timestamp_from_gps_time(gprmc_data['date'].to_numpy(), gprmc_data['seconds_in_day'].to_numpy(), gprmc_data['message_timestamp'].to_numpy())
    gprmc_data['longitude'] = -gprmc_data['longitude']
    # gprmc_data['heading'] = (gprmc_data['heading'] + 180) % 360

    ## If track index is undefined in the test log, show the user so they can enter the track ID
    if 'track_index' not in locals() or track_index is None:
        select_sub_track_user_input(must_data, gps_data, test_name, intersection_image_path, output_folder)
        return
    ## If there is a track ID defined, grab all data in that track
    break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(),
                                                    must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())
    valid_data_indices = range(break_indices[track_index], break_indices[track_index + 1])
    must_data = must_data.iloc[valid_data_indices[1:]].reset_index(drop=True)

    ## Resample data to 50hz, for time synchronization and metric comparisons
    gps_data_resampled = resample_df(gps_data, 0.02)
    must_data_resampled = resample_df(must_data, 0.02)
    gprmc_data_resampled = resample_df(gprmc_data, 0.02)

    ## Compute and apply the optimal time shift
    time_offset_index = find_optimal_time_shift(gprmc_data_resampled['latitude'].to_numpy(), gprmc_data_resampled['longitude'].to_numpy(),
                                                must_data_resampled['latitude'].to_numpy(), must_data_resampled['longitude'].to_numpy())
    time_offset = gprmc_data_resampled['epoch_time'][time_offset_index] - gprmc_data_resampled['epoch_time'][0]
    gps_data_matched = gps_data_resampled
    must_data_matched = must_data_resampled
    must_data_matched['epoch_time'] = must_data_matched['epoch_time'] + time_offset
    gps_data_matched = gps_data_matched[time_offset_index-1:time_offset_index-1 + len(must_data_matched)].reset_index(drop=True)

    gps_data['sim time'] = gps_data['epoch_time'] - gps_data['epoch_time'][0]#  + (gps_data['epoch_time'][0] - must_data['epoch_time'][0] - 117.5)
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0] + time_offset
    gprmc_data['sim time'] = gprmc_data['epoch_time'] - gprmc_data['epoch_time'][0]

    ## Create and save the lat/lon, heading, and speed plots
    if plot_results:
        # Set up the figure and axes
        fig = plt.figure(figsize=(20, 12), dpi=100)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        ax_map = fig.add_subplot(gs[:, :2])

        # Create a Basemap instance to plot lat/lon on an image
        intersection_image = imread(intersection_image_path)
        map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                    llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax_map)

        # Show the image as the background
        map.imshow(intersection_image, origin='upper')

        # Convert lat/lon to map projection coordinates
        must_x, must_y = map(must_data['longitude'], must_data['latitude'])
        gps_x, gps_y = map(gps_data['longitude'], gps_data['latitude'])
        gprmc_x, gprmc_y = map(gprmc_data['longitude'], gprmc_data['latitude'])
        dist = 0.3
        x_above = dist*x_to_lon*np.cos((-gps_data['heading'].to_numpy()) * np.pi / 180)
        y_above = dist*y_to_lat*np.sin((-gps_data['heading'].to_numpy()) * np.pi / 180)
        x_below = dist*x_to_lon*np.cos((-gps_data['heading'].to_numpy() - 180) * np.pi / 180)
        y_below = dist*y_to_lat*np.sin((-gps_data['heading'].to_numpy() - 180) * np.pi / 180)
        gps_x_e1, gps_y_e1 = map(gps_data['longitude'] + x_above, gps_data['latitude'] + y_above)
        gps_x_e2, gps_y_e2 = map(gps_data['longitude'] + x_below, gps_data['latitude'] + y_below)

        map.plot(must_x, must_y, markersize=10, label=f'MUST track')
        map.plot(gps_x, gps_y, markersize=10, label=f'GPS track')
        # map.plot(gprmc_x, gprmc_y, markersize=5, label=f'GPRMC track')
        # map.plot(gps_x_e1, gps_y_e1, markersize=5, label=f'GPS error bars')
        # map.plot(gps_x_e2, gps_y_e2, markersize=5, label=f'GPS error bars')
        ax_map.legend()

        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax2.plot(must_data['sim time'], must_data['speed'], label='MUST speed')
        ax2.plot(gps_data['sim time'], gps_data['speed'], label='GPS speed')
        # ax2.plot(gprmc_data['sim time'], gprmc_data['speed'], label='GPRMC speed')
        # ax2.plot(must_data['sim time'], must_data['speed_interpolated'], label='MUST speed (interpolated)')
        ax2.legend()
        ax3.plot(must_data['sim time'], must_data['heading'], label='MUST heading')
        ax3.plot(gps_data['sim time'], gps_data['heading'], label='GPS heading')
        # ax3.plot(gprmc_data['sim time'], gprmc_data['heading'], label='GPRMC heading')
        # ax3.plot(must_data['sim time'], must_data['heading_interpolated'], label='MUST heading (interpolated)')

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
        plt.savefig(os.path.join(output_folder, f'{test_name}_latlon_heading_speed.png'), dpi=100)
        # plt.show()
        plt.clf()

    ## Compute and print/store metrics
    print(f'Test {test_name}')
    # Metric 1: position accuracy (90% <30cm error)
    distance_errors = np.array([haversine_distance(must_data_matched['latitude'].iloc[i],
                                          must_data_matched['longitude'].iloc[i],
                                          gps_data_matched['latitude'].iloc[i],
                                          gps_data_matched['longitude'].iloc[i])
                       for i in range(len(must_data_matched) - 1)])
    distance_mean = np.mean(distance_errors)
    distance_stdev = np.std(distance_errors)
    distances_pct_below_limit = 100 * len(distance_errors[distance_errors < 0.3]) / len(distance_errors)
    print(f'Metric 1, distance error <0.3m. Mean error: {distance_mean:.2f}, stdev: {distance_stdev:.2f}, percentage below 0.3m: {distances_pct_below_limit:.2f}%')
    metric_1_pass = bool(distances_pct_below_limit >= 90)

    # # This plot visualizes the position vs. time from gps and the MUST sensor, to get better insight if the distance error metric looks worse than you expect
    # if plot_results:
    #     fig = plt.figure()
    #     fig.add_subplot(111, projection='3d')
    #     ax = fig.get_axes()
    #     ax = ax[0]
    #     ax.set_title(f'Test {test_name}')
    #     ax.set_xlabel('x (m)')
    #     ax.set_ylabel('y (m)')
    #     ax.set_zlabel('time (seconds)')
    #     must_x = (must_data['longitude'].to_numpy() - intersection_center[0]) * lon_to_x
    #     must_y = (must_data['latitude'].to_numpy() - intersection_center[1]) * lat_to_y
    #     gps_x = (gps_data['longitude'].to_numpy() - intersection_center[0]) * lon_to_x
    #     gps_y = (gps_data['latitude'].to_numpy() - intersection_center[1]) * lat_to_y
    #     ax.scatter(must_x, must_y, must_data['sim time'], label=f'MUST data')
    #     ax.scatter(gps_x, gps_y, gps_data['sim time'], label=f'gps data')
    #     ax.set_xlim(np.min(must_x) - 1, np.max(must_x) + 1)
    #     ax.set_ylim(np.min(must_y) - 1, np.max(must_y) + 1)
    #     ax.set_zlim(must_data['sim time'][0], must_data['sim time'][len(must_data)-1])
    #     # plt.savefig(os.path.join(output_folder, f'{test_name}_position_comparison.png'), dpi=100)
    #     plt.show()
    #     # plt.clf()

    # Metric 2: speed accuracy (90% <3mph error)
    speed_errors = np.abs(np.array([must_data_matched['speed'].iloc[i] - gps_data_matched['speed'].iloc[i]
                       for i in range(len(must_data_matched) - 1)])) * 2.23694
    speed_mean = np.mean(speed_errors)
    speed_stdev = np.std(speed_errors)
    # Compute metric performance, plus 3mph -> m/s to match data
    speed_pct_below_limit = 100 * len(speed_errors[speed_errors < 3]) / len(speed_errors)
    print(f'Metric 2, speed error <3 mph. Mean error: {speed_mean:.2f}, stdev: {speed_stdev:.2f}, percentage below 3 mph: {speed_pct_below_limit:.2f}%')
    metric_2_pass = bool(speed_pct_below_limit >= 90)

    # Metric 3: heading accuracy (90% <3 degrees error)
    must_data_matched['heading_unwrapped'] = np.unwrap(must_data_matched['heading'].to_numpy(), period=360, discont=270)
    gps_data_matched['heading_unwrapped'] = np.unwrap(gps_data_matched['heading'].to_numpy(), period=360, discont=270)
    heading_errors = np.abs(gps_data_matched['heading_unwrapped'] - must_data_matched['heading_unwrapped'])
    heading_mean = np.mean(heading_errors)
    heading_stdev = np.std(heading_errors)
    heading_pct_below_limit = 100 * len(heading_errors[heading_errors < 3]) / len(heading_errors)
    print(f'Metric 3, heading error <3 degrees. Mean error: {heading_mean:.2f}, stdev: {heading_stdev:.2f}, percentage below 3 deg: {heading_pct_below_limit:.2f}%')
    metric_3_pass = bool(heading_pct_below_limit >= 90)

    # Metric 4: detection frequency (30hz +- 3hz)
    deduplicated_timestamp_list = np.sort(np.unique(must_data['epoch_time'].to_numpy()))
    recording_periods = np.diff(deduplicated_timestamp_list)
    freq_mean = 1 / np.mean(recording_periods)
    freq_stdev = np.std(1 / recording_periods)
    freq_pct_below_limit = 100 * len(recording_periods[recording_periods < (1.0 / (30 - 3))]) / len(recording_periods)
    # insert histogram. X axis frequency (0-30), Y axis percentage
    print(f'Metric 4, detection frequency 30hz +- 3hz. Mean frequency: {freq_mean:.2f}, stdev: {freq_stdev:.2f}, percentage above 27hz: {freq_pct_below_limit:.2f}%')
    metric_4_pass = bool(freq_pct_below_limit >= 90)

    # Metric 5: 90% of vehicle IDs do not fluctuate
    vehicle_ids = np.unique(must_data['vehicle id'].to_numpy())
    pct_mathing_class = 0
    track_counts = []
    for vehicle_id in vehicle_ids:
        this_vehicle_data = must_data[must_data['vehicle id'] == vehicle_id]
        num_tracks = len(find_sub_tracks(this_vehicle_data['epoch_time'].to_numpy(),
                                        this_vehicle_data['latitude'].to_numpy(), this_vehicle_data['longitude'].to_numpy())) - 2
        track_counts.append(num_tracks)
    track_counts = np.array(track_counts)
    tracks_mean = np.mean(track_counts)
    tracks_stdev = np.std(track_counts)
    tracks_pct_below_limit = 100 * len(track_counts[track_counts < 1]) / len(track_counts)

    print(f'Metric 5 >90% of IDs do not fluctuate. Mean number of ID swaps: {tracks_mean:.2f}, stdev: {tracks_stdev:.2f}, percentage that did not fluctuate: {tracks_pct_below_limit}%')
    metric_5_pass = bool(tracks_pct_below_limit >= 90)

    print()
    # Store metrics to CSV
    with open(os.path.join(output_folder, f'short_metrics.csv'), 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([test_name, distance_mean, distance_stdev, distances_pct_below_limit, int(metric_1_pass),
                         speed_mean, speed_stdev, speed_pct_below_limit, int(metric_2_pass),
                         heading_mean, heading_stdev, heading_pct_below_limit, int(metric_3_pass),
                         freq_mean, freq_stdev, freq_pct_below_limit, int(metric_4_pass),
                         tracks_mean, tracks_stdev, tracks_pct_below_limit, int(metric_5_pass), ])


def main(args):
    ## Folder/data paths
    base_folder = os.path.join(Path.home(), 'fcp_ws', 'other')
    intersection_image = 'must_sensor_intersection_1.png'
    test_log = os.path.join(base_folder, 'MUST_CP_Week2_test_log_uw_process_9-12.csv')
    novatel_folder = os.path.join(base_folder, 'Novatel Data_Week2_v1.0')
    udp_folder = os.path.join(base_folder, 'MUST UDP Data_Week2_v1.0', 'uw_processed_9-13')
    output_folder = os.path.join(base_folder, 'Analysis_Week2_uw_processed_9-13')

    # Writing header for metrics file
    with open(os.path.join(output_folder, f'short_metrics.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['test name', 'distance mean', 'distance stdev', 'distance pct below limit', 'metric 1 pass/fail',
                         'speed mean', 'speed stdev', 'speed pct below limit', 'metric 2 pass/fail',
                         'heading mean', 'heading stdev', 'heading pct below limit', 'metric 3 pass/fail',
                         'freq mean', 'freq stdev', 'freq pct below limit', 'metric 4 pass/fail',
                         'tracks mean', 'tracks stdev', 'tracks pct below limit', 'metric 5 pass/fail', ])

    test_names = ['MUST-NR_1', 'MUST-ER_1', 'MUST-SR_1', 'MUST-WR_1',
                  'MUST-NS_1', 'MUST-ES_1', 'MUST-SS_1', 'MUST-WS_1',
                  'MUST-NL_1', 'MUST-EL_1', 'MUST-SL_1', 'MUST-WL_1']
    for test_name in test_names:
        generate_plots(test_name, test_log, intersection_image, novatel_folder, udp_folder, output_folder)


if __name__ == "__main__":
    main(sys.argv)
