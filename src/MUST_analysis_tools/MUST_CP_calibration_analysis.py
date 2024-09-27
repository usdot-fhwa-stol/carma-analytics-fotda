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
from pathlib import Path
from enum import Enum
import cv2
import csv
from pyproj import Transformer


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

MUST_sensor_loc = np.array([-122.143246, 47.627705])  # [longitude, latitude]
intersection_center = np.array([-122.1431239, 47.6278859])  # [longitude, latitude]

# Conversion done using a local tangent plane approximation (accurate to <1mm at 1km), with a zero point of
#   [lat_zero, lon_zero] -> intersection_center
# For small regions on the earth, latitude and longitude lines are essentially perpendicular and can be used as a
#   cartesian coordinate system. The convention used here is NWU (x -> north, y -> west, z -> up (here we use 2D so z is
#   ignored)). 90 degrees of longitude is defined as 1/4 the earth circumference, or ~10 million meters. Therefore,
#   latitude is equal to: meters_per_degree(90 degrees / 10E6 meters) * (lat - lat_zero). The longitudinal circumference
#   is a function of latitude, proportional to the cos of latitude. longitude is equal to: meters_per_degree(
#   90 degrees / 10E6 meters * cos(lat_zero)) * (lon - lon_zero).
lon_to_x = 111111.0 * np.cos(intersection_center[1] * np.pi / 180)
lat_to_y = 111111.0
x_to_lon = (1 / 111111.0) / np.cos(intersection_center[1] * np.pi / 180)
y_to_lat = (1 / 111111.0)
lower_left_longitude = -122.1436116  # Lower-left corner longitude
lower_left_latitude = 47.6275314  # Lower-left corner latitude
upper_right_longitude = -122.1423166  # Upper-right corner longitude
upper_right_latitude = 47.6283264  # Upper-right corner latitude

lower_left_x = (lower_left_longitude - MUST_sensor_loc[0]) * lon_to_x
lower_left_y = (lower_left_latitude - MUST_sensor_loc[1]) * lat_to_y
upper_right_x = (upper_right_longitude - MUST_sensor_loc[0]) * lon_to_x
upper_right_y = (upper_right_latitude - MUST_sensor_loc[1]) * lat_to_y
img_x_to_local_x = (upper_right_x - lower_left_x) / 1280.0
img_y_to_local_y = (upper_right_y - lower_left_y) / 720.0
img_x_to_lon = (upper_right_longitude - lower_left_longitude) / 1280.0
img_y_to_lat = (upper_right_latitude - lower_left_latitude) / 720.0

M_img_to_meters = np.array([[img_x_to_local_x, 0, lower_left_x],
                   [0, -img_y_to_local_y, upper_right_y],
                   [0, 0, 1]])
M_img_to_latlon = np.array([[img_x_to_lon, 0, lower_left_longitude],
                   [0, -img_y_to_lat, upper_right_latitude],
                   [0, 0, 1]])
# Set the bounds of the map (in lat/lon coordinates)
lower_left_longitude = -122.1436116  # Lower-left corner longitude
lower_left_latitude = 47.6275314  # Lower-left corner latitude
upper_right_longitude = -122.1423166  # Upper-right corner longitude
upper_right_latitude = 47.6283264  # Upper-right corner latitude

# # Intersection 0.3111535608768463
# # Quite good
# K = [[9.67481868e+02, 0.00000000e+00, 6.46144849e+02],
#  [0.00000000e+00, 1.02946621e+03, 3.49298579e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# d = [-0.80916231,  0.52110152,  0.0016839,  -0.00374791, -0.14427669]
# newcameramatrix = [[104.30975877,   0.,         255.37791574],
#  [  0.,         344.09854501, 244.28976651],
#  [  0.,           0.,           1.        ]]
# H = [[-1.60736160e-01,  4.37323614e-02,  2.05116017e+01],
#  [ 2.16078130e-01,  3.09327514e-02, -6.94170640e+01],
#  [-7.65483284e-04, -5.58295093e-03,  1.00000000e+00]]

# # Intersection 0.308316171169281
# # Good, top left/right meh
# K = [[9.55446941e+02, 0.00000000e+00, 6.42755045e+02],
#  [0.00000000e+00, 1.10529923e+03, 3.93271960e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# d = [-7.95306168e-01,  3.23083873e-01,  4.53204202e-04, -2.91438668e-02, 1.15545090e-02]
# newcameramatrix = [[542.94733087,   0.,         585.54157833],
#  [  0.,         551.39945831, 402.59043169],
#  [  0.,           0.,           1.        ]]
# H = [[-3.52545664e-02,  3.44100246e-02, -3.01476420e+00],
#  [ 4.31464548e-02,  2.43783923e-02, -4.18002903e+01],
#  [-2.01743900e-04, -3.89314072e-03,  1.00000000e+00]]

# # Intersection 0.29929041862487793
# # Okay
# K = [[1.11471890e+03, 0.00000000e+00, 6.40272752e+02],
#  [0.00000000e+00, 1.29345437e+03, 3.91566781e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# d = [-1.16250129,  0.70058963, -0.00344463, -0.04282112, -0.01051898],
# newcameramatrix = [[579.97690153,   0.,         530.42913328],
#  [  0.,         595.46969942, 383.25276888],
#  [  0.,           0.,           1.        ]]
# H = [[-3.60011503e-02,  3.46174107e-02, -3.29580234e+00],
#  [ 4.43074686e-02,  2.30061271e-02, -3.87521256e+01],
#  [-2.09345344e-04, -3.98561922e-03,  1.00000000e+00]]

# # All 0.8685863614082336
# # Insanely good with 0.25 height offset
# K = [[8.88730423e+02, 0.00000000e+00, 6.29019650e+02],
#  [0.00000000e+00, 1.11557049e+03, 3.64531944e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# d = [-5.33402562e-01,  1.31052753e-01,  5.47765301e-03, 4.61787259e-04, 2.01708172e-02]
# newcameramatrix = [[502.41048518,   0.,         632.41972543],
#  [  0.,         642.09241708, 375.92236659],
#  [  0.,           0.,           1.        ]]
# # H undistorted, pixel coords -> latlon
# H = [[ 2.76427066e-02,  5.55785368e-01, -1.22143222e+02],
#  [-1.07785197e-02, -2.16718663e-01,  4.76271630e+01],
#  [-2.26319283e-04, -4.55027047e-03,  1.00000000e+00]]
# # H undistorted, pixel coords -> local x/y in meters
# # H = [[-4.98131259e-02,  4.21563707e-02,  1.78770989e+00],
# #  [ 6.09274042e-02,  3.07399157e-02, -6.02174639e+01],
# #  [-2.26319283e-04, -4.55027047e-03,  1.00000000e+00]]

# All 0.9052174687385559
# Insanely good with 0.25 height offset
K = np.array([[8.94429165e+02, 0.00000000e+00, 6.45495370e+02],
 [0.00000000e+00, 1.12363936e+03, 4.20210159e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
d = np.array([-0.51498051,  0.10524621, -0.00603029, -0.02139855,  0.00616998])
newcameramatrix = np.array([[387.33544107,   0.,         534.09059684],
 [  0.,         505.10401128, 436.17367482],
 [  0.,           0.,           1.        ]])
H = np.array([[ 3.31445642e-02,  4.00938084e-01, -1.22143253e+02],
 [-1.29239710e-02, -1.56338683e-01,  4.76273545e+01],
 [-2.71362493e-04, -3.28251998e-03,  1.00000000e+00]])


# Converts the vehicle's bounding box into lat/lon coordinate using the given homography matrix
def vehicle_center_to_latlon(image_cx, image_cy, bbox_width, bbox_height):
    # image_cx, image_cy = center
    # x1, y1, x2, y2 = bbox
    # bbox_height = np.abs(y2 - y1)
    distorted_points = np.float32(np.vstack((image_cx, image_cy + bbox_height*0.2)).T).reshape(-1, 1, 2)
    image_coords_und = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    latlon_coords = cv2.perspectiveTransform(image_coords_und, H)
    lon_offset = 6.677e-06
    lat_offset = 4.500e-06
    lon_final = latlon_coords[:, 0, 0][0] + lon_offset # 0.5, 0.75 0.905
    lat_final = latlon_coords[:, 0, 1][0] - lat_offset # 0.5, 0.75 0.905
    # print(f'final lat/lon: [{lat_final}, {lon_final}]')
    # must_sensor_dist = haversine_distance(MUST_sensor_loc[1], MUST_sensor_loc[0], lat_final, lon_final)
    # print(f'distance from MUST sensor: {must_sensor_dist}')

    return lat_final, lon_final


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
    plt.savefig(os.path.join(output_folder, f'{test_name}_tracks_by_vehicle_id_0.5_0.7.png'), dpi=100)
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
    return GPS_VEHICLE_ID, track_index, must_filename, gps_filename


# Convert MUST sensor output local x/y in meters to lat/lon
def lat_lon_from_x_y_must(x, y):
    longitude = MUST_sensor_loc[0] + x * x_to_lon
    latitude = MUST_sensor_loc[1] + y * y_to_lat
    return latitude, longitude


def compute_show_metrics(test_name, must_data, gps_data, must_data_matched, gps_data_matched, output_folder, name_str):

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
    #     ax.set_zlim(must_data['sim time'][0], must_data['sim time'][len(must_data) - 1])
    #     # plt.savefig(os.path.join(output_folder, f'{test_name}_position_comparison.png'), dpi=100)
    #     # plt.show()
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
    print(f'Metric 4, detection frequency 30hz +- 3hz. Mean frequency: {freq_mean:.2f}, stdev: {freq_stdev:.2f}, percentage above 27hz: {freq_pct_below_limit:.2f}%')
    metric_4_pass = bool(freq_pct_below_limit >= 90)

    # Metric 5: Object type staying consistent (>90% the same class)
    classes_present = np.bincount(np.int32(must_data['class id'].to_numpy()))
    most_common_class = np.argmax(classes_present)

    pct_mathing_class = 100 * classes_present[most_common_class] / len(must_data)
    print(f'Metric 5, >90% the same class. Most common class: {most_common_class}, percentage matching class: {pct_mathing_class:.2f}%')
    metric_5_pass = bool(pct_mathing_class >= 90)
    print()
    with open(os.path.join(output_folder, f'short_metrics_{name_str}.csv'), 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([test_name, distance_mean, distance_stdev, distances_pct_below_limit, int(metric_1_pass),
                         speed_mean, speed_stdev, speed_pct_below_limit, int(metric_2_pass),
                         heading_mean, heading_stdev, heading_pct_below_limit, int(metric_3_pass),
                         freq_mean, freq_stdev, freq_pct_below_limit, int(metric_4_pass),
                         most_common_class, pct_mathing_class, int(metric_5_pass), ])


# Computes a backwards-looking moving average for up to n elements of the given array
def moving_average(n, data):
    data_smoothed = [data[0]]
    for i in range(1, n):
        tmp = 0
        for j in range(i + 1):
            tmp += data[i - j]
        data_smoothed.append(tmp / (i+1))

    for i in range(n, len(data)):
        tmp = 0
        for j in range(n):
            tmp += data[i - j]
        data_smoothed.append(tmp / n)
    return np.array(data_smoothed)


# Do the analysis and plot the results
def generate_plots(test_name, test_log, intersection_image_path, gps_folder, must_folder, udp_uw_folder, output_folder):

    ## Load data from test log
    GPS_VEHICLE_ID, track_index, must_filename, gps_filename = load_metadata(test_name, test_log, gps_folder)
    must_uw_filename = must_filename[:20] + 'uw_code_base_9-18' + must_filename[-12:]
    plot_results = True

    ## Load MUST sensor data
    must_header = ['class str', 'x', 'y', 'heading', 'speed', 'image_x', 'image_y', 'bbox_width', 'bbox_height', 'lat_uw', 'lon_uw', 'size', 'confidence', 'vehicle id', 'epoch_time']
    must_data = pd.read_csv(str(os.path.join(must_folder, must_filename)), names=must_header)
    must_data['class id'] = must_data['class str'].apply(lambda x: Object_class(x).value)
    must_data['vehicle id'] = must_data['vehicle id'].astype(np.int32)
    must_data = must_data[must_data['vehicle id'] == GPS_VEHICLE_ID].reset_index(drop=True)
    must_data = must_data.tail(-1)
    # Usse updated calibration to compute lat/lon position
    must_latlons = np.array([vehicle_center_to_latlon(must_data['image_x'].iloc[i], must_data['image_y'].iloc[i],
                                             must_data['bbox_width'].iloc[i], must_data['bbox_height'].iloc[i])
                 for i in range(len(must_data))])
    must_data['latitude'], must_data['longitude'] = must_latlons[:, 0], must_latlons[:, 1]
    must_data['lat_from_local'], must_data['lon_from_local'] = lat_lon_from_x_y_must(must_data['x'].to_numpy(), must_data['y'].to_numpy())
    must_data.sort_values('epoch_time')

    ## Compute smoothed values
    must_data['longitude_smoothed'] = moving_average(5, must_data['longitude'].to_numpy())
    must_data['latitude_smoothed'] = moving_average(5, must_data['latitude'].to_numpy())
    # This block computes an interpolated velocity column
    # Calculate distance between consecutive pedestrian points
    distances = [haversine_distance(must_data['latitude_smoothed'].iloc[i],
                                          must_data['longitude_smoothed'].iloc[i],
                                          must_data['latitude_smoothed'].iloc[i + 1],
                                          must_data['longitude_smoothed'].iloc[i + 1])
                       for i in range(len(must_data) - 1)]
    # Calculate time differences between consecutive points
    time_diffs = [(must_data['epoch_time'].iloc[i+1] - must_data['epoch_time'].iloc[i])
                        for i in range(len(must_data)-1)]
    speed_interpolated = [dist/time if time != 0 else 0 for dist, time in zip(distances, time_diffs)]
    must_data['speed_interpolated'] = [speed_interpolated[0]] + speed_interpolated
    must_data['speed_interpolated'] = must_data['speed_interpolated'].round(2)
    must_data.loc[must_data['speed_interpolated'] < MUST_STATIONARY_NOISE, 'speed_interpolated'] = 0

    heading_interpolated = [latlon_angle(must_data['latitude_smoothed'].iloc[i],
                                                    must_data['longitude_smoothed'].iloc[i],
                                                    must_data['latitude_smoothed'].iloc[i + 1],
                                                    must_data['longitude_smoothed'].iloc[i + 1])
                                for i in range(len(must_data) - 1)]
    must_data['heading_interpolated'] = [heading_interpolated[0]] + heading_interpolated
    must_data['heading_interpolated'] = must_data['heading_interpolated'] % 360
    must_data['longitude'] = must_data['longitude_smoothed']
    must_data['latitude'] = must_data['latitude_smoothed']
    must_data['speed'] = must_data['speed_interpolated']
    must_data['heading'] = must_data['heading_interpolated']
    # must_data['heading'] = (must_data['heading_interpolated'] + 180) % 360

    ## Load original UW data
    must_uw_header = ['class str', 'x', 'y', 'heading', 'speed', 'size', 'confidence', 'vehicle id', 'epoch_time']
    must_uw_data = pd.read_csv(str(os.path.join(udp_uw_folder, must_uw_filename)), names=must_uw_header)
    must_uw_data['class id'] = must_uw_data['class str'].apply(lambda x: Object_class(x).value)
    must_uw_data['vehicle id'] = must_uw_data['vehicle id'].astype(np.int32)
    # Convert to unix timestamp (epoch time) in UTC
    must_uw_data = must_uw_data[must_uw_data['vehicle id'] == 62].reset_index(drop=True)
    must_uw_data = must_uw_data.tail(-1)
    must_uw_data['latitude'], must_uw_data['longitude'] = lat_lon_from_x_y_must(must_uw_data['x'].to_numpy(),
                                                                                     must_uw_data['y'].to_numpy())
    must_uw_data.sort_values('epoch_time')

    # Load novatel GPS data
    gps_header = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'latitude stdev', 'longitude stdev', 'altitude stdev', 'heading error', 'speed error']
    gps_data = pd.read_csv(str(os.path.join(gps_folder, gps_filename)), names=gps_header, skiprows=1)
    gps_data['epoch_time'] = gps_data['timestamp']
    # gps_data['heading'] = (gps_data['heading'] + 180) % 360

    ## If track index is undefined in the test log, show the user so they can enter the track ID
    if 'track_index' not in locals() or track_index is None:
        select_sub_track_user_input(must_uw_data, gps_data, test_name, intersection_image_path, output_folder)
        return
    ## If there is a track ID defined, grab all data in that track
    break_indices = find_sub_tracks(must_data['epoch_time'].to_numpy(),
                                                    must_data['latitude'].to_numpy(), must_data['longitude'].to_numpy())
    valid_data_indices = range(break_indices[track_index], break_indices[track_index + 1])
    must_data = must_data.iloc[valid_data_indices[1:]].reset_index(drop=True)
    must_uw_data = must_uw_data.iloc[valid_data_indices[1:]].reset_index(drop=True)

    ## Resample data to 50hz, for time synchronization and metric comparisons
    gps_data_resampled = resample_df(gps_data, 0.02)
    must_data_resampled = resample_df(must_data, 0.02)
    must_uw_data_resampled = resample_df(must_uw_data, 0.02)

    ## Compute and apply the optimal time shift
    time_offset_index = find_optimal_time_shift(gps_data_resampled['latitude'].to_numpy(), gps_data_resampled['longitude'].to_numpy(),
                                                must_data_resampled['latitude'].to_numpy(), must_data_resampled['longitude'].to_numpy())
    time_offset = gps_data_resampled['epoch_time'][time_offset_index] - gps_data_resampled['epoch_time'][0]
    gps_data_matched = gps_data_resampled.copy()
    must_data_matched = must_data_resampled.copy()
    must_uw_data_matched = must_uw_data_resampled.copy()
    gps_data_matched_uw = gps_data_resampled.copy()
    must_data_matched['epoch_time'] = must_data_matched['epoch_time'] + time_offset
    must_uw_data_matched['epoch_time'] = must_uw_data_matched['epoch_time'] + time_offset
    gps_data_matched = gps_data_matched[time_offset_index-1:time_offset_index-1 + len(must_data_matched)].reset_index(drop=True)
    gps_data_matched_uw = gps_data_matched_uw[time_offset_index-1:time_offset_index-1 + len(must_uw_data_matched)].reset_index(drop=True)
    gps_data['sim time'] = gps_data['epoch_time'] - gps_data['epoch_time'][0]#  + (gps_data['epoch_time'][0] - must_data['epoch_time'][0] - 117.5)
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0] + time_offset
    must_uw_data['sim time'] = must_uw_data['epoch_time'] - must_uw_data['epoch_time'][0] + time_offset

    ## Create and save the lat/lon, heading, and speed plots
    if plot_results:
        # Set up the figure and axes
        fig = plt.figure(figsize=(20, 12), dpi=100)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        ax_map = fig.add_subplot(gs[:, :2])

        # Create a Basemap instance
        intersection_image = imread(intersection_image_path)
        map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                    llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax_map)

        # Show the image as the background
        map.imshow(intersection_image, origin='upper')

        # Convert lat/lon to map projection coordinates
        must_x, must_y = map(must_data['longitude'], must_data['latitude'])
        must_x_smoothed, must_y_smoothed = map(must_data['longitude_smoothed'], must_data['latitude_smoothed'])
        gps_x, gps_y = map(gps_data['longitude'], gps_data['latitude'])
        must_uw_x, must_uw_y = map(must_uw_data['longitude'], must_uw_data['latitude'])

        # Plot the data points
        map.plot(must_x_smoothed, must_y_smoothed, markersize=5, label=f'MUST calibrated smoothed')
        map.plot(gps_x, gps_y, markersize=5, label=f'GPS track')
        map.plot(must_uw_x, must_uw_y, markersize=5, label=f'MUST UW track')
        ax_map.legend()

        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax2.plot(must_data['sim time'], must_data['speed_interpolated'], label='MUST cal speed smoothed')
        ax2.plot(gps_data['sim time'], gps_data['speed'], label='GPS speed')
        ax2.plot(must_uw_data['sim time'], must_uw_data['speed'], label='MUST uw speed')
        ax2.legend()
        ax3.plot(must_data['sim time'], must_data['heading_interpolated'], label='MUST cal heading smoothed')
        ax3.plot(gps_data['sim time'], gps_data['heading'], label='GPS heading')
        ax3.plot(must_uw_data['sim time'], must_uw_data['heading'], label='MUST uw heading')

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

    compute_show_metrics(test_name, must_uw_data, gps_data, must_uw_data_matched, gps_data_matched_uw, output_folder, 'uw')
    compute_show_metrics(test_name, must_data, gps_data, must_data_matched, gps_data_matched, output_folder, 'calibrated')


def main(args):
    base_folder = os.path.join(Path.home(), 'fcp_ws', 'other')
    intersection_image = 'must_sensor_intersection_1.png'
    test_log = os.path.join(base_folder, 'MUST_CP_Week2_test_log_annika_process.csv')
    novatel_folder = os.path.join(base_folder, 'Novatel Data_Week2_v1.0')
    udp_folder = os.path.join(base_folder, 'MUST UDP Data_Week2_v1.0', 'annika_calib_v1_plus_coords')
    udp_uw_folder = os.path.join(base_folder, 'MUST UDP Data_Week2_v1.0', 'annika_processed_uw_code_9-12')
    output_folder = os.path.join(base_folder, 'Analysis_calibration')
    test_names = ['MUST-NR_1', 'MUST-ER_1', 'MUST-SR_1', 'MUST-WR_1',
                  'MUST-NS_1', 'MUST-ES_1', 'MUST-SS_1', 'MUST-WS_1',
                  'MUST-NL_1', 'MUST-EL_1', 'MUST-SL_1', 'MUST-WL_1']
    # test_names = ['MUST-NS_1']
    for test_name in test_names:
        generate_plots(test_name, test_log, intersection_image, novatel_folder, udp_folder, udp_uw_folder, output_folder)


if __name__ == "__main__":
    main(sys.argv)
