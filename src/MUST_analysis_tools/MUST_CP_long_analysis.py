import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.basemap import Basemap
from matplotlib.image import imread
import os
from pathlib import Path
from enum import Enum
import csv

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


# Loads filenames/similar from the test log
def load_metadata(test_name, test_log_fname):
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
    must_filename = test_log['UDP file name'][0]

    return must_filename


# Convert MUST sensor output local x/y in meters to lat/lon
def lat_lon_from_x_y_must(x, y):
    longitude = MUST_sensor_loc[0] + x * x_to_lon
    latitude = MUST_sensor_loc[1] + y * y_to_lat
    return latitude, longitude


# Do the analysis and plot the results
def generate_plots(test_name, test_log, intersection_image_path, must_folder, output_folder):

    ## Load data from test log
    must_filename = load_metadata(test_name, test_log)
    plot_results = True

    ## Load MUST sensor data
    must_header = ['class str', 'x', 'y', 'heading', 'speed', 'size', 'confidence', 'vehicle id', 'epoch_time']
    must_data = pd.read_csv(str(os.path.join(must_folder, must_filename)), names=must_header)
    must_data['class id'] = must_data['class str'].apply(lambda x: Object_class(x).value)
    must_data['vehicle id'] = must_data['vehicle id'].astype(np.int32)
    # Get lat/lon from local x/y coordinates
    must_data['latitude'], must_data['longitude'] = lat_lon_from_x_y_must(must_data['x'].to_numpy(), must_data['y'].to_numpy())
    must_data.sort_values('epoch_time')
    must_data['sim time'] = must_data['epoch_time'] - must_data['epoch_time'][0]

    ## Create lat/lon figure of all vehicle tracks
    # We are working under the assumption that there are NO vehicles with duplicate IDs
    if plot_results:
        # Set up the figure and axes
        fig = plt.figure(figsize=(12, 12), dpi=100)
        ax_map = fig.add_subplot()

        # Create a Basemap instance
        intersection_image = imread(intersection_image_path)
        map = Basemap(projection='merc', llcrnrlat=lower_left_latitude, urcrnrlat=upper_right_latitude,
                    llcrnrlon=lower_left_longitude, urcrnrlon=upper_right_longitude, resolution='i', ax=ax_map)

        # Show the image as the background
        map.imshow(intersection_image, origin='upper')

        # Convert lat/lon to map projection coordinates
        vehicle_ids = np.unique(must_data['vehicle id'].to_numpy())
        for vehicle_id in vehicle_ids:
            this_vehicle_data = must_data[must_data['vehicle id'] == vehicle_id]
            veh_x, veh_y = map(this_vehicle_data['longitude'], this_vehicle_data['latitude'])
            map.plot(veh_x, veh_y, markersize=5, label=vehicle_id)

        fig.suptitle(f'Test {test_name}')
        plt.savefig(os.path.join(output_folder, f'{test_name}_latlon.png'), dpi=100)

    ## Compute and print/store metrics
    print(f'Test {test_name}')
    # Metric 6: confidence score >90% for >90% of vehicle tracks
    vehicle_ids = np.unique(must_data['vehicle id'].to_numpy())
    confidence_mean = 0
    confidence_stdev = 0
    passed_count = 0
    for vehicle_id in vehicle_ids:
        this_vehicle_data = must_data[must_data['vehicle id'] == vehicle_id]
        confidences = this_vehicle_data['confidence'].to_numpy()
        confidence_mean_veh = np.mean(confidences)
        confidence_stdev_veh = np.std(confidences)
        confidence_pct_below_limit_veh = 100 * len(confidences[confidences > 90]) / len(confidences)
        confidence_mean += confidence_mean_veh
        confidence_stdev += confidence_stdev_veh
        if confidence_pct_below_limit_veh > 90:
            passed_count += 1
    confidence_mean /= len(vehicle_ids)
    confidence_stdev /= len(vehicle_ids)
    passed_pct_confidence = 100 * passed_count / len(vehicle_ids)
    print(f'Metric 6, confidence score >90% per-vehicle data. Mean confidence: {confidence_mean}, stdev: {confidence_stdev}, percentage >90%: {passed_pct_confidence}%')
    metric_6_pass = bool(passed_pct_confidence >= 90)

    # Metric 7: detection frequency (30hz +- 3hz)
    deduplicated_timestamp_list = np.sort(np.unique(must_data['epoch_time'].to_numpy()))
    recording_periods = np.diff(deduplicated_timestamp_list)
    freq_mean = 1 / np.mean(recording_periods)
    freq_stdev = np.std(1 / recording_periods)
    freq_pct_below_limit = 100 * len(recording_periods[recording_periods < (1.0 / (30 - 3))]) / len(recording_periods)
    # insert histogram. X axis frequency (0-30), Y axis percentage
    print(f'Metric 7, detection frequency 30hz +- 3hz. Mean frequency: {freq_mean:.2f}, stdev: {freq_stdev:.2f}, percentage above 27hz: {freq_pct_below_limit:.2f}%')
    metric_7_pass = bool(freq_pct_below_limit >= 90)

    # Metric 8: 90% of vehicle IDs do not fluctuate
    vehicle_ids = np.unique(must_data['vehicle id'].to_numpy())
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
    print(f'Metric 8, >90% of IDs do not fluctuate. Mean number of ID swaps: {tracks_mean:.2f}, stdev: {tracks_stdev:.2f}, percentage that did not fluctuate: {tracks_pct_below_limit}%')
    metric_8_pass = bool(tracks_pct_below_limit >= 90)

    with open(os.path.join(output_folder, f'long_metrics.csv'), 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([test_name, confidence_mean, confidence_stdev, passed_pct_confidence, int(metric_6_pass),
                         freq_mean, freq_stdev, freq_pct_below_limit, int(metric_7_pass),
                         tracks_mean, tracks_stdev, tracks_pct_below_limit, int(metric_8_pass)])


def main(args):
    ## Folder/data paths
    base_folder = os.path.join(Path.home(), 'fcp_ws', 'other')
    intersection_image = 'must_sensor_intersection_1.png'
    test_log = os.path.join(base_folder, 'MUST_CP_Week2_test_log.csv')
    udp_folder = os.path.join(base_folder, 'MUST UDP Data_Week2_v1.0', 'uw_processed_9-18')
    output_folder = os.path.join(base_folder, 'Analysis_Week2_uw_processed_9-18')

    # Writing header for metrics file
    with open(os.path.join(output_folder, f'long_metrics.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            ['test name', 'confidence mean', 'confidence stdev', 'confidence pct below limit', 'metric 6 pass/fail',
             'freq mean', 'freq stdev', 'freq pct below limit', 'metric 7 pass/fail',
             'tracks mean', 'tracks stdev', 'tracks pct below limit', 'metric 8 pass/fail', ])

    test_names = ['MUST-LT_1', 'MUST-LT_2']
    for test_name in test_names:
        generate_plots(test_name, test_log, intersection_image, udp_folder, output_folder)


if __name__ == "__main__":
    main(sys.argv)
