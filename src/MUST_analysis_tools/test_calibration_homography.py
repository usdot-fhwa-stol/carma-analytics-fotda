import numpy as np
import cv2


# # Haversine function to calculate distance between two lat-long points
# def haversine_distance(lat1, lon1, lat2, lon2):
#     from math import radians, sin, cos, sqrt, atan2
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1-a))
#     distance = 6371 * c * 1000  # Convert to meters
#     return abs(distance)
#
#
# uw_must_sensor_loc = np.array([-122.1432525, 47.6277146])
# zero_pos = np.array([-122.143246, 47.627705])
# MUST_sensor_loc = np.array([-122.143246, 47.627705])
# intersection_center = np.array([-122.1431239, 47.6278859])
# lon_to_x = 111111.0 * np.cos(intersection_center[1] * np.pi / 180)
# lat_to_y = 111111.0
# x_to_lon = (1 / 111111.0) / np.cos(intersection_center[1] * np.pi / 180)
# y_to_lat = (1 / 111111.0)
# lower_left_longitude = -122.1436116  # Lower-left corner longitude
# lower_left_latitude = 47.6275314  # Lower-left corner latitude
# upper_right_longitude = -122.1423166  # Upper-right corner longitude
# upper_right_latitude = 47.6283264  # Upper-right corner latitude
#
# lower_left_x = (lower_left_longitude - zero_pos[0]) * lon_to_x
# lower_left_y = (lower_left_latitude - zero_pos[1]) * lat_to_y
# upper_right_x = (upper_right_longitude - zero_pos[0]) * lon_to_x
# upper_right_y = (upper_right_latitude - zero_pos[1]) * lat_to_y
# img_x_to_local_x = (upper_right_x - lower_left_x) / 1280.0
# img_y_to_local_y = (upper_right_y - lower_left_y) / 720.0
# img_x_to_lon = (upper_right_longitude - lower_left_longitude) / 1280.0
# img_y_to_lat = (upper_right_latitude - lower_left_latitude) / 720.0
# # image to lat/lon
#
#
# M_img_to_meters = np.array([[img_x_to_local_x, 0, lower_left_x],
#                    [0, -img_y_to_local_y, upper_right_y],
#                    [0, 0, 1]])
# M_img_to_latlon = np.array([[img_x_to_lon, 0, lower_left_longitude],
#                    [0, -img_y_to_lat, upper_right_latitude],
#                    [0, 0, 1]])

# All 0.9052174687385559
# Insanely good with 3/4 height offset
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

def vehicle_center_to_latlon(center, bbox):
    image_cx, image_cy = center
    x1, y1, x2, y2 = bbox
    bbox_height = np.abs(y2 - y1)
    distorted_points = np.float32(np.vstack((image_cx, image_cy + bbox_height*0.25)).T).reshape(-1, 1, 2)
    image_coords_und = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    latlon_coords = cv2.perspectiveTransform(image_coords_und, H)
    lon_offset = 6.677e-06
    lat_offset = 4.500e-06
    lon_final = latlon_coords[0, 0, 0] + lon_offset # 0.5, 0.75 0.905
    lat_final = latlon_coords[0, 0, 1] - lat_offset # 0.5, 0.75 0.905
    # print(f'final lat/lon: [{lat_final}, {lon_final}]')
    # must_sensor_dist = haversine_distance(MUST_sensor_loc[1], MUST_sensor_loc[0], lat_final, lon_final)
    # print(f'distance from MUST sensor: {must_sensor_dist}')

    return (lat_final, lon_final)


def main():
    # 1015 544 133 84 47.62764556748977 -122.14306385256734
    xc = 1180 # xc = 1015 + 133/2 = 1180
    yc = 586 # yc = 544 + 84/2 = 586
    center = (xc, yc)
    bbox = (1015, 544, 1015+133, 544+84) # height = 133
    must_measured_latlon = (47.62764556748977, -122.14306385256734)
    center_latlon = vehicle_center_to_latlon(center, bbox)
    print(f'calibrated lat/lon: [{center_latlon[0]}, {center_latlon[1]}]')
    print(f'MUST lat/lon: [{must_measured_latlon[0]}, {must_measured_latlon[1]}]')
    # lat_diff = must_measured_latlon[0] - center_latlon[0]
    # lon_diff = must_measured_latlon[1] - center_latlon[1]
    # lat_diff_m = lat_diff * lat_to_y
    # lon_diff_m = lon_diff * lon_to_x
    # print(f'lat diff: {lat_diff_m}, lon_diff: {lon_diff_m}')


if __name__ == "__main__":
    main()
