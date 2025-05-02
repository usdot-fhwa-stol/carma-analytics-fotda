import random
import numpy as np
import cv2
import copy

zero_pos = [-122.143246, 47.627705]
lon_to_x = 111111.0 * np.cos(zero_pos[1] * np.pi / 180)
lat_to_y = 111111.0
x_to_lon = (1 / 111111.0) / np.cos(zero_pos[1] * np.pi / 180)
y_to_lat = (1 / 111111.0)
lower_left_longitude = -122.143605  # Lower-left corner longitude
lower_left_latitude = 47.627545  # Lower-left corner latitude
upper_right_longitude = -122.142310  # Upper-right corner longitude
upper_right_latitude = 47.628340  # Upper-right corner latitude

lower_left_x = (lower_left_longitude - zero_pos[0]) * lon_to_x
lower_left_y = (lower_left_latitude - zero_pos[1]) * lat_to_y
upper_right_x = (upper_right_longitude - zero_pos[0]) * lon_to_x
upper_right_y = (upper_right_latitude - zero_pos[1]) * lat_to_y
img_x_to_local_x = (upper_right_x - lower_left_x) / 1280.0
img_y_to_local_y = (upper_right_y - lower_left_y) / 720.0

M_img_to_meters = np.array([[img_x_to_local_x, 0, lower_left_x],
                   [0, -img_y_to_local_y, upper_right_y],
                   [0, 0, 1]])

x_test = (-122.1432382396437 - zero_pos[0]) * lon_to_x
y_test = (47.62800703622254 - zero_pos[1]) * lat_to_y
x_test2 = (-122.1432382396437 - lower_left_longitude) * lon_to_x + lower_left_x
y_test2 = (47.62800703622254 - lower_left_latitude) * lat_to_y + lower_left_y
x_test_img = (x_test - lower_left_x) / img_x_to_local_x
y_test_img = (y_test - lower_left_y) / img_y_to_local_y
x_test3 = 201 * img_x_to_local_x + lower_left_x
y_test3 = 346 * img_y_to_local_y + lower_left_y


def draw_points(image, points, labels=True):
    """
    Draws points on the image and labels them if provided.

    Args:
    - image: The image to draw on.
    - points: List of points to draw.
    - labels: Optional list of labels for each point.
    """
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        if labels:
            cv2.putText(image, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)


def display_pairs(image1, points1, image2, points2, labels=True):
    """
    Displays two images side by side with corresponding points and labels.

    Args:
    - image1: The first image.
    - points1: Points on the first image.
    - labels1: Labels for points on the first image.
    - image2: The second image.
    - points2: Points on the second image.
    - labels2: Labels for points on the second image.
    """
    # Draw points and labels on images
    draw_points(image1, points1, labels=labels)
    draw_points(image2, points2, labels=labels)

    # Resize images to the same height for side-by-side display
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    new_height = min(height1, height2)
    image1_resized = cv2.resize(image1, (int(width1 * new_height / height1), new_height))
    image2_resized = cv2.resize(image2, (int(width2 * new_height / height2), new_height))

    # Concatenate images horizontally
    combined_image = np.hstack((image1_resized, image2_resized))

    # Display the result
    cv2.imshow('Image Pair with Points', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Computes the reprojection error in meters for the given intrinsics and optimal homography matrix
def measure_reproj_error(K, d, distorted_points, object_points, object_points_meters):
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))
    undistorted_points = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    H_und_nometers, status = cv2.findHomography(undistorted_points, object_points, method=cv2.RANSAC)
    try:
        H_und_final = M_img_to_meters @ H_und_nometers
    except ValueError:
        return 10000

    # Reproject the object points
    reprojected_points = cv2.perspectiveTransform(undistorted_points, H_und_final)
    reprojected_points = reprojected_points.reshape(-1, 2)
    error = np.linalg.norm(object_points_meters - reprojected_points, axis=1)
    mean_error = np.mean(error)
    return mean_error


# Updates and returns the intrinsics and distortion coefficients with one variable stepped based on 'currently_updating'
def step_K_d(K, d, step, currently_updating, currently_updating_map_K):
    if currently_updating <= 3:
        K2 = copy.copy(K)
        K2[tuple(currently_updating_map_K[currently_updating])] += step
        return K2, d
    else:
        d2 = copy.copy(d)
        d2[currently_updating - 4] += step
        return K, d2


# Uses a simulated annealing algorithm to compute an optimized set of intrinsics for the given point pairs
# The 'temperature' of the simulation (variable of the same name) starts high ang gradually decreases over iterations.
#   That temperature is proportional to both the step size and the probability of stepping uphill instead of downhill.
#   Currently, the temperature decreases by 3% per step
#   The probability to step uphill is [random number in the range (0, 1)] < np.power(Temperature, 0.2). This was
#       found to jostle the initial guess sufficiently, while still settling down by the end.
def optimize_intrinsics(img_points, object_points, frame, reference):
    gray_camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_maps_image = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    object_points_norm = np.float32(np.append(object_points, np.ones((len(object_points), 1)), 1))
    object_points_meters = np.float32((M_img_to_meters @ object_points_norm.T).T[:, 0:2])
    K_init = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
    d_init = np.array([-0.5, 0.1, 0.01, 0.0, 0.0])
    K = K_init
    d = d_init
    currently_updating_map_K = [[0, 0], [0, 2], [1, 1], [1, 2]]

    distorted_points = img_points.reshape(-1, 1, 2)
    Temperature = 5
    iter = 0
    # position_multiplier -> [K[0, 0], K[0, 2], K[1, 1], K[1, 2], d[0], d[1], d[2], d[3], d[4]]
    position_multiplier = [1, 0.2, 1, 0.2, 0.1, 0.01, 0.001, 0.001, 0.0001]
    while Temperature > 0.02:
        iter += 1
        Temperature *= 0.97
        # print(f'Temperature: {Temperature}, iteration: {iter}, reproj_error: {measure_reproj_error(K, d, distorted_points, object_points, object_points_meters)}')
        for currently_updating in range(9):
            still_updating = True
            while still_updating:
                step = Temperature * position_multiplier[currently_updating]
                rep_cur = measure_reproj_error(K, d, distorted_points, object_points, object_points_meters)
                K_up, d_up = step_K_d(K, d, step, currently_updating, currently_updating_map_K)
                rep_up = measure_reproj_error(K_up, d_up, distorted_points, object_points, object_points_meters)
                K_down, d_down = step_K_d(K, d, -step, currently_updating, currently_updating_map_K)
                rep_down = measure_reproj_error(K_down, d_down, distorted_points, object_points, object_points_meters)
                choice = random.random()*2 + 1

                if rep_cur <= rep_up + 0.0001 and rep_cur <= rep_down + 0.0001:
                    if choice < np.power(Temperature, 0.2):
                        if rep_up <= rep_cur and rep_up <= rep_down:
                            K, d = K_up, d_up
                        else:
                            K, d = K_down, d_down
                    still_updating = False
                elif rep_up <= rep_cur and rep_up <= rep_down:
                    K, d = K_up, d_up
                else:
                    K, d = K_down, d_down
    print(f'------DONE--------')
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))
    undistorted_points = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    H_und_nometers, status = cv2.findHomography(undistorted_points, object_points, method=cv2.RANSAC)
    H_und_final = M_img_to_meters @ H_und_nometers

    # Reproject the object points
    reprojected_points = cv2.perspectiveTransform(undistorted_points, H_und_final)
    reprojected_points = reprojected_points.reshape(-1, 2)
    error = np.linalg.norm(object_points_meters - reprojected_points, axis=1)
    mean_error = np.mean(error)
    print('----------------------------')
    print(f'K: {K}')
    print(f'd: {d}')
    print(f'newcameramatrix: {newcameramatrix}')
    print(f'H: {H_und_final}')
    print(f'reprojection error: {mean_error}')
    print('----------------------------')

    und = cv2.undistort(gray_camera_image, K, d, None, newcameramatrix)
    result = cv2.warpPerspective(und, H_und_nometers, (1280, 720))
    # Display results
    cv2.imshow('frame_warped', result)
    added_image = cv2.addWeighted(gray_maps_image, 0.4, result, 0.5, 0)
    cv2.imshow('overlay', added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return K, d, mean_error


def main():
    reference = cv2.imread('must_sensor_intersection_1_rescaled.png')
    gray_maps_image = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    frame = cv2.imread('must_sensor_image.png')
    gray_camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Points from the MUST sensor view, must_sensor_image.png
    img_points = [[22, 673], [362, 630], [160, 441], [54, 360], [854, 584], [1106, 667], [784, 358], [832, 325],
                       [673, 268], [775, 206], [512, 188], [558, 175], [526, 257], [199, 371], [346, 286], [246, 236],
                       [132, 634], [184, 681], [707, 631], [119, 576], [130, 518], [716, 246], [797, 200], [462, 407],
                       [754, 315], [760, 340], [698, 344], [169, 317], [163, 272], [346, 207], [170, 386], [45, 451],
                       [286, 515], [229, 419], [102, 540], [45, 581], [66, 567], [92, 664], [137, 702], [36, 489],
                       [33, 416], [37, 403], [42, 390], [865, 706], [306, 303], [481, 285], [211, 572], [162, 610],
                       [325, 321], [310, 315], [609, 274], [584, 285], [757, 538], [682, 507], [923, 609], [1029, 646]]

    # Points from google maps, must_sensor_intersection_1_rescaled.png
    object_points = [[132, 425], [346, 497], [288, 342], [108, 52], [479, 585], [480, 670], [640, 530], [705, 538],
                       [732, 436], [1062, 428], [885, 188], [971, 191], [677, 346], [340, 288], [525, 270], [518, 4],
                       [235, 446], [256, 473], [433, 562], [234, 415], [249, 387], [818, 439], [1138, 436], [472, 446],
                       [688, 502], [653, 512], [613, 493], [325, 126], [343, 4], [706, 8], [310, 288], [145, 257],
                       [349, 444], [349, 361], [223, 389], [164, 389], [187, 389], [200, 446], [225, 469], [138, 300],
                       [105, 180], [105, 153], [106, 128], [432, 595], [472, 271], [613, 360], [296, 443], [259, 444],
                       [472, 310], [470, 289], [682, 411], [651, 410], [482, 558], [486, 538], [480, 612], [479, 642]]

    img_points = np.float32(img_points)
    object_points = np.float32(object_points)

    # Adding a third row of all zeros to represent existing on the flat image
    object_points_norm = np.float32(np.append(object_points, np.ones((len(object_points), 1)), 1))
    # Converting the image coordinates to local x/y coordinates in meters (the @ is numpy matrix multiplication)
    object_points_meters = np.float32((M_img_to_meters @ object_points_norm.T).T[:, 0:2])

    # # Uncomment if you want to compute new intrinsics
    # min_reproj_error = 10000
    # for i in range(5):
    #     K_tmp, d_tmp, reproj_error = optimize_intrinsics(img_points, object_points, frame, reference)
    #     if reproj_error < min_reproj_error:
    #         K = K_tmp
    #         d = d_tmp
    K = np.array([[8.94429165e+02, 0.00000000e+00, 6.45495370e+02],
                  [0.00000000e+00, 1.12363936e+03, 4.20210159e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    d = np.array([-0.51498051, 0.10524621, -0.00603029, -0.02139855, 0.00616998])

    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))
    print(f'newmcaeramatrix: {newcameramatrix}')
    print(K)
    print(d)
    # Undistort the camera image
    und = cv2.undistort(gray_camera_image, K, d, None, newcameramatrix)

    # Undistort the points
    distorted_points = img_points.reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)

    # Find homography
    H_und_nometers, status = cv2.findHomography(undistorted_points, object_points, method=cv2.RANSAC)
    print(f'homography matrix image coords: {H_und_nometers}')

    # Warp perspective of the camera image
    result = cv2.warpPerspective(und, H_und_nometers, (1280, 720))

    H_und_final = M_img_to_meters @ H_und_nometers
    print(f'homography matrix meters: {H_und_final}')
    # Reproject the object points
    reprojected_points = cv2.perspectiveTransform(undistorted_points, H_und_nometers)
    reprojected_points = reprojected_points.reshape(-1, 2)

    # Compute reprojection error
    error = np.linalg.norm(object_points - reprojected_points, axis=1)
    mean_error = np.mean(error)
    print(f'reprojection error: {mean_error}')

    # Display results
    cv2.imshow('frame_warped', result)
    added_image = cv2.addWeighted(gray_maps_image, 0.4, result, 0.5, 0)
    cv2.imshow('overlay', added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
