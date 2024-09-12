import numpy as np
import cv2
import skimage
import os
import csv

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
# X * 0.0757 + -26.88 = 0, X = 355
# Y * -0.122 + -17.77 = 0, Y = 146
# img * 0.0757 - 26.88 = meter
# img = (meter - lower_left_x) / img_x_to_local_x

M_img_to_meters = np.array([[img_x_to_local_x, 0, lower_left_x],
                   [0, -img_y_to_local_y, upper_right_y],
                   [0, 0, 1]])
# newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))

# K = np.array([[1120, 0, 638], [0, 1330, 384], [0, 0, 1]])
# d = np.array([-0.562, 0.06, 0.01, 0.009, 0])
# newcameramatrix = np.array([[326, 0, 724], [0, 523, 375], [0, 0, 1]])
# H_und_final = np.array([[-4.82123442e-02, 2.95270511e-02, 1.73626347e+01],
#                         [ 5.02026236e-02, 2.21597888e-02, -4.89406274e+01],
#                         [-4.43067120e-04, -2.93027231e-03, 1.00000000e+00]])
# H_und_nometers = np.array([[-7.93599077e-01, -6.50039543e-01, 5.84020354e+02],
#                         [-6.64004545e-01, -1.86580638e+00, 9.74007023e+02],
#                         [-4.43067120e-04, -2.93027231e-03, 1.00000000e+00]])

# 2024_08_28_10_56_00.21 3 2 1 1015 544 133 84 47.62764556748977 -122.14306385256734  0.0 0.0
# 2024_08_28_10_56_24.57 344 2 10 201 346 44 45 47.62800703622254 -122.1432382396437  2.5264302245829278 -18.994369088108694
x_test = (-122.1432382396437 - zero_pos[0]) * lon_to_x
y_test = (47.62800703622254 - zero_pos[1]) * lat_to_y
x_test2 = (-122.1432382396437 - lower_left_longitude) * lon_to_x + lower_left_x
y_test2 = (47.62800703622254 - lower_left_latitude) * lat_to_y + lower_left_y
x_test_img = (x_test - lower_left_x) / img_x_to_local_x
y_test_img = (y_test - lower_left_y) / img_y_to_local_y
x_test3 = 201 * img_x_to_local_x + lower_left_x
y_test3 = 346 * img_y_to_local_y + lower_left_y

def image_xy_to_local_xy_meters(image_x, image_y):
    distorted_points = np.float32([[image_x, image_y]]).reshape(-1, 1, 2)
    image_coords_und = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    # image_coords = cv2.perspectiveTransform(image_coords_und, H_und_nometers)
    # local_coords_conv = (M_img_to_meters @ np.array([[image_coords[0, 0, 0]], [image_coords[0, 0, 1]], [1]])).T[0:2]
    local_coords = cv2.perspectiveTransform(image_coords_und, H_und_final)

    return local_coords[0, 0, 0], local_coords[0, 0, 1]


def draw_points(image, points):
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
        cv2.putText(image, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)


def display_pairs(image1, points1, image2, points2):
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
    draw_points(image1, points1)
    draw_points(image2, points2)

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


def load_csv_data(filename):

    img_points = []
    object_points = []
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        header = True
        for line in reader:
            if header:
                header = False
                continue
            camera_x, camera_y, google_x, google_y = line
            img_points.append([camera_x, camera_y])
            object_points.append([google_x, google_y])
    return img_points, object_points


def test_homography():
    # 635, 864
    x_m, y_m = image_xy_to_local_xy_meters(201, 346)
    # x_m, y_m = image_xy_to_local_xy_meters(199, 385)
    print()


def main():
    reference = cv2.imread('/home/annika/Documents/LEIDOS/Freight CP/must_sensor_intersection_1_rescaled.png')
    gray_maps_image = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    # reference = skimage.transform.resize(reference, (720, 1280))
    # reference = cv2.normalize(reference, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite('/home/annika/Documents/LEIDOS/Freight CP/must_sensor_intersection_1_rescaled.png', reference)
    # reference = reference.dtype(np.uint8)
    frame = cv2.imread('/home/annika/Documents/LEIDOS/Freight CP/must_sensor_image.png')
    gray_camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pts1 = np.float32([[157, 443], [458, 265], [756, 313], [1106, 667]])
    # Points from the MUST sensor view, must_sensor_image.png
    img_points = np.float32([[22, 673], [362, 630], [160, 441], [54, 360], [854, 584], [1106, 667], [784, 358], [832, 325],
                       [673, 268], [775, 206], [512, 188], [558, 175], [526, 257], [199, 371], [346, 286], [246, 236],
                       [132, 634], [184, 681], [707, 631], [119, 576], [130, 518], [716, 246], [797, 200], [462, 407],
                       [754, 315], [760, 340], [698, 344], [169, 317], [163, 272], [346, 207], [170, 386], [45, 451],
                       [286, 515], [229, 419], [102, 540], [45, 581], [66, 567], [92, 664], [137, 702], [36, 489],
                       [33, 416], [37, 403], [42, 390], [865, 706], [306, 303], [481, 285], [211, 572], [162, 610],
                       [325, 321], [310, 315], [609, 274], [584, 285], [757, 538], [682, 507], [923, 609], [1029, 646]])

    # pts2 = np.float32([[285, 343], [627, 313*scale*y_scale], [694*scale*x_scale, 501*scale*y_scale], [480*scale*x_scale, 670]])
    # pts2 = np.float32([[285, 343], [627, 313], [694, 501], [480, 670]])
    # Points from google maps, must_sensor_intersection_1_rescaled.png
    object_points = np.float32([[132, 425], [346, 497], [288, 342], [108, 52], [479, 585], [480, 670], [640, 530], [705, 538],
                       [732, 436], [1062, 428], [885, 188], [971, 191], [677, 346], [340, 288], [525, 270], [518, 4],
                       [235, 446], [256, 473], [433, 562], [234, 415], [249, 387], [818, 439], [1138, 436], [472, 446],
                       [688, 502], [653, 512], [613, 493], [325, 126], [343, 4], [706, 8], [310, 288], [145, 257],
                       [349, 444], [349, 361], [223, 389], [164, 389], [187, 389], [200, 446], [225, 469], [138, 300],
                       [105, 180], [105, 153], [106, 128], [432, 595], [472, 271], [613, 360], [296, 443], [259, 444],
                       [472, 310], [470, 289], [682, 411], [651, 410], [482, 558], [486, 538], [480, 612], [479, 642]])
    object_points_3d = np.float32(np.append(object_points, np.zeros((len(object_points), 1)), 1))

    # img_points2, object_points2 = load_csv_data('/home/annika/fcp_ws/other/Analysis_UWTest2/MUST-NL_2_vehicle_image_data.csv')
    # display_pairs(gray_camera_image, img_points2, gray_maps_image, object_points2)

    object_points_norm = np.float32(np.append(object_points, np.ones((len(object_points), 1)), 1))
    object_points_meters = np.float32((M_img_to_meters @ object_points_norm.T).T[:, 0:2])

    points_allowed = [1, 17, 16, 19, 34, 36, 2, 30, 13, 32, 33, 44, 14, 23, 45, 12, 8, 26, 24, 25, 6, 18, 4, 7,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    # pt 50 (28 in accepted) -1 x pixel objects
    # pt  (15 in accepted) [526, 259
    img_points_allowed = np.float32([img_points[i] for i in points_allowed])
    object_points_allowed = np.float32([object_points[i] for i in points_allowed])
    object_points_meters_allowed = np.float32([object_points_meters[i] for i in points_allowed])

    # Camera matrix and distortion coefficients
    # K = np.array([[1120, 0, 638], [0, 1330, 384], [0, 0, 1]])
    # d = np.array([-0.562, 0.06, 0.01, 0.009, 0])
    # Best for all points, 1.15 pixels
    # K = np.array([[1140, 0, 639], [0, 1390, 388], [0, 0, 1]])
    # d = np.array([-0.60, 0.1014, 0.0197, 0.0073, 0])
    # BR, BL better
    # K = np.array([[1140, 0, 550], [0, 1390, 300], [0, 0, 1]])
    # d = np.array([-0.60, 0.1014, 0.0197, 0.0073, 0])
    K = np.array([[1140, 0, 550], [0, 1390, 300], [0, 0, 1]])
    d = np.array([-0.7, 0.1014, 0.0197, 0.0073, 0])
    # K = np.array([[1000, 0, 640], [0, 1000, 360] ,[0, 0, 1]])
    # d = np.array([[0, 0, 0, 0, 0.0]])
    # rep_error2, K, d, rvecs, tvecs = cv2.calibrateCamera([object_points_3d], [img_points], (1280, 720), None, None)
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))
    print(f'newmcaeramatrix: {newcameramatrix}')
    print(K)
    print(d)
    # Undistort the camera image
    und = cv2.undistort(gray_camera_image, K, d, None, newcameramatrix)

    # Undistort the points
    distorted_points = img_points.reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)
    distorted_points_allowed = img_points_allowed.reshape(-1, 1, 2)
    undistorted_points_allowed = cv2.undistortPoints(distorted_points_allowed, K, d, P=newcameramatrix)

    # Find homography
    # H_und_nometers, status = cv2.findHomography(undistorted_points, object_points, method=cv2.RANSAC)
    H_und_nometers, status = cv2.findHomography(undistorted_points_allowed, object_points_allowed, method=cv2.RANSAC)
    print(f'homography matrix image coords: {H_und_nometers}')

    # Warp perspective of the camera image
    result = cv2.warpPerspective(und, H_und_nometers, (1280, 720))
    # result = cv2.warpPerspective(gray_camera_image, H, (1280, 720))

    # H_und_m, status = cv2.findHomography(undistorted_points, object_points_meters, method=cv2.RANSAC)
    H_und_final = M_img_to_meters @ H_und_nometers
    print(f'homography matrix meters: {H_und_final}')
    # Reproject the object points
    reprojected_points = cv2.perspectiveTransform(undistorted_points_allowed, H_und_final)
    reprojected_points = reprojected_points.reshape(-1, 2)

    # Compute reprojection error
    error = np.linalg.norm(object_points_meters_allowed - reprojected_points, axis=1)
    mean_error = np.mean(error)
    # print(f'\n\n')
    print(f'reprojection error: {mean_error}')

    # display_pairs(gray_camera_image, img_points, gray_maps_image, object_points)

    # Display results
    cv2.imshow('frame_warped', result)
    added_image = cv2.addWeighted(gray_maps_image, 0.4, result, 0.5, 0)
    cv2.imshow('overlay', added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_homography()
    main()
