import numpy as np
import cv2
import skimage


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
                       [325, 321], [310, 315], [609, 274], [584, 285], [757, 538], [682, 507]])

    # pts2 = np.float32([[285, 343], [627, 313*scale*y_scale], [694*scale*x_scale, 501*scale*y_scale], [480*scale*x_scale, 670]])
    # pts2 = np.float32([[285, 343], [627, 313], [694, 501], [480, 670]])
    # Points from google maps, must_sensor_intersection_1_rescaled.png
    object_points = np.float32([[132, 425], [346, 497], [288, 342], [108, 52], [479, 585], [480, 670], [640, 530], [705, 538],
                       [732, 436], [1062, 428], [885, 188], [971, 191], [677, 346], [340, 288], [525, 270], [518, 4],
                       [235, 446], [256, 473], [433, 562], [234, 415], [249, 387], [818, 439], [1138, 436], [472, 446],
                       [688, 502], [653, 512], [613, 493], [325, 126], [343, 4], [706, 8], [310, 288], [145, 257],
                       [349, 444], [349, 361], [223, 389], [164, 389], [187, 389], [200, 446], [225, 469], [138, 300],
                       [105, 180], [105, 153], [106, 128], [432, 595], [472, 271], [613, 360], [296, 443], [259, 444],
                       [472, 310], [470, 289], [682, 411], [651, 410], [482, 558], [486, 538]])
    object_points_3d = np.float32(np.append(object_points, np.zeros((len(object_points), 1)), 1))
    points_allowed = [1, 17, 16, 19, 34, 36, 2, 30, 13, 32, 33, 44, 14, 23, 45, 12, 8, 26, 24, 25, 6, 18, 4, 7,
                      46, 47, 48, 49, 50, 51, 52, 53]
    # pt 50 (28 in accepted) -1 x pixel objects
    # pt  (15 in accepted) [526, 259
    img_points = np.float32([img_points[i] for i in points_allowed])
    object_points = np.float32([object_points[i] for i in points_allowed])

    # Camera matrix and distortion coefficients
    K = np.array([[1120, 0, 637], [0, 1320, 383], [0, 0, 1]])
    d = np.array([-0.562, 0.075, 0.010, 0.009, 0])
    # K = np.array([[1140, 0, 639], [0, 1390, 388], [0, 0, 1]])
    # d = np.array([-0.60, 0.1014, 0.0197, 0.0073, 0])
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (1280, 720), 1, (1280, 720))

    # Undistort the camera image
    und = cv2.undistort(gray_camera_image, K, d, None, newcameramatrix)

    # Undistort the points
    distorted_points = img_points.reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(distorted_points, K, d, P=newcameramatrix)

    # Find homography
    H_und, status = cv2.findHomography(undistorted_points, object_points, method=cv2.RANSAC)
    print(f'homography matrix: {H_und}')

    # Compute the perspective transformation matrix for warping
    H = K @ H_und @ np.linalg.inv(K)

    # Warp perspective of the camera image
    # result = cv2.warpPerspective(und, H_und, (1280, 720))
    result = cv2.warpPerspective(gray_camera_image, H, (1280, 720))

    # Reproject the object points
    reprojected_points = cv2.perspectiveTransform(distorted_points, H)
    reprojected_points = reprojected_points.reshape(-1, 2)

    # Compute reprojection error
    error = np.linalg.norm(object_points - reprojected_points, axis=1)
    mean_error = np.mean(error)
    print(f'reprojection error: {mean_error}')

    # display_pairs(gray_camera_image, img_points, gray_maps_image, object_points)

    # Display results
    cv2.imshow('frame_warped', result)
    added_image = cv2.addWeighted(gray_maps_image, 0.4, result, 0.5, 0)
    cv2.imshow('overlay', added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
