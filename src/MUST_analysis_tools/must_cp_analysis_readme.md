## Introduction

This code is data analysis for the UDP output from the MUST sensor in the CDA research task 4 Freight cooperative perception project. Testing for that project was completed on September 12th. Testing involved one instrumented vehicle, a black rental car with a novatel GPS unit installed inside and an antenna mounted on the roof. UDP and GPS ground truth data are available on google drive (links below). Raw data (pure video and timestamp files) are available on google drive, but require processing through the model code. The raw data allows you to 

The 'short' testing data and associated python file `MUST_CP_short_analysis.py` contains the instrumented ground truth vehicle, and is intended for testing the position, velocity, and heading of that specific vehicle. The 'long' testing data and associated python file `MUST_CP_long_analysis.py` does not contain the instrumented vehicle, and is intended for analyzing the class/ID switching behavior and manually verifying bounding boxes. 

The other files, `MUST_CP_calibration_analysis.py` and `compute_homography_image_warp.py` are for calibrating the camera without a checkerboard. For the original testing, we did not have any intrinsics computed nor a way to being a checkerboard up to the camera to do it properly. `compute_homography_image_warp.py` contains point correspondences and code to compute the intrinsics. The point correspondences were manually defined and entered, but the opencv function to compute the intrinsics was not working with so few points. A simulated annealing algorithm was created and tuned to find intrinsics through step-by-step optimization, and it works pretty well. The output intrinsics and homography transformation can be fed into `MUST_CP_calibration_analysis.py` to validate the effectiveness of that calibration vs. the original calibration from UW. This is basically the same as `MUST_CP_short_analysis.py`, but set up to read in extended data with image coordinates for the vehicles and process them for testing the calibration. 

## Setup

1. Installation folder
    All of these folder/filenames are defined at the end of each file. The default is in a foder under home named `fcp_ws`, with the code under `src` and the other files in folders under `other`. 
1. Test log
    Already filled columns
    a. test case
    b. run number
    c. UDP file name
    Need to look at/update
    a. Vehicle ID
    b. Track ID (almost always zero)
2. UDP data
    From google drive, default is the most recent testing: https://drive.google.com/drive/u/1/folders/1uwloOWMgiepaIZEeKijmrVNFjPoDDlAf
3. Novatel GPS data
    From google drive, default is the most recent testing: https://drive.google.com/drive/u/1/folders/12tHzJNX_k-XS9Fd_U7kSsihLqjZ4QFMa

## Running the code

python3 MUST_CP_short_analysis.py intersection_image_path test_log_path novatel_folder_path udp_folder_path output_folder_path
In the specified output folder, you will get one lat/lon image per test case, and one line in the short_metrics.csv file

python3 MUST_CP_long_analysis.py intersection_image_path test_log_path udp_folder_path output_folder_path
In the specified output folder, you will get one lat/lon image per test case, and one line in the long_metrics.csv file
