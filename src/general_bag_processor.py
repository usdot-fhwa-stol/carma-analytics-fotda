import multiprocessing
import rosbag #pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
import boto3 #pip install boto3
from urllib.parse import unquote_plus
import s3fs
import csv
from multiprocessing import Pool, Process
from functools import partial
import datetime as dt

s3_client = boto3.client('s3')
fs = s3fs.S3FileSystem()

# User initialized parameters
SRC_BUCKET = 'raw-carma-core-validation-prod'
DST_BUCKET = 'preprocessed-carma-core-validation-prod'

FILE_IDS = ['down-selected', '20210623'] # this can be an list of indetifiers for the folder in which raw data is stored - like, "20210318" and "down-selected"

PARSE_SPECIFIC_VEHICLE = False #set to True tp process a sepcific vehicle's bagfiles and provide 'VEHICLE_ID' (not a list)
VEHICLE_ID = 'Ford'

PARSE_SPECIFIC_BAGFILE = True #set to True if a specific bagfiles need to be parsed and provide bagfile S3 path list in 'BAGFILE_ID' variable
BAGFILE_ID = [
    'bagfiles/Core_Validation_Testing/Facility_Summit_Point/Vehicle_Black_Pacifica/20210623/_2021-06-23-14-02-22_down-selected.bag',
    'bagfiles/Core_Validation_Testing/Facility_Summit_Point/Vehicle_Black_Pacifica/20210623/_2021-06-23-14-17-32_down-selected.bag'
]

PARSE_SPECIFIC_TOPICS = True #set to True if specific topics in the bagfile need to be processed and provide topic list in 'TOPICS_ID'
TOPICS_ID = [
    '/hardware_interface/ds_fusion/brake_cmd',
    '/hardware_interface/ds_fusion/brake_report',
    '/hardware_interface/ds_fusion/imu/data_raw',
    '/hardware_interface/ds_fusion/steering_cmd',
    '/hardware_interface/ds_fusion/steering_report',
    '/hardware_interface/ds_fusion/throttle_cmd',
    '/hardware_interface/ds_fusion/throttle_report',
    '/hardware_interface/pacmod/as_rx/accel_cmd',
    '/hardware_interface/pacmod/as_rx/brake_cmd',
    '/hardware_interface/pacmod/as_rx/steer_cmd',
    '/hardware_interface/pacmod/parsed_tx/accel_rpt',
    '/hardware_interface/pacmod/parsed_tx/brake_rpt',
    '/hardware_interface/pacmod/parsed_tx/steer_rpt',
    '/hardware_interface/pacmod/parsed_tx/vehicle_speed_rpt',
    '/hardware_interface/pacmod/parsed_tx/yaw_rate_rpt',
    '/hardware_interface/accelerator_pedal_cmd',
    '/hardware_interface/accelerator_pedal_report',
    '/hardware_interface/brake_2_report',
    '/hardware_interface/brake_cmd',
    '/hardware_interface/imu/data_raw',
    '/hardware_interface/misc_report',
    '/hardware_interface/steering_2_report',
    '/hardware_interface/steering_cmd',
    '/guidance/state',
    '/hardware_interface/arbitrated_speed_commands',
    '/hardware_interface/corrimudata',
    '/hardware_interface/imu_raw',
    '/hardware_interface/velocity_accel_cov',
    '/localization/current_pose',
    '/guidance/route_state',
    '/environment/active_geofence',
    '/guidance/state',
    '/guidance/route',
    '/message/incoming_geofence_control',
    '/message/incoming_mobility_operation',
    '/hardware_interface/vehicle_status',
    '/guidance/plan_trajectory',
    '/hardware_interface/steering_wheel'
]

def parse_topics(bag_obj, dst_dir, topic_name):
    file_topic_name = topic_name.replace('/', '_')
    file_topic_name = file_topic_name[1:] # remove the leading underscore character from name
    filename = f'{dst_dir}/{file_topic_name}.csv'

    df_out = []
    first_iteration = True # allows header row
    for _, msg, t in bag_obj.read_messages(topic_name):
        msg_string = str(msg)
        msg_list = msg_string.split('\n')

        instantaneous_data_list = []
        for name_value_pair in msg_list:
            split_pair = name_value_pair.split(':')
            for i in range(len(split_pair)): # should be 0 to 1
                split_pair[i] = split_pair[i].strip()
            instantaneous_data_list.append(split_pair)

        # write the first row from the first element of each pair
        if first_iteration: # header
            headers = ["rosbagTimestamp"] # first column header
            for pair in instantaneous_data_list:
                headers.append(pair[0])

            df_out.append(headers)
            first_iteration = False
        # write the value from each pair to the file
        values = [str(t)] #first column will have rosbag timestamp
        for pair in instantaneous_data_list:
            if len(pair) > 1:
                values.append(pair[1])

        df_out.append(values)

    with fs.open(f'{DST_BUCKET}/{filename}', 'w') as fw:
        writer = csv.writer(fw)
        writer.writerows(df_out)

def process_bags(src_file):
    # create the destination directory name from source directory
    src_file_splitter = src_file.split('/')
    dst_dir = f'csvfiles/{"/".join(src_file_splitter[1:-1])}/{src_file_splitter[-1]}'
    dst_dir = dst_dir.split(".bag")[0]
    
    s3_client.put_object(Bucket=DST_BUCKET, Key=(dst_dir+'/'))
    print('Process started...')
    
    f = fs.open(f'{SRC_BUCKET}/{src_file}')

    bag = rosbag.Bag(f, mode='r')
    bag_types_topics = bag.get_type_and_topic_info()
    list_of_topics = list(bag_types_topics[1].keys())
    
    print(f'Total topics = {len(list_of_topics)}')

    n_cpu = multiprocessing.cpu_count()
    pool = Pool(processes=n_cpu)
    func = partial(parse_topics, bag, dst_dir)
    pool.map(func, list_of_topics)
    pool.close()
    pool.join()

    f.close()

if __name__ == '__main__':
    if PARSE_SPECIFIC_BAGFILE:
        src_files_to_process = BAGFILE_ID
    else:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=SRC_BUCKET)

        src_files_to_process = []
        for page in pages:
            files_on_page = [obj['Key'] for obj in page['Contents']]
            src_files_to_process = src_files_to_process + files_on_page

        if PARSE_SPECIFIC_VEHICLE:
            src_files_to_process = [file for file in src_files_to_process if FILE_IDS[0] in file and FILE_IDS[1] in file and VEHICLE_ID in file]
        else:
            src_files_to_process = [file for file in src_files_to_process if FILE_IDS[0] in file and FILE_IDS[1] in file]

    total_files = len(src_files_to_process)
    print(f'Total bagfiles to process: {total_files}')

    # TODO: Parallelize the processing of multiple bagfiles in same flow
    # jobs = []
    # for idx, src_file in enumerate(src_files_to_process):
    #     print(f'Processing file {idx+1} of {total_files}: {src_file.split("/")[-1]}')
    #     print(src_file)
    # #     p = Process(target=process_bags, args=(src_file))
    # #     jobs.append(p)
    # #     p.start()
    # # p.join()

    for src_file in src_files_to_process:
        start_time = dt.datetime.now()
    # create the destination directory name from source directory
        src_file_splitter = src_file.split('/')
        dst_dir = f'csvfiles/{"/".join(src_file_splitter[1:-1])}/{src_file_splitter[-1]}'
        dst_dir = dst_dir.split(".bag")[0]
        
        s3_client.put_object(Bucket=DST_BUCKET, Key=(dst_dir+'/'))
        print('Process started...')
        
        f = fs.open(f'{SRC_BUCKET}/{src_file}')

        bag = rosbag.Bag(f, mode='r')

        if PARSE_SPECIFIC_TOPICS:
            list_of_topics = TOPICS_ID
        else:
            bag_types_topics = bag.get_type_and_topic_info()
            list_of_topics = list(bag_types_topics[1].keys())
        
        print(f'Total topics = {len(list_of_topics)}')

        # Parallelize Topic processing within the same bagfile
        n_processes = multiprocessing.cpu_count()
        pool = Pool(processes=n_processes)
        func = partial(parse_topics, bag, dst_dir)
        pool.map(func, list_of_topics)
        pool.close()
        pool.join()

        f.close()
        print(f'Time to process {src_file}: {dt.datetime.now() - start_time}')
