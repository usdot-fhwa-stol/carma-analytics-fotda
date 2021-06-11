import rosbag #pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
import boto3 #pip install boto3
from urllib.parse import unquote_plus
import s3fs
import csv


s3_client = boto3.client('s3')
fs = s3fs.S3FileSystem()

SRC_BUCKET = 'raw-carma-core-validation'
DST_BUCKET = 'preprocessed-carma-core-validation'

USE_FILE_IDS = True
FILE_IDS = ['down-selected', '20210415'] # this can be an list of indetifiers for the folder in which raw data is stored - like, "20210318" and "down-selected"

## list of specific filename to exclude; leave the list blank if all files need to be processed
# exclusion_list = []
exclusion_list = ['_2021-04-06-17-51-07_down-selected.bag',
                  '_2021-04-08-19-32-52_down-selected.bag',
                  '_2021-04-08-19-40-19_down-selected.bag',
                  '_2021-04-08-19-44-22_down-selected.bag']

def process_bags(src_file):
    # create the destination directory name from source directory
    src_file_splitter = src_file.split('/')
    dst_dir = f'csvfiles/{"/".join(src_file_splitter[1:-1])}/{src_file_splitter[-1]}'
    dst_dir = dst_dir.split(".bag")[0]
    
    s3_client.put_object(Bucket=DST_BUCKET, Key=(dst_dir+'/'))
    print('Process started...')
    
    with fs.open(f'{SRC_BUCKET}/{src_file}') as f:
        bag = rosbag.Bag(f, mode='r')
        
        bag_types_topics = bag.get_type_and_topic_info()
        list_of_topics = list(bag_types_topics[1].keys())
        print(f'Total topics = {len(list_of_topics)}')
        counter = 0
        
        for topic_name in list_of_topics:
            # create a new CSV file for each topic
            file_topic_name = topic_name.replace('/', '_')
            file_topic_name = file_topic_name[1:] # remove the leading underscore character from name
            filename = f'{dst_dir}/{file_topic_name}.csv'
    
            df_out = []
            first_iteration = True # allows header row
            for subtopic, msg, t in bag.read_messages(topic_name):
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

            with fs.open(f'{DST_BUCKET}/{filename}', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(df_out)
            counter += 1
            print(f'{counter}/{len(list_of_topics)} topics complete: {topic_name}')
        print('all topics are complete.')
        bag.close()


if __name__ == '__main__':
	src_bucket_files = [key['Key'] for key in s3_client.list_objects(Bucket=SRC_BUCKET)['Contents']]

	if USE_FILE_IDS:
	    src_files_to_process = [file for file in src_bucket_files if FILE_IDS[0] in file and FILE_IDS[1] in file]
	else:
	    src_files_to_process = src_bucket_files
	
	if len(exclusion_list) > 0:
		for filename in exclusion_list:
		    src_files_to_process = [x for x in src_files_to_process if filename not in x]

	total_files = len(src_files_to_process)
	for idx, src_file in enumerate(src_files_to_process):
	    print(f'Processing file {idx+1} of {total_files}: {src_file.split("/")[-1]}')
	    process_bags(src_file)