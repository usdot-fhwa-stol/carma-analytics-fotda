# title           : bagfiles_transfer.py
# description     : Python Script to transfer bagfile data from IS300 servers at STOL 
#                   to AWS S3 on FOTDA cloud.
#                   This is scheduled on EC2 instance as a cron job.
# author          : Ankur Tyagi (Leidos)
# license         : MIT license
# ===================================================================================

import os
import sys
from datetime import datetime as dt
import boto3
import botocore

# Define upload paramaters - Change these parameters to the corresponding S3 bucket name and testing date
local_path = '../Share_data/CARMA_Analytics/'
bucket_name = 'raw-carma-core-validation'
filepath_id = '20210415' # date of data transfer

available_files = []
files_to_transfer = []


def get_file_paths(path):
    for obj in os.listdir(path):
        obj_path = os.path.join(path, obj)
        if os.path.isfile(obj_path):
            available_files.append(obj_path)
        else:
            get_file_paths(obj_path)
    
    return available_files

def file_exists(s3_session_obj, key):
    try:
        s3_session_obj.Object(bucket_name, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            print('Error reading s3 object head.')
    else:
        return True

def upload_new_files_to_s3(files_to_transfer):
    aws_session = boto3.session.Session(profile_name='saml')
    s3 = aws_session.resource('s3')
    count = 0
    total_files = len(files_to_transfer)
    
    for file_path in files_to_transfer:
        upload_datetime = dt.fromtimestamp(os.path.getmtime(file_path))
        current_datetime = dt.today()
        
        s3_filepath = file_path.replace('/logs','') # remove the 'logs' folder name to match directory structure on S3
        
        if (current_datetime-upload_datetime).total_seconds() < 86401: ## check if a new file is added since yesterday
            dir_structure_on_s3 = f'bagfiles/{s3_filepath[len(local_path):]}'
            path_split = dir_structure_on_s3.split('/')
            filename = path_split[len(path_split)-1]
            
            if not file_exists(s3, dir_structure_on_s3):
                s3.meta.client.upload_file(file_path,
                                           bucket_name,
                                           dir_structure_on_s3)
            else:
                print(f'{dir_structure_on_s3.split("/")[-1]} already exists on s3.')
            
            count += 1    
        print(f'{count}/{total_files} files transferred.')
    
    return f'Transfer complete. Total files transferred to S3 = {count}'

if __name__ == '__main__':
    available_files = get_file_paths(local_path)
    files_to_file = [transfer for file in available_files if (file[-4:] == '.bag' and filepath_id in file)] # only select the files with .bag suffix

    s3_upload_status = upload_new_files_to_s3(files_to_transfer)
    print(s3_upload_status)

