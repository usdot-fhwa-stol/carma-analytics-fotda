#!/usr/bin/python3

# title           : getCredentialsUsingSAML.py
# description     : Python Script to generate temporary session keys for using AWS CLI
# author          : DOT - ECS (Enterprise Cloud Services)
# license         : MIT license
# =====================================================================================

import sys
import boto.sts
import boto.s3
import requests
import getpass
import configparser
import base64
import logging
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
from os.path import expanduser
from urllib.parse import urlparse, urlunparse


## Add assertion string here from the saml federated profile
assertion='PHNhbWxwOlJlc3BvbnNlIElEPSJfNDE2MDAxMmEtMjhlNC00MmY5LWI0YjMtY2ZmZWNlY2U2MWM3IiBWZXJzaW9uPSIyLjAiIElzc3VlSW5zdGFudD0iMjAyMC0xMi0xNVQxNjoyODozMi4xNThaIiBEZXN0aW5hdGlvbj0iaHR0cHM6Ly9zaWduaW4uYXdzLmFtYXpvbi5jb20vc2FtbCIgQ29uc2VudD0idXJuOm9hc2lzOm5hbWVzOnRjOlNBTUw6Mi4wOmNvbnNlbnQ6dW5zcGVjaWZpZWQiIHhtbG5zOnNhbWxwPSJ1cm46b2FzaXM6bmFtZXM6dGM6U0FNTDoyLjA6cHJvdG9jb2wiPjxJc3N1ZXIgeG1sbnM9InVybjpvYXNpczpuYW1lczp0YzpTQU1MOjIuMDphc3NlcnRpb24iPmh0dHA6Ly9hZGZzLmRvdC5nb3YvYWRmcy9zZXJ2aWNlcy90cnVzdDwvSXNzdWVyPjxzYW1scDpTdGFQtQXBwbGljYXRpb25BZG1pbixhcm46YXdzOmlhbTo6NTc3MDc2MDE5NDc2OnNhbWwtcHJvdmlkZXIvRE9ULUFERlM8L0F0dHJpYnV0ZVZhbHVlPjwvQXR0cmlidXRlPjwvQXR0cmlidXRlU3RhdGVtZW50PjxBdXRoblN0YXRlbWVudCBBdXRobkluc3RhbnQ9IjIwMjAtMTItMTVUMTY6MjU6MzYuNDc3WiIgU2Vzc2lvbkluZGV4PSJfOWNiNjI1ZjEtMThlYS00YTc2LTkzNGQtZmM1M2QyNzk5YTMyIj48QXV0aG5Db250ZXh0PjxBdXRobkNvbnRleHRDbGFzc1JlZj51cm46ZmVkZXJhdGlvbjphdXRoZW50aWNhdGlvbjp3aW5kb3dzPC9BdXRobkNvbnRleHRDbGFzc1JlZj48L0F1dGhuQ29udGV4dD48L0F1dGhuU3RhdGVtZW50PjwvQXNzZXJ0aW9uPjwvc2FtbHA6UmVzcG9uc2U+'


#Set your home directory
home = '/home/carma'

# region: The default AWS region that this script will connect
# to for all API calls
region = 'us-east-1'

# output format: The AWS CLI output format that will be configured in the
# saml profile (affects subsequent CLI calls)
outputformat = 'json'

# awsconfigfile: The file where this script will store the temp
# credentials under the saml profile
awsconfigfile = '/.aws/credentials'

# SSL certificate verification: Whether or not strict certificate
# verification is done, False should only be used for dev/test
sslverification = True

# idpentryurl: The initial url that starts the authentication process.
idpentryurl = 'https://adfs.dot.gov/adfs/ls/IdpInitiatedSignOn.aspx?loginToRp=urn:amazon:webservices'


if (assertion == ''):
    #TODO: Insert valid error checking/handling
    print('Response did not contain a valid SAML assertion')
    sys.exit(0)

# Debug only
# print(base64.b64decode(assertion))

# Parse the returned assertion and extract the authorized roles 
awsroles = [] 
root = ET.fromstring(base64.b64decode(assertion))
 
for saml2attribute in root.iter('{urn:oasis:names:tc:SAML:2.0:assertion}Attribute'): 
    if (saml2attribute.get('Name') == 'https://aws.amazon.com/SAML/Attributes/Role'): 
        for saml2attributevalue in saml2attribute.iter('{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue'):
            awsroles.append(saml2attributevalue.text)
 
# Note the format of the attribute value should be role_arn,principal_arn 
# but lots of blogs list it as principal_arn,role_arn so let's reverse 
# them if needed 
for awsrole in awsroles: 
    chunks = awsrole.split(',') 
    if'saml-provider' in chunks[0]:
        newawsrole = chunks[1] + ',' + chunks[0] 
        index = awsroles.index(awsrole) 
        awsroles.insert(index, newawsrole) 
        awsroles.remove(awsrole)

# If I have more than one role, ask the user which one they want, 
# otherwise just proceed 
print("") 
if len(awsroles) > 1: 
    i = 0 
    print("Please choose the role you would like to assume:") 
    for awsrole in awsroles: 
        print('[', i, ']: ', awsrole.split(',')[0]) 
        i += 1 
								 
							   

    print("Selection: ", end=' ') 
    selectedroleindex = input() 
 
    # Basic sanity check of input 
    if int(selectedroleindex) > (len(awsroles) - 1): 
        print('You selected an invalid role index, please try again') 
        sys.exit(0) 
 
    role_arn = awsroles[int(selectedroleindex)].split(',')[0] 
    principal_arn = awsroles[int(selectedroleindex)].split(',')[1]
 
else: 
    role_arn = awsroles[0].split(',')[0] 
    principal_arn = awsroles[0].split(',')[1]

# Use the assertion to get an AWS STS token using Assume Role with SAML
# client = boto3.client('sts')
conn = boto.sts.connect_to_region(region)
token = conn.assume_role_with_saml(role_arn, principal_arn, assertion)

# Write the AWS STS token into the AWS credential file
#home = expanduser("~")
#home = "%USERPROFILE%"

filename = home + awsconfigfile
#filename = awsconfigfile
 
# Read in the existing config file
config = configparser.RawConfigParser()
config.read(filename)
 
# Put the credentials into a saml specific section instead of clobbering
# the default credentials
if not config.has_section('saml'):
    config.add_section('saml')
 
config.set('saml', 'output', outputformat)
config.set('saml', 'region', region)
config.set('saml', 'aws_access_key_id', token.credentials.access_key)
config.set('saml', 'aws_secret_access_key', token.credentials.secret_key)
config.set('saml', 'aws_session_token', token.credentials.session_token)
print ("aws_access_key_id", token.credentials.access_key)
print ("token.credentials.secret_key", token.credentials.secret_key)
print ("token.credentials.session_token", token.credentials.session_token)
# Write the updated config file
with open(filename, 'w') as configfile:
     config.write(configfile)
#with open(filename, 'w') as output: 
#     config.write(filename)
#baconFile = open(filename, 'w')
#baconFile.write(config.text)
# Give the user some basic info as to what has just happened
print('\n\n----------------------------------------------------------------')
print('Your new access key pair has been stored in the AWS configuration file {0} under the saml profile.'.format(filename))
print('Note that it will expire at {0}.'.format(token.credentials.expiration))
print('After this time, you may safely rerun this script to refresh your access key pair.')
print('To use this credential, call the AWS CLI with the --profile option (e.g. aws --profile saml ec2 describe-instances).')
print('----------------------------------------------------------------\n\n')

# Use the AWS STS token to list all of the S3 buckets
s3conn = boto.s3.connect_to_region(region,
                     aws_access_key_id=token.credentials.access_key,
                     aws_secret_access_key=token.credentials.secret_key,
                     security_token=token.credentials.session_token)
 
buckets = s3conn.get_all_buckets()
 
print('Simple API example listing all S3 buckets:')
print(buckets)