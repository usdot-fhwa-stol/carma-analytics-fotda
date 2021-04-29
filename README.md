# CARMA Analytics
CARMA Analytics is a secure, scalable and cloud agnostic data architecture to enable research analysis on big data sets by defining repeatable and scalable processes for ingesting data, creation of a data repository to fuse, process and perform quality assurance.  The architecture also identifies the tools required for data analysis for the research teams.  This architecture addresses the following:

1. Data ingest and transfer:
- Data ingested using a direct connection from an on-prem sandbox (ex. Isilon at TFHRC's STOL lab)
- Data ingested from a remote test environment (ex. using AWS snowball when a direct connection is not available)
- Moving or accessing data from other static and real-time sources to support basic test/dev activities


2. Data storage and fusion:
- Storing and fusing of dat
- Quality assurance
- Transforming and processing data from disparate static and real-time sources into a format that supports research analysis


3. Developer tools and data analytics:
- Provide tools for basic quality assurance and analysis
List of currently available tools:


| S.No. | Tool/Application |
| ----------- | ----------- |
| 1 | Python 3.8 |
| 2 | aws cli |
| 3 | Visual Studio Code |
| 4 | Libre Office |
| 5 | RStudio |
| 6 | R |
| 7 | ROS noetic (Robot Operating System) - Only on Windows EC2 VMs |
| 8 | Jupyter Notebook |
| 9 | Sublime Text |
| 10 | Google Chrome |


Terminal Services are installed in the Windows EC2 VMs to allow regular DOT users to RDP to the desktop. Currently, the following AWS roles are supported:
- Application Admins: This is restricted to devlopers and administrators on the AWS environment. This role has root access to EC2 VMs.

- Data Scientist: This role is primarily for technical users - researchers/analysts who can RDP into the EC2 VMs and access data stored in S3 buckets or on Redshift databases. However, non-technical users can use this role to access AWS Quicksight service which allows them to access Redshift databases and create visualization dashboards.



[Architecture Demo](https://usdot-carma.atlassian.net/wiki/spaces/CRMALN/pages/1380024341/CARMA+Analytics+-+Demo+Video)


## Architecture Diagram
![Architecture diagram](https://github.com/usdot-fhwa-stol/carma-analytics-fotda/blob/main/reference_docs/Architecture_Diagram.PNG)
