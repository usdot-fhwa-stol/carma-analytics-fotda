# AWS Account Infrastructure Setup

Step 1 : Create s3 bucket manually to facilitate package and deployment of CF stacks
  Bucket Name - <s3_bucket_name>

Step 2 : Package code and push to S3
  aws cloudformation package --template-file networkInfra-params.yml --s3-bucket <s3_bucket_name> --output-template-file networkpackaged.yml

Step 3 : Deploy code
  aws cloudformation deploy --template-file networkpackaged.yml --stack-name FHWA-FOTDA-Prod-Infrastructure --parameter-overrides CfnBucketName=<s3_bucket_name>