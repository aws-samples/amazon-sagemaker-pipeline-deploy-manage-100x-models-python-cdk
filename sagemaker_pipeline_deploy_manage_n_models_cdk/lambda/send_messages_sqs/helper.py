from decimal import Decimal
import json
import time
import boto3
import pandas as pd

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
sqs = boto3.client("sqs")

def read_csv_from_s3(bucket, key,):
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)  
    df = pd.read_csv(s3_obj['Body'])
    print(df.shape)
    print("CSV data read successfully")
    return df

def send_message_to_sqs(queue_url, df):
    for index, row in df.iterrows():
        message_body = row.to_json() 
        response = sqs.send_message(
            QueueUrl=queue_url,
            DelaySeconds=10,
            MessageBody=(message_body)
        )
    print("Total {} messages sent to SQS".format(len(df)))

def move_file_from_source_to_destination_bucket(source_bucket, source_key, destination_bucket, destination_key):
    # copy the object from stagging to input bucket
    s3_resource.Object(destination_bucket, destination_key).copy_from(CopySource={'Bucket': source_bucket, 'Key': source_key})
    print("Copied the object from stagging to input bucket successfully")
    # delete the object form stagging bucket
    s3_resource.Object(source_bucket, source_key).delete()
    print("Deleted the object form stagging bucket successfully")
    