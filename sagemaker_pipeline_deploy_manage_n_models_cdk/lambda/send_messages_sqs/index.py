import os
from helper import *

def lambda_handler(event, context):
    print(event)
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]
        queue_url = os.environ['queue_url']
        
    except Exception as e:
        print("we are in dev mode")
        bucket = "inferenceresultsstack-inferencebucket95585283-u2lbrl0oaeqx"
        key = "inference-dataset/DummyDim1/dim_1_test.csv"
        
        queue_url = "https://sqs.eu-west-1.amazonaws.com/123456789012/InferenceResultsStack-RecordQueueCE027E6F-9rEpD04Gk0Ou"

    desired_dim = key.split("/")[-2]

    df = read_csv_from_s3(bucket, key)
    df["desired_dim"] = desired_dim   ## this will help to indentify which endpoint needs to be called
    df = df.head(5)  ## for testing only

    send_message_to_sqs(queue_url, df)
    destination_key =  f"processed/{key}.out"
    #move file to processed folder
    move_file_from_source_to_destination_bucket(source_bucket=bucket,
                                            source_key=key,
                                            destination_bucket=bucket,
                                            destination_key=destination_key)

    return {
        'statusCode': 200
    }

if __name__ == "__main__":
   lambda_handler(None, None)