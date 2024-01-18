import boto3
import json
import csv
import io
from decimal import Decimal
from botocore.config import Config

# This is the overcome the endpoint Throttling exception
config = Config(
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'
        }
    )

sqs = boto3.client("sqs")
dynamodb_client = boto3.client('dynamodb')
sagemaker_client = boto3.client('sagemaker')
dynamodb_resource = boto3.resource('dynamodb')
sagemaker_runtime_client  = boto3.client("runtime.sagemaker")

# get items from the dynamodb table based on the attribute value
def get_items_from_dynamodb_table(table_name, partition_key, attribute_value):
    table = dynamodb_resource.Table(table_name)
    response = table.get_item(
        Key={
            partition_key: attribute_value
        }
    )
    return response

def convert_floats_to_decimals(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    return obj

# write a funciton which write the item to the dynamodb table
def write_single_item_to_dynamodb(item, table_name):
    table = dynamodb_resource.Table(table_name)
    response = table.put_item(Item=item)
    return response

def get_desired_endpoint_name(endpoint_name):
    response = sagemaker_client.list_endpoints(
        NameContains=endpoint_name,
        StatusEquals='InService',
        MaxResults=100
    )
    if len(response["Endpoints"]) == 0:
        return "Endpoint not found"
    elif len(response["Endpoints"]) > 1:
        print("More than one endpoint is found for the combination: {}".format(endpoint_name))

    return response["Endpoints"][0]["EndpointName"]



def convert_list_to_csv_string(csv_list):
    # Convert list to CSV string
    csv_string = io.StringIO()
    csv_writer = csv.writer(csv_string)
    csv_writer.writerow(csv_list)

    csv_string = csv_string.getvalue()
    return csv_string.encode("utf-8")

def query_endpoint(payload, endpoint_name):
    content_type = "text/csv"
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType=content_type, 
        Body=payload
    )
    return response

def parse_resonse(query_response):
    prediction = json.loads(query_response["Body"].read())
    return prediction
