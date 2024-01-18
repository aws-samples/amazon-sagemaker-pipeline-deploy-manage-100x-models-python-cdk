import json
import boto3
import datetime

region = boto3.Session().region_name
aws_account_id = boto3.client('sts').get_caller_identity()["Account"]
sagemaker_client = boto3.client('sagemaker', region_name=region)
print(f'boto3 version: {boto3.__version__}')
print(f'region: {region}')


## Get model package version
def get_model_pacakage_version(event, model_package_group_name):
    try:
        model_package_version = event['model_package_version']
    except:
        response = sagemaker_client.list_model_packages(ModelPackageGroupName=model_package_group_name)
        model_package_version = response['ModelPackageSummaryList'][0]['ModelPackageVersion']
    return model_package_version

## Get model name
def get_model_name(event, model_package_group_name,model_package_version):
    try:
        model_name = event['model_name']
    except:
        model_name = f'{model_package_group_name}-model-{model_package_version}'
    return model_name

## Delete model if exists
def delete_model(model_name):
    try: 
        response = sagemaker_client.delete_model(ModelName=model_name)
    except:
        response ="All good, there was no model to delete"
    return response

def create_model(model_name, container_list, role_arn):
    try:
        response = sagemaker_client.create_model(ModelName=model_name,
                            Containers=container_list,
                            ExecutionRoleArn=role_arn)
    except:
        response = "Model failed to create"
    return response



############ Delete existing endpoints #################
def delete_existing_endpoints(partial_endpoint_name):
        # partial_endpoint_name = partial_endpoint_name.replace("_", "-")
        config_list = list_endpoint_configs(partial_endpoint_name)
        endpoint_list = list_endpoints(partial_endpoint_name)
        if len(config_list)>0:
            delete_previous_models(config_list)
            delete_endpoint_config(config_list)
        if len(endpoint_list)>0:
            delete_endpoints(endpoint_list)

def delete_previous_models(config_list):
    for config in config_list:
        endpoint_config_name = config["EndpointConfigName"]
        response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        model_name = response['ProductionVariants'][0]['ModelName']
        delete_model(model_name)
    print("{} Previous Models has been deleted successfully".format(len(config_list)))
      
def list_endpoint_configs(endpoint_name):
    response = sagemaker_client.list_endpoint_configs(
        NameContains=endpoint_name,
        MaxResults=100
    )
    return response["EndpointConfigs"]

def delete_endpoint_config(config_list):
    print("Total number of endpoint configs going to be deleted: {}\n".format(len(config_list)))
    for config in config_list:
        endpoint_config_name = config["EndpointConfigName"]
        print(endpoint_config_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print("{} Endpoint configs deleted successfully".format(len(config_list)))
        
def list_endpoints(endpoint_name):
    response = sagemaker_client.list_endpoints(
        NameContains=endpoint_name,
        StatusEquals='InService',
        MaxResults=100
    )
    return response["Endpoints"]

def delete_endpoints(endpoint_list):
    print("Total number of endpoint going to be deleted: {}\n".format(len(endpoint_list)))
    for endpoint in endpoint_list:
        endpoint_name = endpoint["EndpointName"]
        print(endpoint_name)
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print("{} Endpoints deleted successfully".format(len(endpoint_list)))


def generate_endpoint_name(dimension):
    current_datetime = datetime.datetime.now()
    # Format the date and time as yyyy-mm-dd-hh-mm
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M")
    endpoint_name = "SM-{}-{}".format(dimension, formatted_datetime)
    endpoint_name = endpoint_name.replace("_", "-")
    return endpoint_name


##### Serverless Endpoint #####
def create_serverless_endpoint(model_name, endpoint_name):
    sl_endpoint_name = f'{endpoint_name}'
    sl_endpoint_config_name = f'{sl_endpoint_name}-config'

    config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=sl_endpoint_config_name,
            ProductionVariants=[
                {
                    'ModelName': model_name,
                    'VariantName': 'AllTraffic',
                    "ServerlessConfig": {
                            "MemorySizeInMB": 6144,
                            "MaxConcurrency": 1,
                            },
                    }
            ],
        )
    endpoint_response = sagemaker_client.create_endpoint(
                            EndpointName=sl_endpoint_name, 
                            EndpointConfigName=sl_endpoint_config_name
                        )
    return endpoint_response


def create_realtime_endpoint(model_name, endpoint_name, model_package_version_arn):
    endpoint_name = f'{endpoint_name}'
    endpoint_config_name = f'{endpoint_name}-config'

    model_package_details = sagemaker_client.describe_model_package(
                                        ModelPackageName=model_package_version_arn)
    
    realtime_inference_instance_types = model_package_details['InferenceSpecification']['SupportedRealtimeInferenceInstanceTypes']

    config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'InstanceType': realtime_inference_instance_types[0],
                    'InitialVariantWeight': 1,
                    'InitialInstanceCount': 1,
                    'ModelName': model_name,
                    'VariantName': 'AllTraffic',
                }
            ],
        )
    endpoint_response = sagemaker_client.create_endpoint(
                            EndpointName=endpoint_name, 
                            EndpointConfigName=endpoint_config_name)
    return endpoint_response
    
    