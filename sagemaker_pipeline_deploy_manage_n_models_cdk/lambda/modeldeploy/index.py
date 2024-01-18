import json
import boto3
import datetime
from helper import *

region = boto3.Session().region_name
aws_account_id = boto3.client('sts').get_caller_identity()["Account"]
sagemaker_client = boto3.client('sagemaker', region_name=region)
print(f'boto3 version: {boto3.__version__}')
print(f'region: {region}')


def lambda_handler(event, context):

    print(f"event:: {event}")

    state_dim = event['state_dim']
    model_package_group_name = event['model_package_group_name']
    role_arn = event['role_arn']
    
    model_package_version = get_model_pacakage_version(event, model_package_group_name)
    
    model_name = get_model_name(event, model_package_group_name,model_package_version)

    model_package_version_arn = f'arn:aws:sagemaker:{region}:{aws_account_id}:model-package/{model_package_group_name}/{model_package_version}'
    
    container_list = [{'ModelPackageName': model_package_version_arn}]
    
    # Approve model package to be used as SageMaker model and for deployment
    sagemaker_client.update_model_package(ModelPackageArn=model_package_version_arn,
                                   ModelApprovalStatus='Approved')
    

    delete_model_res = delete_model(model_name)
    
    create_model_res = create_model(model_name, container_list, role_arn)
    
    partial_endpoint_name = f'SM-{state_dim}'
    delete_existing_endpoints(partial_endpoint_name)
     
    endpoint_name = generate_endpoint_name(state_dim)
    
    # If you want to deploy serverlss inference
    # sl_endpoint_res = create_serverless_endpoint(model_name, endpoint_name)
    # print(sl_endpoint_res)
    # If you want to deploy realtime inference
    realtime_endpoint_res = create_realtime_endpoint(model_name, endpoint_name, model_package_version_arn)
    print(realtime_endpoint_res)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Created Endpoint!')
    }


if __name__ == "__main__":
    event = {'role_arn': 'arn:aws:iam::123456789012:role/SMPipelineStack-ProjectRoleFAA285A5-nRnXHTFTy2gH', 
             'state_dim': 'state-01', 
             'model_package_group_name': 'mpg-state-01'}

    lambda_handler(event, None)
