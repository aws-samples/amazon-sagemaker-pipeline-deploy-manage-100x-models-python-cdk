import os
from helper import *

def lambda_handler(event, context):
    print(event)
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]
        sm_pipeline_name = os.environ["sm_pipeline_name"]

    except:
        print("We are using dev environment")

        bucket="smpipelineexecutionstack-trainingbucketeb7bb5c9-kdxthutnl8ps"
        key="training-dataset/DummyDim3/churn_dummy_dim3.csv"
        sm_pipeline_name = "model-train-deploy-pipeline"

   
    input_data_uri = f's3://{bucket}/{key}'
    print(input_data_uri)
    state_dim = key.split("/")[-2]

    model_package_group_name = f"{state_dim}-model-registry"
    print(model_package_group_name)

    pipeline_name= get_pipeline_name(pipeline_name_prefix=sm_pipeline_name)
    print(pipeline_name)

    approved_packages = get_approved_packages(model_package_group_name)
    print("Number of approved packages found: ", len(approved_packages))

    execution = start_pipeline(pipeline_name, input_data_uri, state_dim, model_package_group_name, approved_packages)
    print(execution)

    return {
        'statusCode': 200,
    }

if __name__=="__main__":
    lambda_handler(None, None)