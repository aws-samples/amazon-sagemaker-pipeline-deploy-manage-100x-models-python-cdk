
import boto3
import json
sagemaker_client = boto3.client('sagemaker')

def get_approved_packages(model_package_group_name):
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved',
        MaxResults=100
    )
    approved_packages = response['ModelPackageSummaryList']
    return approved_packages 

def get_pipeline_name(pipeline_name_prefix="model-train-deploy-pipeline"):
    response = sagemaker_client.list_pipelines(
        PipelineNamePrefix=pipeline_name_prefix
    )
    pipeline_name = response['PipelineSummaries'][0]["PipelineName"]
    return pipeline_name

def start_pipeline(pipeline_name, input_data_uri, state_dim, model_package_group_name, approved_packages):
        if len(approved_packages) == 0:
            print("No approved packages found, so running execution for first time ")
            response = sagemaker_client.start_pipeline_execution(
                                    PipelineName=pipeline_name,
                                    PipelineParameters=[
                                            {"Name" : "InputDataUri", "Value" : input_data_uri},
                                            {"Name" : "StateDim", "Value" : state_dim},
                                            {"Name" : "ModelPackageGroupName", "Value" : model_package_group_name},
                                            ### first run
                                            #For data
                                            {"Name" : "SkipDataQualityCheck", "Value" : "True"},
                                            {"Name" : "RegisterNewDataQualityBaseline", "Value" : "True"},
                                            {"Name" : "SkipDataBiasCheck", "Value" : "True"},
                                            {"Name" : "RegisterNewDataBiasBaseline", "Value" : "True"},
                                            #For Model
                                            {"Name" : "SkipModelQualityCheck", "Value" : "True"},
                                            {"Name" : "RegisterNewModelQualityBaseline", "Value" : "True"},
                                            {"Name" : "SkipModelBiasCheck", "Value" : "True"},
                                            {"Name" : "RegisterNewModelBiasBaseline", "Value" : "True"},
                                            {"Name" : "SkipModelExplainabilityCheck", "Value" : "True"},
                                            {"Name" : "RegisterNewModelExplainabilityBaseline", "Value" : "True"},
                                    ]
                                    )
        else:
            print("Already approved packages has been found")
            response = sagemaker_client.start_pipeline_execution(
                                        PipelineName=pipeline_name,
                                        PipelineParameters=[
                                                {"Name" : "InputDataUri", "Value" : input_data_uri},
                                                {"Name" : "StateDim", "Value" : state_dim},
                                                {"Name" : "ModelPackageGroupName", "Value" : model_package_group_name},
                                        ]
                                    )

        return response
