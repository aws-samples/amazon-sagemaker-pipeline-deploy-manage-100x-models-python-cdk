import json
import cdk_nag
from aws_cdk import Aspects
import aws_cdk as cdk

from sagemaker_pipeline_deploy_manage_n_models_cdk.stacks.sagemaker_studio_setup_stack import SagemakerStudioSetupStack
from sagemaker_pipeline_deploy_manage_n_models_cdk.stacks.sagemaker_pipeline_stack import SagemakerPipelineStack
from sagemaker_pipeline_deploy_manage_n_models_cdk.stacks.start_sagemaker_pipeline_stack import StartSagemakerPipelineStack
from sagemaker_pipeline_deploy_manage_n_models_cdk.stacks.inference_results_stack import InferenceResultsStack

file = open("project_config.json")
variables = json.load(file)
main_stack_name = variables["MainStackName"]
app = cdk.App()

# This Stack will create resources to create SageMaker Studio notebook
sm_studio_stack = SagemakerStudioSetupStack(app, "SMStudioSetupStack")

# This stack will create resources to create SageMaker Pipeline
sm_pipeline_stack = SagemakerPipelineStack(app, "SMPipelineStack")
sm_pipeline_stack.add_dependency(sm_studio_stack)

# This stack will create resources to execute SageMaker Pipeline
start_sm_pipeline_stack = StartSagemakerPipelineStack(app, "SMPipelineExecutionStack")
start_sm_pipeline_stack.add_dependency(sm_pipeline_stack)

# This stack will create resources to get inference results from SageMaker Endpoint
inference_results_stack = InferenceResultsStack(app, "InferenceResultsStack")
inference_results_stack.add_dependency(start_sm_pipeline_stack)

Aspects.of(app).add(cdk_nag.AwsSolutionsChecks(reports=True, verbose=True))

app.synth()