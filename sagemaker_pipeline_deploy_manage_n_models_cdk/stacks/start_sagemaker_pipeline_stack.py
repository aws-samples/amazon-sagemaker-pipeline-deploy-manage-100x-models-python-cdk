from aws_cdk import (
    Fn,
    RemovalPolicy,
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_s3_notifications,
    aws_iam as iam,
    Duration,
    aws_lambda as _lambda,
    aws_s3 as s3,
    RemovalPolicy,
)
import json
from cdk_nag import NagSuppressions
from constructs import Construct

class StartSagemakerPipelineStack(Stack):
    # Standard definition for CDK stack
    def __init__(self, scope: Construct, construct_id: str,  **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        stack_name = Stack.of(self).stack_name.lower()
        file = open("project_config.json")
        variables = json.load(file)
        sm_pipeline_name = variables["SageMakerPipelineName"]

        #import value from another stack
        access_log_bucket_arn = Fn.import_value("accesslogbucketarn")
        pipeline_project_role_arn = Fn.import_value("pipelineprojectrolearn")
        
        access_logs_bucket = s3.Bucket.from_bucket_arn(self, "AccessLogsBucket", access_log_bucket_arn)
        pipeline_project_role = iam.Role.from_role_arn(self, "Project Role", pipeline_project_role_arn)

        ## training data s3 bucket
        training_bucket_s3  = s3.Bucket(
            self,
            "TrainingBucket",
            removal_policy= RemovalPolicy.DESTROY,
            block_public_access= s3.BlockPublicAccess.BLOCK_ALL,
            encryption= s3.BucketEncryption.S3_MANAGED,
            server_access_logs_bucket=access_logs_bucket,
            enforce_ssl=True,
            auto_delete_objects=True,
        )
        training_bucket_s3.grant_read_write(pipeline_project_role)

        ## start sagemaker pipeline lambda
        start_sm_pipeline_lambda  = _lambda.Function(
            self,
            "start-sm-pipeline-lambda",
            handler='index.lambda_handler',
            code=_lambda.Code.from_asset('./sagemaker_pipeline_deploy_manage_n_models_cdk/lambda/start_sm_pipeline/'),  
            runtime=_lambda.Runtime.PYTHON_3_11,
            architecture=_lambda.Architecture.ARM_64,
            timeout=Duration.seconds(90),
            environment={
                'sm_pipeline_name': sm_pipeline_name,
            },
        )
        #When you are moving to production follow least privilage access. Use custom policies instead of managed access policies
        start_sm_pipeline_lambda.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        training_bucket_s3.grant_read_write(start_sm_pipeline_lambda)
        
        ## trigger start_pipeline_lambda when a new object is uploaded to the training bucket        
        training_bucket_s3.add_event_notification(
                            s3.EventType.OBJECT_CREATED,
                            aws_s3_notifications.LambdaDestination(start_sm_pipeline_lambda),
                            s3.NotificationKeyFilter(
                                    prefix="training-dataset",
                                    suffix=".csv",
                                ),
                        )
        
        ## CDK Nag Suppression
        NagSuppressions.add_resource_suppressions([start_sm_pipeline_lambda.role,],
                            suppressions=[{
                                            "id": "AwsSolutions-IAM4",
                                            "reason": "Allowing AmazonSageMakerFullAccess as it is sample code, for production usecase scope down the permission",
                                            }
                                        ],
                            apply_to_children=True)
        
        NagSuppressions.add_resource_suppressions([start_sm_pipeline_lambda.role, pipeline_project_role],
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "This code is for demo purposes. So granted access to all indices of S3 bucket.",
                                            }
                                        ],
                            apply_to_children=True)
        
        NagSuppressions.add_stack_suppressions(self, [
                                {
                                    "id": 'AwsSolutions-IAM4',
                                    "reason": 'Lambda execution policy for custom resources created by higher level CDK constructs',
                                    "appliesTo": [
                                            'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                                        ],
                                }])

        NagSuppressions.add_resource_suppressions_by_path(            
            self,
            path="/SMPipelineExecutionStack/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role/DefaultPolicy/Resource",
            suppressions = [
                            { "id": 'AwsSolutions-IAM5', "reason": 'CDK BucketNotificationsHandler L1 Construct' },
                        ],
            apply_to_children=True
        )