from aws_cdk import (
    Fn,
    RemovalPolicy,
    Stack,
    aws_sqs as sqs,
    aws_s3 as s3,
    aws_iam as iam,
    aws_s3_notifications,
    aws_lambda_event_sources as lambda_event_sources,
    aws_iam as iam,
    Duration,
    aws_lambda as _lambda,
    aws_s3 as s3,
    RemovalPolicy,
    aws_dynamodb as dynamodb,
    aws_lambda_python_alpha as _alambda,
)
import json
from cdk_nag import NagSuppressions
from constructs import Construct

class InferenceResultsStack(Stack):
    # Standard definition for CDK stack
    def __init__(self, scope: Construct, construct_id: str,  **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        stack_name = Stack.of(self).stack_name.lower()
        file = open("project_config.json")
        variables = json.load(file)
        sm_pipeline_name = variables["SageMakerPipelineName"]
        USE_AMT = variables["USE_AMT"]

        #import value from another stack
        access_log_bucket_arn = Fn.import_value("accesslogbucketarn")
        access_logs_bucket = s3.Bucket.from_bucket_arn(self, "AccessLogsBucket", access_log_bucket_arn)

        ## training data s3 bucket
        inference_bucket_s3  = s3.Bucket(
            self,
            "InferenceBucket",
            removal_policy= RemovalPolicy.DESTROY,
            block_public_access= s3.BlockPublicAccess.BLOCK_ALL,
            encryption= s3.BucketEncryption.S3_MANAGED,
            server_access_logs_bucket=access_logs_bucket,
            enforce_ssl=True,
            auto_delete_objects=True,
        )
        ## Dead Letter Queue
        dlq = sqs.Queue(
            self,
            id="dead_letter_queue_id",
            retention_period=Duration.days(7),
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )
        dead_letter_queue = sqs.DeadLetterQueue(
            max_receive_count=2,
            queue=dlq,
        )
        # Create SQS Queue
        record_queue = sqs.Queue(
            self,
            "RecordQueue",
            receive_message_wait_time=Duration.seconds(10), #Time that the poller waits for new messages before returning a response
            visibility_timeout = Duration.seconds(540),  # This should be bingger than Lambda time out
            dead_letter_queue=dead_letter_queue,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,            
        )

        ## DynamoDB table to store input data in key-value pair
        inference_ddb_table = dynamodb.Table(self, "inference_table",
                       partition_key=dynamodb.Attribute(name= "id", type=dynamodb.AttributeType.STRING),
                       billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                       encryption=dynamodb.TableEncryption.AWS_MANAGED,
                       point_in_time_recovery=True,
                       removal_policy= RemovalPolicy.DESTROY
                )

        ## Pandas layer
        pandas_lambda_layer = _alambda.PythonLayerVersion(self, 'pandas-layer',
            entry = './sagemaker_pipeline_deploy_manage_n_models_cdk/lambda/lambda_layer/pandas_layer/',
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
            compatible_architectures=[_lambda.Architecture.ARM_64],
        )
        
        # Lambda 1: Read CSV file and send messages to SQS
        send_messages_to_sqs_lambda = _lambda.Function(
            self,
            "SendMessagesToSQS",
            handler="index.lambda_handler",
            code=_lambda.Code.from_asset('./sagemaker_pipeline_deploy_manage_n_models_cdk/lambda/send_messages_sqs/'),  
            runtime=_lambda.Runtime.PYTHON_3_11,
            architecture=_lambda.Architecture.ARM_64,
            layers=[
                pandas_lambda_layer
            ],
            environment={
                "queue_url": record_queue.queue_url,
            },
            timeout=Duration.seconds(90),
        )
        inference_bucket_s3.grant_read_write(send_messages_to_sqs_lambda)
        record_queue.grant_send_messages(send_messages_to_sqs_lambda)

        # Invoke send_messages_to_sqs_lambda when a new object is CREATED to the Inference bucket        
        inference_bucket_s3.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            aws_s3_notifications.LambdaDestination(send_messages_to_sqs_lambda),
            s3.NotificationKeyFilter(
                                    prefix="",
                                    suffix=".csv",
                                ),
        )

        # Lambda 2: Process messages from SQS
        consume_messages_from_sqs_lambda = _lambda.Function(
            self,
            "ConsumeMessagesFromSQS",
            handler="index.lambda_handler",
            code=_lambda.Code.from_asset('./sagemaker_pipeline_deploy_manage_n_models_cdk/lambda/consume_messages_sqs/'),
            runtime=_lambda.Runtime.PYTHON_3_11,
            architecture=_lambda.Architecture.ARM_64,
            layers=[
                pandas_lambda_layer
            ],
            environment={
                "inference_ddb_table": inference_ddb_table.table_name,
                "partition_key": "id",
            },
            timeout=Duration.seconds(90),
        )
        record_queue.grant_consume_messages(consume_messages_from_sqs_lambda)
        inference_ddb_table.grant_read_write_data(consume_messages_from_sqs_lambda)
        consume_messages_from_sqs_lambda.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        
        ## This event will be triggered by SQS when a new message is received
        invoke_event_source = lambda_event_sources.SqsEventSource(record_queue, batch_size=1)
        consume_messages_from_sqs_lambda.add_event_source(invoke_event_source)
        
        ## CDK Nag Suppression
        NagSuppressions.add_resource_suppressions([send_messages_to_sqs_lambda.role],
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "This code is for demo purposes. So granted access to all indices of S3 bucket.",
                                            }
                                        ],
                            apply_to_children=True)

        ## CDK Nag Suppression
        NagSuppressions.add_resource_suppressions([consume_messages_from_sqs_lambda.role,],
                            suppressions=[{
                                            "id": "AwsSolutions-IAM4",
                                            "reason": "Allowing AmazonSageMakerFullAccess as it is sample code, for production usecase scope down the permission",
                                            }
                                        ],
                            apply_to_children=True)
        
        # CDK NAG suppression
        NagSuppressions.add_stack_suppressions(self, [
                                            {
                                                "id": 'AwsSolutions-IAM4',
                                                "reason": 'Lambda execution policy for custom resources created by higher level CDK constructs',
                                                "appliesTo": [
                                                        'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                                                    ],
                                            }])
        # CDK NAG suppression
        NagSuppressions.add_resource_suppressions_by_path(            
            self,
            path="/InferenceResultsStack/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role/DefaultPolicy/Resource",
            suppressions = [
                            { "id": 'AwsSolutions-IAM5', "reason": 'CDK BucketNotificationsHandler L1 Construct' },
                        ],
            apply_to_children=True
        )