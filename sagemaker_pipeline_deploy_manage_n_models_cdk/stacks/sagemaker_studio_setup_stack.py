from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_sagemaker as sagemaker,
    CfnOutput,
    aws_lambda as _lambda,
    aws_logs
)
import json
from aws_cdk import aws_lambda_python_alpha as _alambda
from constructs import Construct
from aws_cdk.custom_resources import Provider
import aws_cdk as core
from constructs import Construct
from cdk_nag import NagSuppressions

class SagemakerStudioSetupStack(Stack):
    # Standard definition for CDK stack
    def __init__(self, scope: Construct, construct_id: str,  **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        file = open("project_config.json")
        variables = json.load(file)
        stack_name = Stack.of(self).stack_name.lower()

        # Create Studio Role
        role = iam.Role(
            self,
            "Studio Role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"), #When you are moving to production follow least privilage access. Use custom policies instead of managed access policies
            ],
            inline_policies={
                "CustomRules": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "codewhisperer:GenerateRecommendations*",
                            ],
                            resources=["*"],
                        )
                    ]
                )
            },
        )

        vpc = ec2.Vpc(self, "VPC")

        self.public_subnet_ids = [
            public_subnet.subnet_id for public_subnet in vpc.public_subnets
        ]

        flow_log_group = aws_logs.LogGroup(self, "vpcFlowLogGroup")

        flow_log_role = iam.Role(self, 
                                "vpcFLowLogRole",
                                assumed_by=iam.ServicePrincipal("vpc-flow-logs.amazonaws.com")
                                )

        ec2.FlowLog(self, "FlowLog",
            resource_type=ec2.FlowLogResourceType.from_vpc(vpc),
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(flow_log_group, flow_log_role)
        )

        # Create domain with IAM auth, role created above, VPC created above and subnets created above
        domain = sagemaker.CfnDomain(
            self,
            "Domain",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=role.role_arn,
            ),
            domain_name=f"Studio-{stack_name}",
            subnet_ids=self.public_subnet_ids,
            vpc_id=vpc.vpc_id,
        )
        
        #Create the Custom Resource to enable sagemaker projects for the different personas
        enable_sm_project_lambda = _alambda.PythonFunction(
            self,
            "sg-project-function",
            runtime=_lambda.Runtime.PYTHON_3_11,
            architecture=_lambda.Architecture.ARM_64,
            entry="./sagemaker_pipeline_deploy_manage_n_models_cdk/lambda/enable_sm_projects/",
            timeout=core.Duration.seconds(120),
        )
        enable_sm_project_lambda.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:EnableSagemakerServicecatalogPortfolio",
                    "servicecatalog:ListAcceptedPortfolioShares",
                    "servicecatalog:AssociatePrincipalWithPortfolio",
                    "servicecatalog:AcceptPortfolioShare",
                    "iam:GetRole",
                ],
                resources=["*"],
            ),
        )
        sm_project_provider = Provider(self, "sg-project-lead-provider", on_event_handler=enable_sm_project_lambda)

        core.CustomResource(
            self,
            "sg-project",
            service_token=sm_project_provider.service_token,
            removal_policy=core.RemovalPolicy.DESTROY,
            resource_type="Custom::EnableSageMakerProjects",
            properties={
                "iteration": 1,
                "ExecutionRoles": [role.role_arn],
            },
        )
        # Create users using variables file
        if variables["SageMakerUserProfiles"]:
            for user in variables["SageMakerUserProfiles"]:
                sagemaker.CfnUserProfile(
                    self,
                    f"User-{user}",
                    domain_id=domain.attr_domain_id,
                    user_profile_name=user,
                    user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                        execution_role=role.role_arn,
                    ),
                )

        CfnOutput(self, "domain_id", value=domain.attr_domain_id)
        
        ## CDK NAG suppression
        NagSuppressions.add_resource_suppressions(role,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM4",
                                            "reason": "Sagemaker Notebook policies need to be broad to allow access to ",
                                            },{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "SageMaker Studio Role requires access to all indicies",
                                            }
                                        ],
                            apply_to_children=True)
        
        NagSuppressions.add_resource_suppressions(enable_sm_project_lambda,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "To enable SageMaker projects requires access to all indicies",
                                            }
                                        ],
                            apply_to_children=True)
        
        NagSuppressions.add_resource_suppressions(sm_project_provider,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "SageMaker Project provider requires access to all indicies",
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