import argparse
import os
import sys
import traceback
import sagemaker
from sagemaker import hyperparameters as hp
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.model_monitor import DatasetFormat
from sagemaker.processing import (FrameworkProcessor, ProcessingInput,ProcessingOutput)
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import (ContinuousParameter, HyperparameterTuner, IntegerParameter)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.lambda_step import (LambdaOutput, LambdaOutputTypeEnum,LambdaStep)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ( ParameterInteger, ParameterString, ParameterBoolean)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import (DataQualityCheckConfig, ModelQualityCheckConfig, QualityCheckStep)
from sagemaker.workflow.steps import (CacheConfig, ProcessingStep, TrainingStep, TuningStep, TransformStep)
from sagemaker.transformer import Transformer

def start_pipeline():
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )
    parser.add_argument(
        "-sm-pipeline-name",
        "--sm-pipeline-name",
        dest="sm_pipeline_name",
        type=str,
        help="SageMaker Pipeline Name",
        default="model-train-deploy-pipeline",
    )
    parser.add_argument(
        "-default-bucket",
        "--default-bucket",
        dest="default_bucket",
        type=str,
        help="The default bucket your sagemaker session will assume.",
        default="smpipelinestack-pipelinebucket8ea2fa6c-12skbwxub27we"
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
        default="arn:aws:iam::123456789012:role/SMPipelineStack-ProjectRoleFAA285A5-aQVDxtzFb9xO"
    )
    parser.add_argument(
        "-use-amt",
        "--use-amt",
        dest="use_amt",
        type=str,
        help="Model Tuning use",
        default='no',
    )
    parser.add_argument(
        "-sm-model-deploy-lambda-arn",
        "--sm-model-deploy-lambda-arn",
        dest="sm_model_deploy_lambda_arn",
        type=str,
        help="SageMaker Model Deploy lambda ARN",
        default="arn:aws:lambda:eu-west-1:123456789012:function:SMPipelineStack-sagemakermodeldeploy3D95D968-Qou7VCIMt5cW"
    )
    
    args = parser.parse_args()
    print(args)

    try:
        pipeline_session = PipelineSession(default_bucket=args.default_bucket)
        sm_pipeline_name = args.sm_pipeline_name
        default_bucket = pipeline_session.default_bucket()
        role_arn = args.role_arn
        sm_model_deploy_lambda_arn = args.sm_model_deploy_lambda_arn
        s3_prefix = "mlops-pipeline"
        use_amt = args.use_amt
        region = sagemaker.Session().boto_region_name
        pipleine_exec_id=ExecutionVariables.PIPELINE_EXECUTION_ID
        # pipleine_exec_id="static-pipeline-exec-id"    ## only for dev purpose

        cache_config = CacheConfig(enable_caching=True, expire_after="p30d")

        processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
        processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
        
        training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
        training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
        training_instance_volume = ParameterInteger(name="TrainingInstanceVolume", default_value=20)

        model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

        job_config_instance_count = ParameterInteger(name="JobConfigInstanceCount", default_value=1)
        job_config_instance_type = ParameterString(name="JobConfigInstanceType", default_value="ml.c5.xlarge")
        job_config_volume = ParameterInteger(name="JobConfigVolume", default_value=20)

        ### for data quality check step
        skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
        register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
        supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value="")
        supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value="")

        ### for data bias check step
        skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
        register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
        supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value="")

        ### for model quality check step
        skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
        register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
        supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value="")
        supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value="")

        ### for model bias check step
        skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
        register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
        supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value="")

        ### for model explainability check step
        skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
        register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
        supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value="")

        ### Other parameters
        state_dim = ParameterString(name="StateDim", default_value="")
        input_data_uri = ParameterString(name="InputDataUri", default_value="")
        model_package_group_name = ParameterString(name="ModelPackageGroupName", default_value="modelpackagegroupname")

        ################ processing_step ################
        # Processing step for feature engineering
        sklearn_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version="1.2-1",
            role=role_arn,
            instance_type=processing_instance_type,
            instance_count=processing_instance_count,
            base_job_name="process-training-data",
            sagemaker_session=pipeline_session,
        )

        output_path_training = Join(on='/', values=['s3:/', default_bucket, 'training/data/processed', state_dim, pipleine_exec_id, 'training'])
        output_path_validation = Join(on='/', values=['s3:/', default_bucket, 'training/data/processed', state_dim, pipleine_exec_id, 'validation'])
        output_path_test = Join(on='/', values=['s3:/', default_bucket, 'training/data/processed', state_dim, pipleine_exec_id, 'test'])

        step_args = sklearn_processor.run(
            code="preprocess.py",
            source_dir="preprocessing",
            inputs=[
            ProcessingInput(source=input_data_uri, 
                            destination="/opt/ml/processing/input"),  
            ],
            outputs=[
                ProcessingOutput(
                    output_name="training",
                    source="/opt/ml/processing/training",
                    destination=output_path_training
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/validation",
                    destination=output_path_validation
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test",
                    destination=output_path_test
                ),
            ],
            # notice that all arguments passed to a SageMaker processing job should be strings as they are transformed to command line parameters.
            # Your read_parameters function will handle the data types for your code 
            arguments=[
                "--state_dim", state_dim.to_string(),
            ]
        )

        processing_step = ProcessingStep(
            name="PreProcessDatasetStep",
            step_args=step_args,
            cache_config=cache_config,
        )

        ################ data_quality_check_step ################
        # Configure the Data Quality Baseline Job
        check_job_config = CheckJobConfig(
                                            role=role_arn,
                                            instance_count=job_config_instance_count,
                                            instance_type=job_config_instance_type,
                                            volume_size_in_gb=job_config_volume,
                                            sagemaker_session=pipeline_session,
                                        )

        data_quality_check_output_s3 = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'DataQualityCheckStep', state_dim, pipleine_exec_id])

        data_quality_check_config = DataQualityCheckConfig(
                                    baseline_dataset=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
                                    dataset_format=DatasetFormat.csv(
                                        header=False, output_columns_position="START"
                                    ),
                                    output_s3_uri=data_quality_check_output_s3,
                                )

        data_quality_check_step = QualityCheckStep(
            name="DataQualityCheck",
            skip_check=skip_check_data_quality,
            register_new_baseline=register_new_baseline_data_quality,
            quality_check_config=data_quality_check_config,
            check_job_config=check_job_config,
            model_package_group_name=model_package_group_name,
            supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
            cache_config=cache_config,
        )

        ################ data_bias_check_step ################
        # Configure the Data Config
        from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig
        from sagemaker.workflow.clarify_check_step import (
            DataBiasCheckConfig,
            ClarifyCheckStep,
            ModelBiasCheckConfig,
            ModelPredictedLabelConfig,
            ModelExplainabilityCheckConfig,
            SHAPConfig,
        )
        data_bias_check_output_path = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'DataBiasCheckStep', state_dim, pipleine_exec_id])
        data_bias_analysis_config_output_path = f"s3://{default_bucket}/{s3_prefix}/DataBiasCheckStep/AnalysisConfig"
        data_bias_data_config = DataConfig(
            s3_data_input_path=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
            s3_output_path=data_bias_check_output_path,
            label=0,
            dataset_type="text/csv",
            s3_analysis_config_output_path=data_bias_analysis_config_output_path,
        )

        data_bias_config = BiasConfig(
            label_values_or_threshold=[0], 
            facet_name=[17], 
            facet_values_or_threshold=[[0.5]]
        )

        data_bias_check_config = DataBiasCheckConfig(
            data_config=data_bias_data_config,
            data_bias_config=data_bias_config,
        )

        data_bias_check_step = ClarifyCheckStep(
            name="DataBiasCheckStep",
            clarify_check_config=data_bias_check_config,
            check_job_config=check_job_config,
            skip_check=skip_check_data_bias,
            register_new_baseline=register_new_baseline_data_bias,
            supplied_baseline_constraints=supplied_baseline_constraints_data_bias,
            model_package_group_name=model_package_group_name,
            cache_config=cache_config,
        )


        ################ training_step & tuning_step ################
        output_path_model = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'model', state_dim, pipleine_exec_id])
        xgboost_container_img = image_uris.retrieve("xgboost", region, "1.7-1")
        hyperparameters = {
                "max_depth":"5",
                "eta":"0.2",
                "gamma":"4",
                "min_child_weight":"6",
                "subsample":"0.7",
                "objective":"binary:logistic",
                "num_round":"50"}
        
        xgboost_estimator = Estimator(
                            image_uri=xgboost_container_img,
                            hyperparameters=hyperparameters,
                            instance_type=training_instance_type,
                            volume_size=training_instance_volume,
                            instance_count=training_instance_count,
                            output_path=output_path_model,
                            role=role_arn,
                            sagemaker_session=pipeline_session
                    )

        hyperparameter_ranges = {
                "alpha": ContinuousParameter(10, 1000, scaling_type="Logarithmic"),
                "min_child_weight": IntegerParameter(20, 100),
                "subsample": ContinuousParameter(0.5, 1, scaling_type="Logarithmic"),
                "eta": ContinuousParameter(0.1, 1, scaling_type="Logarithmic"),
                "num_round": IntegerParameter(1, 4000),
            }

        if use_amt=='yes':
            tuner = HyperparameterTuner(
                estimator=xgboost_estimator,
                objective_metric_name="validation:auc",
                hyperparameter_ranges=hyperparameter_ranges,
                max_jobs=6,
                max_parallel_jobs=2,
                objective_type="Maximize",
            )
            tuning_step = TuningStep(
                        tuner=tuner,
                        name="ModelHPTuning",
                        inputs={
                            "train": TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri, content_type="csv",),
                            "validation": TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="csv",),
                        },
                        depends_on=[data_bias_check_step.name, 
                                    data_quality_check_step.name],
                    cache_config=cache_config,
            )
        else:
            training_step = TrainingStep(
                name="ModelTraining",
                estimator=xgboost_estimator,
                inputs={
                    "train": TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri, content_type="csv",),
                    "validation": TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="csv",),
                },
                 depends_on=[data_bias_check_step.name, 
                            data_quality_check_step.name],
                cache_config=cache_config,
            )
            
        if use_amt=='yes':
            hp_tune_model_path = Join(on='/', values=[default_bucket, s3_prefix, 'model', state_dim , pipleine_exec_id])
            s3_model_artifact= tuning_step.get_top_model_s3_uri(top_k=0,
                                                                s3_bucket=hp_tune_model_path,
                                                                )
        else:
            s3_model_artifact = training_step.properties.ModelArtifacts.S3ModelArtifacts

        ################ evaluation_step ################
        evaluation_processor = FrameworkProcessor(
                    estimator_cls=SKLearn,
                    framework_version="1.2-1", 
                    role=role_arn,
                    instance_type=processing_instance_type,
                    instance_count=processing_instance_count,
                    base_job_name="evaluation",
                    sagemaker_session=pipeline_session,
                )

        evaluation_report = PropertyFile(
            name="CustEvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        )

        evaluation_output_path_model = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'evaluation', state_dim, pipleine_exec_id])
        evaluation_args = evaluation_processor.run(
            code="evaluation.py",
            source_dir="evaluation",
            inputs=[
                ProcessingInput(
                    #source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    source=s3_model_artifact,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source= processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation", source="/opt/ml/processing/evaluation",
                    destination=evaluation_output_path_model,
                ),
            ],
        )

        evaluation_step = ProcessingStep(
            name="ModelEvaluation",
            step_args=evaluation_args,
            property_files=[evaluation_report],
            cache_config=cache_config,
        )

        ################ post_training_step ################
        # Specify model metric and drift baseline metadata to register
        post_training_processor = FrameworkProcessor(
                    estimator_cls=SKLearn,
                    framework_version="1.2-1",
                    role=role_arn,
                    instance_type=processing_instance_type,
                    instance_count=processing_instance_count,
                    base_job_name="PostTrainingAnalysis",
                    sagemaker_session=pipeline_session,
                )
        post_training_analysis_output_s3 = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'PostTrainingAnalysis', state_dim, pipleine_exec_id])
        feature_importance_output_s3 = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'FeatureImportance', state_dim, pipleine_exec_id])

        post_training_args = post_training_processor.run(
            code="modelevaljob.py",
            source_dir="postprocessing",
            inputs=[
                ProcessingInput(
                    # source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    source=s3_model_artifact,
                    destination="/opt/ml/processing/model",
                    input_name="model",
                ),
                ProcessingInput(
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                    input_name="test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="post_training_analysis",
                    source="/opt/ml/processing/post_training_analysis",
                    destination=post_training_analysis_output_s3,
                ),
                ProcessingOutput(
                    output_name="feature_importance",
                    source="/opt/ml/processing/feature_importance",
                    destination=feature_importance_output_s3,
                ),
            ],
        )
        post_training_step = ProcessingStep(
            name="PostTrainingAnalysisStep",
            step_args=post_training_args,
            cache_config=cache_config,
        )

        # ################ create_model_step ################
        model = Model(
            image_uri=xgboost_container_img,
            model_data=s3_model_artifact,
            role=role_arn,
            sagemaker_session=pipeline_session,
        )

        # Register model step that will be conditionally executed
        create_model_step = ModelStep(
            name="XgBoostCreateModelStep",
            step_args=model.create(),
        )

        # ################ transform_step ################
        transformer_output_s3 = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'transformer', state_dim, pipleine_exec_id])

        transformer = Transformer(
            model_name=create_model_step.properties.ModelName,
            instance_count=processing_instance_count,
            instance_type=processing_instance_type,
            accept="text/csv",
            assemble_with="Line",
            output_path=transformer_output_s3,
            sagemaker_session=pipeline_session,
        )
        transform_arg = transformer.transform(
                    data=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
                    content_type="text/csv",
                    split_type="Line",
                    join_source="Input",
                    input_filter="$[1:]",
                    output_filter="$[0,-1]",
        )
        transform_step = TransformStep(
            name="TransformDataStep",
            step_args=transform_arg,
            cache_config=cache_config,
        )
        
        ################ model_quality_check_step ################
        model_quality_check_step_s3 = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'modelqualitycheckstep', state_dim, pipleine_exec_id])

        model_quality_check_config = ModelQualityCheckConfig(
                baseline_dataset=transform_step.properties.TransformOutput.S3OutputPath,
                dataset_format=DatasetFormat.csv(header=False),
                output_s3_uri=model_quality_check_step_s3,
                problem_type="BinaryClassification",
                probability_attribute="_c1",
                probability_threshold_attribute="0.5",
                ground_truth_attribute="_c0",
            )

        model_quality_check_step = QualityCheckStep(
                name="ModelQualityCheckStep",
                skip_check=skip_check_model_quality,
                register_new_baseline=register_new_baseline_model_quality,
                quality_check_config=model_quality_check_config,
                check_job_config=check_job_config,
                model_package_group_name=model_package_group_name,
                supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
                supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
            )
        
        ################ model_bias_check_step ################
        model_bias_check_output_path = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'ModelBiasCheckStep', state_dim, pipleine_exec_id])
        model_bias_analysis_config_output_path = f"s3://{default_bucket}/{s3_prefix}/ModelBiasCheckStep/AnalysisConfig"

        model_bias_data_config = DataConfig(
            s3_data_input_path=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
            s3_output_path=model_bias_check_output_path,
            label=0,
            dataset_type="text/csv",
            s3_analysis_config_output_path=model_bias_analysis_config_output_path,
        )

        ## At the time of development of this code. You can not parametrize instance_count and instance_type., 
        ## If you try you will get "exception: Object of type ParameterString is not JSON serializable "
        model_config = ModelConfig(
                model_name=create_model_step.properties.ModelName,
                instance_count=1,
                instance_type="ml.m5.xlarge",  
        )
        
        model_bias_config = BiasConfig(
            label_values_or_threshold=[0], 
            facet_name=[17], 
            facet_values_or_threshold=[[0.5]]
        )
        
        model_bias_check_config = ModelBiasCheckConfig(
            data_config=model_bias_data_config,
            data_bias_config=model_bias_config,
            model_config=model_config,
            model_predicted_label_config=ModelPredictedLabelConfig(),
        )

        model_bias_check_step = ClarifyCheckStep(
            name="ModelBiasCheckStep",
            clarify_check_config=model_bias_check_config,
            check_job_config=check_job_config,
            skip_check=skip_check_model_bias,
            register_new_baseline=register_new_baseline_model_bias,
            supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
            model_package_group_name=model_package_group_name,
        )

        ################ model_explainability_check_step ################
        model_explainability_check_output_path = Join(on='/', values=['s3:/', default_bucket, s3_prefix, 'ModelExplainabilityCheckStep', state_dim, pipleine_exec_id])
        model_explainability_analysis_config_output_path = f"s3://{default_bucket}/{s3_prefix}/ModelExplainabilityCheckStep/AnalysisConfig"

        model_explainability_data_config = DataConfig(
            s3_data_input_path=processing_step.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
            s3_output_path=model_explainability_check_output_path,
            s3_analysis_config_output_path=model_explainability_analysis_config_output_path,
            label=0,
            dataset_type="text/csv",
        )

        shap_config = SHAPConfig(seed=123, num_samples=11)

        model_explainability_check_config = ModelExplainabilityCheckConfig(
            data_config=model_explainability_data_config,
            model_config=model_config,
            explainability_config=shap_config,
        )

        model_explainability_check_step = ClarifyCheckStep(
            name="ModelExplainabilityCheckStep",
            clarify_check_config=model_explainability_check_config,
            check_job_config=check_job_config,
            skip_check=skip_check_model_explainability,
            register_new_baseline=register_new_baseline_model_explainability,
            supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
            model_package_group_name=model_package_group_name,
        )

        ################ register_step ################
        model_metrics = ModelMetrics(
            model_data_statistics=MetricsSource(
                s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
                content_type="application/json",
            ),
            model_data_constraints=MetricsSource(
                s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
            bias_pre_training=MetricsSource(
                s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
            model_statistics=MetricsSource(
                s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
                content_type="application/json",
            ),
            model_constraints=MetricsSource(
                s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            bias_post_training=MetricsSource(
                s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
            explainability=MetricsSource(
                s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
        )

        drift_check_baselines = DriftCheckBaselines(
            model_data_statistics=MetricsSource(
                s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
                content_type="application/json",
            ),
            model_data_constraints=MetricsSource(
                s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            bias_pre_training_constraints=MetricsSource(
                s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            model_statistics=MetricsSource(
                s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
                content_type="application/json",
            ),
            model_constraints=MetricsSource(
                s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            bias_post_training_constraints=MetricsSource(
                s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            explainability_constraints=MetricsSource(
                s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            explainability_config_file=FileSource(
                s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
                content_type="application/json",
            ),
        )
        
        ################ create_model_step ################
        model_registry_args = model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            approval_status=model_approval_status,
            model_package_group_name=model_package_group_name,
            customer_metadata_properties={
                "ModelName": create_model_step.properties.ModelName
            },
            drift_check_baselines=drift_check_baselines,
            model_metrics=model_metrics,
        )
        register_step = ModelStep(name="RegisterModel", step_args=model_registry_args)


        ################ model_deployment ################

        output_param_1 = LambdaOutput(output_name='statusCode', output_type=LambdaOutputTypeEnum.String)
        output_param_2 = LambdaOutput(output_name='body', output_type=LambdaOutputTypeEnum.String)

        deploy_lambda_step = LambdaStep(
            name='LambdaModelDeploy',
            lambda_func=Lambda(function_arn=sm_model_deploy_lambda_arn),
            inputs={
                'model_package_group_name': model_package_group_name,
                'state_dim': state_dim,
                'role_arn': role_arn
            },
            outputs=[
                output_param_1, 
                output_param_2
            ],
        )
        deploy_lambda_step.add_depends_on([register_step])

        ################ condition_step ################
        # Condition step for evaluating model quality and branching execution
        cond_lte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path='binary_classification_metrics.accuracy.value',
            ),
            right=0.5,     ## This value needs to be selected if really matters
        )

        condition_step = ConditionStep(
            name="ConditionStep",
            conditions=[cond_lte],
            if_steps=[
                    post_training_step,
                    create_model_step,
                    transform_step,
                    model_quality_check_step,
                    model_bias_check_step,
                    model_explainability_check_step,
                    register_step,
                    deploy_lambda_step,
            ],
            else_steps=[],
        )

        ################ pipeline Start ################
        # Pipeline instance
        pipeline = Pipeline(
            name=sm_pipeline_name,
            steps=[ processing_step,
                    tuning_step if use_amt=='yes' else training_step,
                    data_quality_check_step,  # data_quality_check_step is not supported in local mode
                    data_bias_check_step,
                    evaluation_step,
                    condition_step,
                ],
            parameters=[
                        ## Processing Step
                        processing_instance_count,
                        processing_instance_type,
                        ## Training Step
                        training_instance_type,
                        training_instance_count,
                        training_instance_volume,
                        ## Job Config Step
                        job_config_instance_count,
                        job_config_instance_type,
                        job_config_volume,
                        
                        model_approval_status,

                        ## Data Quality Check step
                        skip_check_data_quality,
                        register_new_baseline_data_quality,
                        supplied_baseline_statistics_data_quality,
                        supplied_baseline_constraints_data_quality,
                        ## Data Bias Check Step
                        skip_check_data_bias,
                        register_new_baseline_data_bias,
                        supplied_baseline_constraints_data_bias,
                        ## Model Quality Check Step
                        skip_check_model_quality,
                        register_new_baseline_model_quality,
                        supplied_baseline_statistics_model_quality,
                        supplied_baseline_constraints_model_quality,
                        ## Model Bias Check Step
                        skip_check_model_bias,
                        register_new_baseline_model_bias,
                        supplied_baseline_constraints_model_bias,
                        ## Model Explainability Check Step
                        skip_check_model_explainability,
                        register_new_baseline_model_explainability,
                        supplied_baseline_constraints_model_explainability,
                        ## new param
                        input_data_uri,
                        state_dim,
                        model_package_group_name,
                    ],
            # sagemaker_session=local_pipeline_session
            sagemaker_session=pipeline_session
        )

        # all_tags = [
        #     {"Key": "sagemaker:project-name", "Value": args.sm_project_name},
        #     {"Key": "sagemaker:project-id", "Value": args.sm_project_id},
        # ]
        # upsert_response = pipeline.upsert(role_arn=role_arn, tags=all_tags)

        upsert_response = pipeline.upsert(role_arn=role_arn)
        print("Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)


    except Exception as e:  # pylint: disable=W0703

        print(f"Exception: {e}")

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    start_pipeline()
