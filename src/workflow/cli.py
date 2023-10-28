"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from model_workflow import model_training


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = "radiq-app-data"
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

DATA_PREPROCESSOR_IMAGE = "lic604/x-ray-app-data-preprocessor"
DATA_SPLITTER_IMAGE = "lic604/x-ray-app-data-splitter"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

    if args.all:
        print("Full Preprocess + Split + Training Pipeline")

        # Define a Container Component
        @dsl.container_component
        def data_proprocessor():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    "--bucket=radiq-app-data"
                ],
            )
            return container_spec
        
        # Define a Container Component
        @dsl.container_component
        def data_splitter():
            container_spec = dsl.ContainerSpec(
                image=DATA_SPLITTER_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    "--bucket=radiq-app-data"
                ],
            )
            return container_spec

        @dsl.pipeline
        def ml_pipeline():
            # Data Preprocessor
            data_proprocessor_task = data_proprocessor().set_display_name(
                "Data Proprocessor"
            )

            data_splitter_task = data_splitter().set_display_name(
                "Data Splitter"
            ).after(data_proprocessor_task)
            
            # Model Training (serverless)
            GCS_BUCKET_NAME = "xray-ml-workflow"
            _ = (
                model_training(
                    project=GCP_PROJECT,
                    location=GCP_REGION,
                    staging_bucket=GCS_PACKAGE_URI,
                    bucket_name=GCS_BUCKET_NAME,
                )
                .set_display_name("Model Training")
                .after(data_splitter_task)
            )
            

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            ml_pipeline, package_path="ml_pipeline.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "x-ray-app-ml_pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="ml_pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_preprocessing:
        print("Full Preprocess Pipeline")
        GCS_BUCKET_NAME = "radiq-app-data"

        # Define a Container Component
        @dsl.container_component
        def data_proprocessor():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        @dsl.pipeline
        def preprocessing_pipeline():
            # Data Collector
            data_proprocessor().set_display_name(
                "Data Proprocessor"
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            preprocessing_pipeline, package_path="data-preprocessor.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "x-ray-app-data-preprocessing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data-preprocessor.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_splitting:
        print("Full splitting Pipeline")
        GCS_BUCKET_NAME = "radiq-app-data"

        # Define a Container Component
        @dsl.container_component
        def data_splitter():
            container_spec = dsl.ContainerSpec(
                image=DATA_SPLITTER_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        @dsl.pipeline
        def splitting_pipeline():
            # Data Collector
            data_splitter().set_display_name(
                "Data Splitter"
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            splitting_pipeline, package_path="data-splitting.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "x-ray-app-data-splitting-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data-splitting.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_workflow:
        print("Serverless Training Pipeline")

        @dsl.pipeline
        def model_training_pipeline():
            model_training(
                project=GCP_PROJECT,
                location=GCP_REGION,
                staging_bucket=GCS_PACKAGE_URI,
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_training_pipeline, package_path="model_training.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "x-ray-app-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_training.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Serverless training",
    )
    parser.add_argument(
        "-m",
        "--model_workflow",
        action="store_true",
        help="Serverless training",
    )
    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="data Preprocessing",
    )
    parser.add_argument(
        "-s",
        "--data_splitting",
        action="store_true",
        help="Data Splitting",
    )

    args = parser.parse_args()

    main(args)