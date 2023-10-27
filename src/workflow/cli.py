"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
import subprocess
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from model_workflow import model_training


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

# DATA_COLLECTOR_IMAGE = "gcr.io/ac215-project/mushroom-app-data-collector"
DATA_PREPROCESSOR_IMAGE = "dlops/mushroom-app-data-processor"
DATA_SPLITTER_IMAGE = "dlops/x-ray-app-data-splitting"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

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

    if args.sample1:
        print("Sample Pipeline 1")

        # Define Component
        @dsl.component
        def square(x: float) -> float:
            return x**2

        # Define Component
        @dsl.component
        def add(x: float, y: float) -> float:
            return x + y

        # Define Component
        @dsl.component
        def square_root(x: float) -> float:
            return x**0.5

        # Define a Pipeline
        @dsl.pipeline
        def sample_pipeline(a: float = 3.0, b: float = 4.0) -> float:
            a_sq_task = square(x=a)
            b_sq_task = square(x=b)
            sum_task = add(x=a_sq_task.output, y=b_sq_task.output)
            return square_root(x=sum_task.output).output

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            sample_pipeline, package_path="sample-pipeline1.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "sample-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="sample-pipeline1.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-c",
        "--data_collector",
        action="store_true",
        help="Run just the Data Collector",
    )
    parser.add_argument(
        "-p",
        "--data_preprocessor",
        action="store_true",
        help="Run just the Data Processor",
    )
    parser.add_argument(
        "-t",
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "-d",
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Mushroom App Pipeline",
    )
    parser.add_argument(
        "-m",
        "--model_workflow",
        action="store_true",
        help="Serverless training",
    )
    parser.add_argument(
        "-s1",
        "--sample1",
        action="store_true",
        help="Sample Pipeline 1",
    )

    args = parser.parse_args()

    main(args)