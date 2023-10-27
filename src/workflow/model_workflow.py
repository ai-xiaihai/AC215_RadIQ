from kfp import dsl


# Define a Container Component
@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_training(
    project: str = "ac215-radiq",
    location: str = "",
    staging_bucket: str = "",
    bucket_name: str = "",
    epochs: int = 30,
    batch_size: int = 32,
    model_name: str = "mobilenetv2",
    train_base: bool = False,
    wandb_key: str = "6ac94bce286531b3989581a1c8c85cb014a32883"
):
    print("Model Training Job")

    import google.cloud.aiplatform as aip

    # Initialize Vertex AI SDK for Python
    aip.init(project=project, location=location, staging_bucket=staging_bucket)

    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
    python_package_gcs_uri = f"{staging_bucket}biovil-trainer.tar.gz"

    job = aip.CustomPythonPackageTrainingJob(
        display_name="x-ray-app-training",
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name="trainer.train",
        container_uri=container_uri,
        project=project,
    )

    CMDARGS = [
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--model_name={model_name}",
        f"--wandb_key={wandb_key}"
    ]
    if train_base:
        CMDARGS.append("--train_base")

    MODEL_DIR = staging_bucket
    TRAIN_COMPUTE = "n1-standard-4"
    TRAIN_GPU = "NVIDIA_TESLA_T4"
    TRAIN_NGPU = 1

    print(python_package_gcs_uri)

    # Run the training job on Vertex AI
    # sync=True, # If you want to wait for the job to finish
    job.run(
        model_display_name=None,
        args=CMDARGS,
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        # accelerator_type=TRAIN_GPU,
        # accelerator_count=TRAIN_NGPU,
        base_output_dir=MODEL_DIR,
        sync=True,
    )
