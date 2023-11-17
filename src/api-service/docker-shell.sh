#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="biovil-api-service"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export MODEL_DIR=$(pwd)/../../model/
export GCS_BUCKET_NAME="radiq-app-data"
export GCS_BUCKET_URI="gs://radiq-app-data/ms_cxr/"
export GCP_PROJECT="ac215-radiq"
export WANDB_KEY="6ac94bce286531b3989581a1c8c85cb014a32883"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
# # M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-v "$MODEL_DIR/health_multimodal":/app/api/health_multimodal \
-v "$MODEL_DIR/model.py":/app/api/model.py \
-v "$MODEL_DIR/dataset_mscxr.py":/app/api/dataset_mscxr.py \
-p 9000:9000 \
-e DEV=1 \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME