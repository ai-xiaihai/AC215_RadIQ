#!/bin/bash

set -e

export IMAGE_NAME=model-training-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export MODEL_DIR=$(pwd)/../../model/
export GCS_BUCKET_URI="gs://radiq-app-data/ms_cxr/"
export GCP_PROJECT="ac215-radiq"
export WANDB_KEY="6ac94bce286531b3989581a1c8c85cb014a32883"


# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$MODEL_DIR/health_multimodal":/app/package/trainer/health_multimodal \
-v "$MODEL_DIR/model.py":/app/package/trainer/model.py \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME