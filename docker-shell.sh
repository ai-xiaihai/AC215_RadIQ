#!/bin/bash

set -e

export IMAGE_NAME=radiq-data-preprocessing
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export DATA_DIR=$(pwd)/../radiq-app-data/
export GCP_PROJECT="AC215-RadIQ"
export DOCKERFILE="src/data_pipeline/Dockerfile"

# Build the image based on the Dockerfile
# docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f $DOCKERFILE .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$DATA_DIR":/data \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
$IMAGE_NAME