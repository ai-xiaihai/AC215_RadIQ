#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME=x-ray-app-data-preprocessor
export IMAGE_NAME_HUB=dooop/x-ray-app-data-preprocessor
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export DATA_DIR=$(pwd)/../data-downloader/radiq-app-data
export GCP_PROJECT="ac215-radiq"
export GCS_BUCKET_NAME=radiq-app-data

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# put this docker image to dockerhub
cat $SECRETS_DIR/dockerhub | docker login --username dooop --password-stdin
docker tag $IMAGE_NAME $IMAGE_NAME_HUB
docker push $IMAGE_NAME_HUB

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$DATA_DIR":/app/data \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
$IMAGE_NAME