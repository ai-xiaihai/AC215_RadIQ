#!/bin/bash

# set -e

export IMAGE_NAME="x-ray-app-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export GCP_PROJECT="ac215-radiq"
export GCS_BUCKET_NAME="x-ray-app-ml-workflow-demo"
export GCS_SERVICE_ACCOUNT="ml-workflow@ac215-radiq.iam.gserviceaccount.com"
export GCP_REGION="us-central1"
export GCS_PACKAGE_URI="gs://x-ray-app-trainer-code"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .


# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$BASE_DIR/../../model":/app/model \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../data-preprocessor":/data-preprocessor \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e GCP_REGION=$GCP_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
$IMAGE_NAME
