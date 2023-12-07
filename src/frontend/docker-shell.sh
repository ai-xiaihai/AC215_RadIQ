#!/bin/bash

set -e

export IMAGE_NAME="radiq-frontend-react"

docker build -t $IMAGE_NAME -f Dockerfile.dev .
docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 5173:5173 $IMAGE_NAME