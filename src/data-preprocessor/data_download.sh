#!/bin/bash

# Local path
mkdir -p radiq-app-data
LOCAL_PATH="./radiq-app-data/"

# GCS paths
GCS_PATHS=()
for input_name in "$@"; do
    # Construct the path and add it to the GCS_PATHS array
    GCS_PATHS+=("gs://radiq-app-data/ms_cxr/$input_name/")
done
declare -a GCS_PATHS

# Download each file or directory from GCS to the local 'data' folder
for GCS_PATH in "${GCS_PATHS[@]}"; do
    gsutil cp -r $GCS_PATH $LOCAL_PATH
    if [ $? -eq 0 ]; then
        echo "Download of $GCS_PATH successful!"
    else
        echo "Error occurred while downloading $GCS_PATH."
        exit 1
    fi
done
