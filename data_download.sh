#!/bin/bash

# Local path
mkdir -p radiq-app-data
LOCAL_PATH="./radiq-app-data/"

# Hard-coded GCS paths
declare -a GCS_PATHS=(
                    "gs://radiq-app-data/ms_cxr/train/"
                    "gs://radiq-app-data/ms_cxr/val/"
                    "gs://radiq-app-data/ms_cxr/test/"
                    "gs://radiq-app-data/ms_cxr/label_1024_split.csv"
                    )

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