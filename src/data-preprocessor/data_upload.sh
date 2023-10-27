#!/bin/bash

LOCAL_PATHS=()
GCS_PATHS=()
for input_name in "$@"; do
    # Construct the path and add it to the GCS_PATHS array
    LOCAL_PATHS+=("./$input_name")
    GCS_PATHS+=("gs://radiq-app-data/ms_cxr/$input_name")
done

declare -a LOCAL_PATHS
declare -a GCS_PATHS

# Download each file or directory from GCS to the local 'data' folder
for ((i = 0; i < ${#LOCAL_PATHS[@]}; i++)); do
    gsutil cp -r ${LOCAL_PATHS[$i]} ${GCS_PATHS[$i]}
    if [ $? -eq 0 ]; then
        echo "Upload $input_name successful!"
    else
        echo "Error occurred while Uploading $input_name."
        exit 1
    fi
done
