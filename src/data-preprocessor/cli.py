"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --clean
"""

import argparse
import subprocess
from data_preprocessing import data_resize, data_downlaod

def main(args=None):
    download_data_command = ["bash", "data_download.sh", "MS_CXR_Local_Alignment_v1.0.0.csv", "raw"]
    preprocessing_command = ["python", "data_preprocessing.py"]
    upload_data_command = ["bash", "data_upload.sh", "label_1024.csv", "downsized"]

    if args.download:
        print("Download dataset")
        try:
            subprocess.run(download_data_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.preprocessing:
        print("Preprocess dataset")
        try:
            subprocess.run(preprocessing_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.download_and_preprocessing:
        print("Download and Preprocess dataset")

        try:
            subprocess.run(download_data_command, check=True)
            subprocess.run(preprocessing_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.upload:
        print("Upload preprocessed dataset")

        try:
            subprocess.run(upload_data_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.all:
        print("Download and Preprocess dataset")

        # If bucket was passed as argument
        GCS_BUCKET_NAME = args.bucket

        try:
            data_downlaod(gcs=GCS_BUCKET_NAME)
            data_resize()
        except subprocess.CalledProcessError as e:
            print(f"Frank Error running Bash script: {e}")

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Preprocessor CLI")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download images",
    )

    parser.add_argument(
        "-p",
        "--preprocessing",
        action="store_true",
        help="Preprocess images",
    )

    parser.add_argument(
        "-dp",
        "--download_and_preprocessing",
        action="store_true",
        help="Download and preprocess images",
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Download, preprocess and upload images",
    )

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload images",
    )

    parser.add_argument(
        "-b", "--bucket", type=str, default="", help="Bucket Name to save the data"
    )

    args = parser.parse_args()

    main(args)