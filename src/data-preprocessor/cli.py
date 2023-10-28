"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --clean
"""

import argparse
import subprocess
from data_preprocessing import data_resize, data_download

def main(args=None):
    upload_data_command = ["bash", "data_upload.sh", "label_1024.csv", "downsized"]

    if args.download:
        print("Download dataset")
        if GCS_BUCKET_NAME != "":
            data_download(gcs=GCS_BUCKET_NAME)
        else:
            data_download()

    if args.preprocessing:
        print("Preprocess dataset")
        data_resize()

    if args.download_and_preprocessing:
        print("Download and Preprocess dataset")

        if GCS_BUCKET_NAME != "":
            data_download(gcs=GCS_BUCKET_NAME)
        else:
            data_download()

        data_resize()

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
        print("GCS_BUCKET_NAME", GCS_BUCKET_NAME)
        
        if GCS_BUCKET_NAME != "":
            data_download(gcs=GCS_BUCKET_NAME)
        else:
            data_download()
            
        data_resize()

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