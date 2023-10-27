"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --clean
"""

import argparse
import subprocess


def main(args=None):
    download_data_command = ["bash", "data_download.sh", "label_1024.csv", "downsized"]
    splitting_command = ["python", "data_splitting.py"]
    upload_data_command = ["bash", "data_upload.sh", "train", "val", "test", "label_1024_split.csv"]

    if args.download:
        print("Download dataset")
        try:
            subprocess.run(download_data_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.splitting:
        print("Split dataset")
        try:
            subprocess.run(splitting_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.download_and_splitting:
        print("Download and split dataset")
        try:
            subprocess.run(download_data_command, check=True)
            subprocess.run(splitting_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.upload:
        print("Upload preprocessed dataset")

        try:
            subprocess.run(upload_data_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

    if args.all:
        print("Doanload, split and upload dataset")

        try:
            subprocess.run(download_data_command, check=True)
            subprocess.run(splitting_command, check=True)
            subprocess.run(upload_data_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

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
        "-s",
        "--splitting",
        action="store_true",
        help="Preprocess images",
    )

    parser.add_argument(
        "-ds",
        "--download_and_splitting",
        action="store_true",
        help="Download and split images",
    )

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload images",
    )

    args = parser.parse_args()

    main(args)