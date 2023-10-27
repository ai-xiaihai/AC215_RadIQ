"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --clean
"""

import argparse
import subprocess


def main(args=None):
    if args.preprocessing:
        print("Preprocess dataset")
        command = ["python", "data_preprocessing.py"]

        try:
            subprocess.run(command, check=True)
            print("Bash script executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Preprocessor CLI")

    parser.add_argument(
        "-p",
        "--preprocessing",
        action="store_true",
        help="Preprocess images",
    )

    args = parser.parse_args()

    main(args)