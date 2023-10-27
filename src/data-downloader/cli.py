import argparse
import subprocess

def main(args=None):
    if args.download:
        command = ["bash", "data_download.sh", "MS_CXR_Local_Alignment_v1.0.0.csv", "raw"]

        try:
            subprocess.run(command, check=True)
            print("Bash script executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="download images for preprocessing step",
    )

    args = parser.parse_args()

    main(args)