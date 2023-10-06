AC215-RadIQ
==============================

Project Organization
---------
- project-root/
    - LICENSE
    - README.md
    - docker-compose.yml
    - notebooks/
        - AC215_RadIQ_EDA.ipynb
    - radiq-app-data/
        - ms_cxr.dvc
    - secrets/
        - data-service-account.json
    - src/
        - __init__.py
        - build_features.py
        - data_extraction.py
        - data_preprocessing/
            - .dockerignore
            - data_preprocessing.py
            - Dockerfile
            - Pipfile
        - data_splitting/
            - .dockerignore
            - data_splitting.py
            - Dockerfile
            - Pipfile



--------

# AC215 - Interactive X-ray Insight

**Team Members**
Martin Ma, Lily Wang, Frank Cheng, Linglai Chen, Chenbing Wang

**Group Name**
RadIQ

**Project**
This project aims to develop an application that allows patients to better understand their chest X-ray diagnosis through an interactive web interface. By integrating chest X-rays with their associated radiology reports through multi-modal learning, users can highlight any phrases in the report, which would light up the relevant region on the X-ray.

### Milestone 3 ###

**Experiment Tracking**
Below you can see the output from our Weights & Biases page. We used this tool to track several iterations of our model training. It was tracked using the `wandb` library we included inside of our `src/data_pipeline/main.py` script.
![WandB Screenshot](./images/wandb.png)


**Docker Container**
- A Dockerfile is created inside /src/data_pipeline. To run it, run `bash docker-shell.sh` on the root level. This will open an interactive bash terminal.
- Inside the container, go to `src/data_pipeline` and run `bash data_download.sh` to download preprocessed image data from GCP bucket.

**Docker Setup**
- Two Dockerfiles are created in /src/data_preprocessing and /src/data_splitting. They are used to create a 
  container for data preprocessing and data splitting, respectively. All docker services are managed in 
  /src/docker-compose.yml. To start a service listed in docker-compose.yml, run `docker-compose up <service-name>` 
  on the root level directory. 
- For data preprocessing and data splitting service, we can also run an interactive terminal. To do this, run 
  `docker-compose run --entrypoint /bin/sh <service-name>` on the root level.

**Preprocess container**
- This container reads image data, resize them into a common size (e.g., 1024 x 1024), resize the ground-truth bounding box labels, and stores both images and ground-truth label file back to GCP.
- The input of this container includes the dimension of the resized image, the path to the image data, the path to the ground-truth label file, and the path to store the downsized image data.


**Split container**
- This container splits the data into training, validation, and testing sets.
- The input of this container includes the path to the image data, and the path to the ground-truth label file.


**Data pipline container**
- This container contains all our model training scripts.
- Run `bash docker-shell.sh` to build and run the data-pipeline Docker container.
- Ensure you have the secret file `secrets/data-service-account.json` before running the container. Once inside the container, you can execute `bash data_download.sh` to download the data from our GCP bucket.
- Inside the container, run `pip install pipenv && pipenv sync && pipenv shell` to set up a virtual environment with the necessary dependencies.
- Inside the container, go to `src/data_pipeline`, run `python3 main.py --log_to_wandb True` to train the model. This script implements `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` to enhance data ingestion and management within machine learning components of the project. Then it loads model architecture stored in `/model` and fits the model. It takes the following key arguments:
    > --log_to_wandb\[bool\]: Flag to log results to wandb, default is False

**Data Verison Control**
- We use [DVC](https://dvc.org) to version our dataset. The metadata file is located in `radiq-app-data/`. There is also a remote copy on google cloud storage at `gs://radiq-app-data/dvc_store/`.

**Useful commands to send data to GCP bucket**
- Look at data in gcp bucket: `gsutil ls gs://radiq-app-data/ms_cxr/`
- Copy a single file from local to gcp bucket: `gsutil cp label.csv gs://radiq-app-data/ms_cxr/`
- Copy a folder from local to gcp bucket: `gsutil -m cp -r raw/ gs://radiq-app-data/ms_cxr/`
