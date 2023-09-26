AC215-RadIQ
==============================

Project Organization
---------
        ├── LICENSE
        ├── README.md
        ├── docker
        │   ├── data_preprocessing
        │   │   └── Dockerfile
        │   └── data_splitting
        │       └── Dockerfile
        ├── docker-compose.yml
        ├── notebooks
        │   └── AC215_RadIQ_EDA.ipynb
        ├── radiq-app-data
        │   └── ms_cxr.dvc
        ├── secrets
        │   └── data-service-account.json
        └── src
            ├── __init__.py
            ├── build_features.py
            ├── data_extraction.py
            ├── data_preprocessing.py
            ├── data_splitting.py
            └── run.sh

--------

# AC215 - Milestone2 - Interactive X-ray Insight

**Team Members**
Martin Ma, Lily Wang, Frank Cheng, Linglai Chen, Chenbing Wang

**Group Name**
RadIQ

**Project**
This project aims to develop an application that allows patients to better understand their chest X-ray diagnosis through an interactive web interface. By integrating chest X-rays with their associated radiology reports through multi-modal learning, users can highlight any phrases in the report, which would light up the relevant region on the X-ray.


### Milestone 2 ###
In this milestone, we worked on the following tasks:
- Set up GCP and share among the team members.
- Store dataset on GCP bucket, and create the data pipeline, which includes data extraction, preprocessing, and splitting.
- Set up Docker.
- Set up DVC.
- Complete exploratory data analysis.


**Preprocess container**
- This container reads image data, resize them into a common size (e.g., 1024 x 1024), resize the ground-truth bounding box labels, and stores both images and ground-truth label file back to GCP.
- The input of this container includes the dimension of the resized image, the path to the image data, the path to the ground-truth label file, and the path to store the downsized image data.


**Split container**
- This container splits the data into training, validation, and testing sets.
- The input of this container includes the path to the image data, and the path to the ground-truth label file.

**Useful commands to send data to GCP bucket**
- Look at data in gcp bucket: `gsutil ls gs://radiq-app-data/ms_cxr/`
- Copy a single file from local to gcp bucket: `gsutil cp label.csv gs://radiq-app-data/ms_cxr/`
- Copy a folder from local to gcp bucket: `gsutil -m cp -r raw/ gs://radiq-app-data/ms_cxr/`