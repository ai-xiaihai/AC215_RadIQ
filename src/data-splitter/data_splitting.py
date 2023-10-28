import pandas as pd
import numpy as np
np.random.seed(123)
import shutil
import os
from google.cloud import storage

dataset_folder = "/app/data/downsized"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def data_download(gcs=GCS_BUCKET_NAME):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs)
    prefix = "ms_cxr/downsized"
    blobs = bucket.list_blobs(prefix=prefix)
    blob_csv = bucket.list_blobs(prefix = "ms_cxr/label_1024.csv")

    for blob in blob_csv:
        destination_file_path = "/app/data/label_1024.csv"
        if not os.path.exists(destination_file_path):
            blob.download_to_filename(destination_file_path)
            print(f'File {blob.name} downloaded to {destination_file_path}')

    # Make dirs
    os.makedirs(dataset_folder, exist_ok=True)

    for blob in blobs:
        destination_file_path = os.path.join(dataset_folder, blob.name[len(prefix)+1:])
        if not os.path.exists(destination_file_path):
            blob.download_to_filename(destination_file_path)
            print(f'File {blob.name} downloaded to {destination_file_path}')


def data_split():
    label_path = 'data/label_1024.csv'
    image_path = "data/downsized"

    # Load CSV file
    df = pd.read_csv(label_path)

    # Check if folder exists, if not create it
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('val'):
        os.makedirs('val')
    if not os.path.exists('test'):
        os.makedirs('test')

    # Split data into train, val, test with 800, 123, 124 images respectively
    image_names = list(df['dicom_id'].unique())
    np.random.shuffle(image_names)
    train_image_names, val_image_names, test_image_names = image_names[:700], image_names[700:923], image_names[923:]

    # Copy image to new folder according to image path
    for image_name in train_image_names:
        image_name = image_name + '.jpg'
        abs_path = os.path.join(image_path, image_name)
        shutil.copy(abs_path, 'train/' + image_name)
    for image_name in val_image_names:
        image_name = image_name + '.jpg'
        abs_path = os.path.join(image_path, image_name)
        shutil.copy(abs_path, 'val/' + image_name)
    for image_name in test_image_names:
        image_name = image_name + '.jpg'
        abs_path = os.path.join(image_path, image_name)
        shutil.copy(abs_path, 'test/' + image_name)

    # Add a new column to the dataframe to indicate which split the image belongs to
    df.loc[df['dicom_id'].isin(train_image_names), 'split'] = 'train'
    df.loc[df['dicom_id'].isin(val_image_names), 'split'] = 'val'
    df.loc[df['dicom_id'].isin(test_image_names), 'split'] = 'test'

    # Save new csv file
    df.to_csv('label_1024_split.csv', index=False)


if __name__ == "__main__":
    data_split()
    print("data splitting success")