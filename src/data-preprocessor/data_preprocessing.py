import pandas as pd
import os
import cv2
from google.cloud import storage

dataset_folder = "/app/data/raw"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def data_download(gcs=GCS_BUCKET_NAME):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs)
    prefix = "ms_cxr/raw"
    blobs = bucket.list_blobs(prefix=prefix)
    blob_csv = bucket.list_blobs(prefix = "ms_cxr/MS_CXR_Local_Alignment_v1.0.0.csv")

    for blob in blob_csv:
        destination_file_path = "/app/data/MS_CXR_Local_Alignment_v1.0.0.csv"
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

def data_resize():
    """Resize image to specified dimensions.
    """
    # Resize images to dim x dim
    dim = 1024
    label_path = 'data/MS_CXR_Local_Alignment_v1.0.0.csv'
    image_original_path = "data/raw"
    image_resized_path = "downsized"

    # Check if folder exists, if not create it
    if not os.path.exists(image_resized_path):
        os.makedirs(image_resized_path)

    # Go through each image in the original path, resize it and save it to the new path
    image_names = os.listdir(image_original_path)
    for i, image_name in enumerate(image_names):
        # Resize image
        image_path = os.path.join(image_original_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (dim, dim))
        cv2.imwrite(os.path.join(image_resized_path, image_name), image)

        if i % 100 == 0:
            print("Finished {} / {} images".format(i, len(image_names)))

    # Update gt bbox
    df = pd.read_csv(label_path)
    for index, row in df.iterrows():
        # Read
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        width = row['image_width']
        height = row['image_height']
        
        # Update
        x = int(x * dim / width)
        y = int(y * dim / height)
        w = int(w * dim / width)
        h = int(h * dim / height)

        # Write
        df.at[index, 'x'] = x
        df.at[index, 'y'] = y
        df.at[index, 'w'] = w
        df.at[index, 'h'] = h
        df.at[index, 'image_width'] = dim
        df.at[index, 'image_height'] = dim

    # Save new csv file
    df.to_csv('label_{}.csv'.format(dim), index=False)


if __name__ == "__main__":
    data_resize()
    print("data preprocessing success")