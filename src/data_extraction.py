import pandas as pd
import shutil
import os


def data_extraction():
    """Extracts images from MIMIC-CXR dataset according to the MS-CXR label file.
    """
    mimic_folder_path = "/home/mam0364/lab/datasets/cxr/MIMIC-CXR/2.0.0"
    label_path = 'MS_CXR_Local_Alignment_v1.0.0.csv'

    # Load CSV file
    df = pd.read_csv(label_path)

    # Check if folder exists, if not create it
    if not os.path.exists('images'):
        os.makedirs('images')

    # Go through each row and copy image to new folder according to image path
    for index, row in df.iterrows():
        relative_path = row['path']
        image_name = relative_path.split('/')[-1]
        abs_path = os.path.join(mimic_folder_path, relative_path)
        shutil.copy(abs_path, 'images/' + image_name)
        
        if index % 100 == 0:
            print("Finished {} / {} images".format(index, len(df)))


if __name__ == "__main__":
    data_extraction()
    print("data extraction success")