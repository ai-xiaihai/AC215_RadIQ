import sys
sys.path.append('/Users/wangxiaochen/Desktop/AC215/AC215_RadIQ/model')

from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.data.io import load_image

from model import ImageTextModel


# Load BioViL Model
text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

model = ImageTextModel(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
    width=1024,
    height=1024,
)

class MSCXR(Dataset):
    def __init__(self, image_dir, label_file, split, device, transform):
        self.image_dir = os.path.join(image_dir, split)
        df = pd.read_csv(label_file)
        self.dataframe = df[df['split'] == split]
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text_prompt = row.label_text
        ground_truth_boxes = torch.tensor([row.x, row.y, row.w, row.h])

        image_path = Path(f"{self.image_dir}/{row.dicom_id}.jpg")
        image = load_image(image_path)
        transformed_image = self.transform(image)

        return transformed_image, text_prompt, ground_truth_boxes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
image_dir = "./../../radiq-app-data"
label_file = "./../../radiq-app-data/label_1024_split.csv"

train_dataset = MSCXR(image_dir, label_file, 'train', device, image_inference.transform)
# train_dataset = MSCXR(image_dir, label_file, 'val', device, image_inference.transform)
# train_dataset = MSCXR(image_dir, label_file, 'test', device, image_inference.transform)

batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



criterion = nn.MSELoss()
opt_params = list(model.text_inference_engine.model.parameters()) + list(model.image_model.parameters())
optimizer = optim.Adam(opt_params, lr=0.001)

n_epochs = 10
# model.to(device)

for epoch in range(n_epochs):

    for batch_idx, (images, text_prompt, ground_truth_boxes) in enumerate(train_loader):
        loss = 0

        similarity_map = model.get_similarity_maps_from_raw_data(
            images=images,
            query_text=text_prompt,
            interpolation="bilinear",
        ).clip(0)
        assert similarity_map.shape[1] == 1024
        assert similarity_map.shape[2] == 1024

        tmp_batch_size = images.shape[0]

        for i in range(tmp_batch_size):
            row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()

            # Calculate the sum within the box
            sum_val = torch.sum(
                similarity_map[i][row_x : row_x + row_w, row_y : row_y + row_h]
            )
            loss -= sum_val / torch.sum(similarity_map[i]) / tmp_batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
            )
