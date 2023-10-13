import sys
import os

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../..", "model"))

import argparse
import wandb
from typing import List
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from google.cloud import storage
from io import BytesIO
from PIL import Image
import numpy as np

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.io import remap_to_uint8
from model import ImageTextModel

from google.cloud import storage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Data split to use')
    parser.add_argument('--epochs', type=int, default=5, help='Data split to use')
    parser.add_argument('--log_to_wandb', type=bool, default=True, help='Flag to log results to wandb')
    parser.add_argument('--architecture', type=str, default='BioViL', help='model architecture')
    return parser.parse_args()


class MSCXR(Dataset):
    def __init__(self, bucket_name, label_file, split, device, transform):
        # Set up GCS bucket
        self.bucket = storage.Client().bucket(bucket_name)
        self.prefix = os.path.join("ms_cxr", split)

        # Load label file
        blob = storage.Blob(label_file, self.bucket)
        data = blob.download_as_text()
        df = pd.read_csv(BytesIO(data.encode()))
        self.dataframe = df[df["split"] == split]
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get label
        row = self.dataframe.iloc[idx]
        text_prompt = row.label_text
        ground_truth_boxes = torch.tensor([row.x, row.y, row.w, row.h])

        # Get image
        image_blob = storage.Blob(f"{self.prefix}/{row.dicom_id}.jpg", self.bucket)
        image_data = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_data))
        image = np.array(image)
        image = remap_to_uint8(image)
        image = Image.fromarray(image).convert("L")
        transformed_image = self.transform(image)

        return transformed_image, text_prompt, ground_truth_boxes
    

def main():
    args = parse_args()

    # Load BioViL Model
    text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL_T)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bucket_name = "radiq-app-data"
    label_file = "ms_cxr/label_1024_split.csv"
    train_dataset = MSCXR(bucket_name, label_file, "train", device, image_inference.transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ImageTextModel(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
        width=1024,
        height=1024,
    )

    opt_params = list(model.text_inference_engine.model.parameters()) + list(model.image_model.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr)
    # model.to(device)

    # Intialize wandb
    if args.log_to_wandb:
        run = wandb.init(
                project="AC215-RadIQ",
                config={
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "architecture": args.architecture
                }
            )

    for epoch in range(args.epochs):

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

            if args.log_to_wandb:
                wandb.log({f"{args.split}/loss": loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
                )

        if args.log_to_wandb:
            # wandb.log({"train/acc": train_acc, "val/acc": val_acc})
            torch.save(model.state_dict(), f'./ckpts/{args.architecture}_{epoch}.pth')
            artifact = wandb.Artifact(f'model_checkpoints_{run.name}', type='model')
            artifact.add_file(f'ckpts/{args.architecture}_{epoch}.pth', name=f'{args.architecture}_{epoch}.pth')
            wandb.log_artifact(artifact)

    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    main()