import sys
import os

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../..", "model"))

import argparse
import wandb
import torch
import torch.optim as optim

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from model import ImageTextModel
from dataset_mscxr import get_mscxr_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Data split to use')
    parser.add_argument('--epochs', type=int, default=5, help='Data split to use')
    parser.add_argument('--log_to_wandb', type=bool, default=False, help='Flag to log results to wandb')
    parser.add_argument('--architecture', type=str, default='BioViL', help='model architecture')
    return parser.parse_args()
    

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize = 512

    # Load dataset
    train_loader = get_mscxr_dataloader("train", args.batch_size, resize, device)
    val_loader = get_mscxr_dataloader("val", args.batch_size, resize, device)

    # Load BioViL Model
    text_inference = get_bert_inference(BertEncoderType.CXR_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL)

    model = ImageTextModel(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
        width=resize,
        height=resize,
    )

    # Define optimizer
    opt_params = list(model.text_inference_engine.model.parameters()) + list(model.image_inference_engine.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr)

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

    # Train
    for epoch in range(args.epochs):

        for batch_idx, data in enumerate(train_loader):
            # Unpack data
            images = data["image"].to(device)
            text_prompt = data["text"]
            ground_truth_boxes = data["ground_truth_boxes"].to(device)

            # Forward run
            similarity_map = model.get_similarity_maps_from_raw_data(
                images=images,
                query_text=text_prompt,
                interpolation="bilinear",
            )

            # Generate masks
            masks = -1 * torch.ones_like(similarity_map)
            for i in range(images.shape[0]):
                row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()
                masks[i][row_x : row_x + row_w, row_y : row_y + row_h] = 1
            
            # Calculate loss
            loss = -torch.sum(similarity_map * masks) / images.shape[0]

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if args.log_to_wandb:
                wandb.log({f"{args.split}/loss": loss})

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
                )
        
        # Validation #TODO
        train_iou = 0
        val_iou = 0

        if args.log_to_wandb:
            wandb.log({"train/acc": train_iou, "val/acc": val_iou})
            torch.save(model.state_dict(), f'./ckpts/{args.architecture}_{epoch}.pth')
            artifact = wandb.Artifact(f'model_checkpoints_{run.name}', type='model')
            artifact.add_file(f'ckpts/{args.architecture}_{epoch}.pth', name=f'{args.architecture}_{epoch}.pth')
            wandb.log_artifact(artifact)

    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    main()