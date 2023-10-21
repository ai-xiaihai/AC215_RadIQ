import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pdb

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../..", "model"))

import yaml
import wandb
import torch
import torch.optim as optim
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from model import ImageTextModel
from dataset_mscxr import get_mscxr_dataloader
from eval import evaluate, dice, get_iou

    
def run_experiment(config_path):
    """Wrapper on top of train(), determine whether to hyperparameter sweep or simply training."""
    # Load configuartions
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # If sweep is true, perform a sweep
    if config["sweep"]:
        with open("sweep_config.yaml", 'r') as file:
            sweep_configuration = yaml.safe_load(file)
    
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='AC215-RadIQ')
        wandb.agent(sweep_id, function=lambda:train(config), count=sweep_configuration["count"]) 
    else:
        train(config) # Train without sweeping
    

def train(config):
    """Training script for BioViL model."""
    # Intialize wandb
    if config['log_to_wandb']:
        run = wandb.init(project="AC215-RadIQ")
        run.config.epochs = config['epochs']
        run.config.architecture = config["architecture"]

        if config['sweep']:
            run.config.learning_rate = wandb.config['lr']
            run.config.batch_size = wandb.config['batch_size']
            
        else:
            run.config.learning_rate = config['lr']
            run.config.batch_size = config['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = get_mscxr_dataloader("train", config["batch_size"], device)
    val_loader = get_mscxr_dataloader("val", config['batch_size'], device)

    # Load BioViL Model
    text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL_T)
    model = ImageTextModel(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
        width=1024,
        height=1024,
    )
    model.to(device)

    # Load checkpoint if exists
    if config['img_ckpt']:
        model.image_inference_engine.load_state_dict(torch.load(config['img_ckpt']))
    if config['txt_ckpt']:
        model.text_inference_engine.model.load_state_dict(torch.load(config['txt_ckpt']))

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    sig = torch.nn.Sigmoid()
    opt_params = list(model.box_head.parameters())
    optimizer = optim.Adam(opt_params, lr=config['lr'])

    # Set training mode
    model.text_inference_engine.model.eval()
    model.image_inference_engine.eval()
    model.box_head.train()

    # Train
    for epoch in range(1, config['epochs']+1):

        for batch_idx, data in enumerate(train_loader):
            # Unpack data
            images = data["image"].to(device)
            text_prompt = data["text"]
            ground_truth_boxes = data["ground_truth_boxes"].to(device)
            dicom_id = data["dicom_id"]

            # Forward pass
            pred_mask = model.get_bbox_from_raw_data(
                images=images,
                query_text=text_prompt,
            )
            pred_mask = sig(pred_mask)

            # Visualize
            path = Path(f"../../../../radiq-app-data/train/" + dicom_id[0])
            fig = plot_phrase_grounding_similarity_map(
                path, 
                pred_mask[0].detach().cpu().numpy(), 
                [ground_truth_boxes[0].cpu().numpy().tolist()]
            )
            fig.savefig(f"test_{epoch}_{batch_idx}.png")

            # Generate gt masks
            masks = torch.zeros_like(pred_mask)
            for i in range(images.shape[0]):
                row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()
                masks[i][row_y : row_y + row_h, row_x : row_x + row_w] = 1

            # Generate background masks
            background_masks = torch.ones_like(masks) - masks
            masks_all = torch.stack([background_masks, masks], dim=1)
            pred_mask_background = torch.ones_like(pred_mask) - pred_mask
            pred_mask_all = torch.stack([pred_mask_background, pred_mask], dim=1)

            # Calculate loss
            bce = criterion(pred_mask_all, masks_all)
            dice_score = dice(masks, pred_mask > 0.5).mean()
            loss = bce

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if config['log_to_wandb']:
                wandb.log({
                    f"train/bce": bce,
                    f"train/miou": dice_score,
                    f"train/loss": loss,
                })

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch}/{config['epochs']}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}, dice: {dice_score.item()}"
                )
                
        
        # Validation
        train_dice = evaluate(model, train_loader, config["threshold"], device)
        val_dice = evaluate(model, val_loader, config["threshold"], device)

        if config['log_to_wandb']:
            # Log
            wandb.log({"train/dice": train_dice, "val/dice": val_dice})
            
            # # Save image encoder
            # torch.save(model.image_inference_engine.state_dict(), f'./ckpts/{config["architecture"]}_image_{epoch}.pth')
            # artifact_img = wandb.Artifact(f'image_checkpoints_{run.name}', type='model')
            # artifact_img.add_file(f'./ckpts/{config["architecture"]}_image_{epoch}.pth', name=f'{config["architecture"]}_image_{epoch}.pth')
            # wandb.log_artifact(artifact_img)
            
            # # Save text encoder
            # torch.save(model.text_inference_engine.model.state_dict(), f'./ckpts/{config["architecture"]}_text_{epoch}.pth')
            # artifact_txt = wandb.Artifact(f'text_checkpoints_{run.name}', type='model')
            # artifact_txt.add_file(f'./ckpts/{config["architecture"]}_text_{epoch}.pth', name=f'{config["architecture"]}_text_{epoch}.pth')
            # wandb.log_artifact(artifact_txt)

            # Save box/segmentation head
            torch.save(model.box_head.state_dict(), f'./ckpts/{config["architecture"]}_box_head_{epoch}.pth')
            artifact_box = wandb.Artifact(f'box_head_checkpoints_{run.name}', type='model')
            artifact_box.add_file(f'./ckpts/{config["architecture"]}_box_head_{epoch}.pth', name=f'{config["architecture"]}_box_head_{epoch}.pth')
            wandb.log_artifact(artifact_box)

    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    run_experiment("config.yaml")