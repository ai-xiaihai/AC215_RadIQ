import sys
import os
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
    train_loader = get_mscxr_dataloader("train", config["batch_size"], device) # TODO
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
    criterion = torch.nn.SmoothL1Loss()
    opt_params = list(model.text_inference_engine.model.parameters()) + list(model.image_inference_engine.parameters()) + list(model.box_head.parameters())
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

            # Forward pass
            pred_boxes = model.get_bbox_from_raw_data(
                images=images,
                query_text=text_prompt,
            )

            # Calculate loss
            loss = criterion(pred_boxes, ground_truth_boxes)
            miou = get_iou(pred_boxes, ground_truth_boxes).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if config['log_to_wandb']:
                wandb.log({
                    f"train/loss": loss,
                    f"train/miou": miou,
                    f"train/gt_box": ground_truth_boxes[0].tolist(),
                    f"train/pred_box": pred_boxes[0].tolist(),
                })

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch}/{config['epochs']}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}, mIoU: {miou.item()}"
                )
                
        
        # Validation
        train_iou = evaluate(model, train_loader, device)
        val_iou = evaluate(model, val_loader, device)

        if config['log_to_wandb']:
            # Log
            wandb.log({"train/iou": train_iou, "val/iou": val_iou})
            
            # Save image encoder
            torch.save(model.image_inference_engine.state_dict(), f'./ckpts/{config["architecture"]}_image_{epoch}.pth')
            artifact_img = wandb.Artifact(f'image_checkpoints_{run.name}', type='model')
            artifact_img.add_file(f'./ckpts/{config["architecture"]}_image_{epoch}.pth', name=f'{config["architecture"]}_image_{epoch}.pth')
            wandb.log_artifact(artifact_img)
            
            # Save text encoder
            torch.save(model.text_inference_engine.model.state_dict(), f'./ckpts/{config["architecture"]}_text_{epoch}.pth')
            artifact_txt = wandb.Artifact(f'text_checkpoints_{run.name}', type='model')
            artifact_txt.add_file(f'./ckpts/{config["architecture"]}_text_{epoch}.pth', name=f'{config["architecture"]}_text_{epoch}.pth')
            wandb.log_artifact(artifact_txt)

    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    run_experiment("config.yaml")