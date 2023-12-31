import sys
import os

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../..", "model"))

import argparse
import yaml
import wandb
import torch
import torch.optim as optim

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from trainer.model import ImageTextModel
from trainer.dataset_mscxr import get_mscxr_dataloader
from trainer.eval import evaluate

    
def run_experiment(config):
    # If sweep is true, perform a sweep
    if config["sweep"]:
        sweep_configuration = {
            "method": "random",
            "metric": {
                "name": "val/acc",
                "goal": "maximize"
            },
            "parameters": {
                "lr": {
                    "distribution": "log_uniform",
                    "min": -11,
                    "max": -6
                },
                "batch_size": {
                    "values": [16, 32]
                }
            },
            "count": 20
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project='AC215-RadIQ')
        wandb.agent(sweep_id, function=lambda:train(config), count=sweep_configuration["count"]) 
    else:
        train(config) # Train without sweeping
    

def train(config):
    # Intialize wandb
    if config['log_to_wandb']:
        wandb.login(key="6ac94bce286531b3989581a1c8c85cb014a32883")
        run = wandb.init(
            project="AC215-RadIQ"
        )
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
    train_loader = get_mscxr_dataloader("train", config["batch_size"], config['resize'], device)
    val_loader = get_mscxr_dataloader("val", config['batch_size'], config['resize'], device)

    # Load BioViL Model
    text_inference = get_bert_inference(BertEncoderType.CXR_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL)
    model = ImageTextModel(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
        width=config['resize'],
        height=config['resize'],
    )

    # Load checkpoint if exists
    if config['img_ckpt']:
        model.image_inference_engine.load_state_dict(torch.load(config['img_ckpt']))
    if config['txt_ckpt']:
        model.text_inference_engine.model.load_state_dict(torch.load(config['txt_ckpt']))

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    opt_params = list(model.text_inference_engine.model.parameters()) + list(model.image_inference_engine.parameters())
    optimizer = optim.Adam(opt_params, lr=config['lr'])

    # Train
    for epoch in range(1, config['epochs']+1):

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
            masks = torch.zeros_like(similarity_map)
            for i in range(images.shape[0]):
                row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()
                masks[i][row_x : row_x + row_w, row_y : row_y + row_h] = 1
            
            # Calculate loss
            loss = criterion(similarity_map.unsqueeze(1), masks.unsqueeze(1))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if config['log_to_wandb']:
                wandb.log({f"train/loss": loss})

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch}/{config['epochs']}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
                )
        
        # Validation
        train_dice = evaluate(model, train_loader, config['threshold'], device)
        val_dice = evaluate(model, val_loader, config['threshold'], device)

        if config['log_to_wandb']:
            # Log
            wandb.log({"train/acc": train_dice, "val/acc": val_dice})
            
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
    config = {
        "log_to_wandb": True,
        "sweep": False,
        "lr": 0.0001,
        "batch_size": 16,
        "epochs": 5,
        "architecture": "biovil",
        "resize": 512,
        "threshold": 0.2,
        "img_ckpt": "",
        "txt_ckpt": ""
    }
    run_experiment(config)