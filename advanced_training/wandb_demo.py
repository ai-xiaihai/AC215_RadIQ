import random
import wandb
import torch
from torch import nn

# Create a MLP model with 2 linear layers
class MLP(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        if ckpt_path:
            self.load_state_dict(torch.load(ckpt_path))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_demo(hyperparam, model, train_loader, val_loader):
    # Load hyperparams at the beginning of training script
    architecture = hyperparam["architecture"]
    epochs = hyperparam["epochs"]
    lr = hyperparam["lr"]
    batch_size = hyperparam["batch_size"]
    log_to_wandb = hyperparam["log_to_wandb"]
    
    # Intialize wandb
    if log_to_wandb:
        run = wandb.init(
                project="AC215-RadIQ",
                config={
                    "architecture": architecture,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )

    # Simulate training
    offset = random.random() / 5
    for epoch in range(2, epochs):
        # Training loop
        for batch_num in range(800):
            loss = 2 ** -epoch + random.random() / epoch + offset
            if log_to_wandb:
                wandb.log({"train/loss": loss})

        # Evaluation
        train_acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        val_acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        if log_to_wandb:
            # Log accuracy metrics
            wandb.log({"train/acc": train_acc, "val/acc": val_acc})
            # Save model checkpoint
            torch.save(model.state_dict(), f'./ckpts/{architecture}_{epoch}.pth')
            artifact = wandb.Artifact(f'model_checkpoints_{run.name}', type='model')
            artifact.add_file(f'ckpts/{architecture}_{epoch}.pth', name=f'{architecture}_{epoch}.pth')
            wandb.log_artifact(artifact)
        
    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    hyperparam = {
        "epochs": 10,
        "lr": 0.01,
        "batch_size": 32,
        "log_to_wandb": True, # set to False to disable logging to wandb
        "architecture": "biovil"
    }
    train_demo(
        hyperparam=hyperparam,
        model=MLP(),
        train_loader=None,
        val_loader=None,
    )