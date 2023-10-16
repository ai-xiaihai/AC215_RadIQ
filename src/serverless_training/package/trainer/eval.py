
import torch
import torch.nn as nn
import numpy as np

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from model import ImageTextModel
from dataset_mscxr import get_mscxr_dataloader

import torch

def dice(pred_mask, gt_mask, epsilon=1e-6):
    """
    Computes the Dice Coefficient.

    Args:
    - pred_mask (torch.Tensor): Predicted masks converted from similarity map, shape (batch_size, image_height, image_width)
    - gt_mask (torch.Tensor): Ground truth masks, shape (batch_size, image_height, image_width)
    - epsilon (float): Small constant to prevent division by zero

    Returns:
    - dice (torch.Tensor): Dice coefficient for each image in the batch, shape (batch_size,)
    """
    
    # Compute the intersection
    intersection = (pred_mask * gt_mask).sum(dim=(1,2))
    
    # Compute the areas of both the predicted and ground truth masks
    pred_area = pred_mask.sum(dim=(1,2))
    gt_area = gt_mask.sum(dim=(1,2))
    
    # Compute Dice Coefficient for each image in the batch
    dice = (2. * intersection + epsilon) / (pred_area + gt_area + epsilon)

    return dice


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, threshold: int, device):
    """
    Evaluates the model on the given loader and calculates the Dice score.

    Parameters:
    model -- PyTorch model to be evaluated
    loader -- DataLoader for the evaluation dataset

    Returns:
    average_dice -- Average Dice score over all samples in the evaluation dataset
    """

    dice_scores = []

    with torch.no_grad():
        tot_sample, tot_dice = 0, 0
        for batch_idx, data in enumerate(loader):
            # Unpack data
            images = data["image"].to(device)
            text_prompt = data["text"]
            ground_truth_boxes = data["ground_truth_boxes"].to(device)

            similarity_map = model.get_similarity_maps_from_raw_data(
                images=images,
                query_text=text_prompt,
                interpolation="bilinear",
            )

            # Convert similarity map to a binary mask
            pred_masks = (similarity_map > threshold).float()

            # Convert bounding box to a binary mask with same size as similarity map
            masks = torch.zeros_like(similarity_map)
            for i in range(images.shape[0]):
                row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()
                masks[i][row_x : row_x + row_w, row_y : row_y + row_h] = 1

            # Calculate Dice score for each sample in the batch
            cur_dice = dice(pred_masks, masks)
            tot_dice += cur_dice.sum().item()
            tot_sample += cur_dice.size(0)

            if batch_idx % 5 == 0:
                print(
                    f"[Evaluation] Batch {batch_idx}/{len(loader)}, val_dice: {tot_dice/tot_sample}"
                )

        average_dice = tot_dice/tot_sample

    return average_dice


class UnitTest:
    """Unit test for evaluation"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval(self):    
        # Load dataset
        batch_size = 16
        resize = 512
        threshold = 0.3
        val_loader = get_mscxr_dataloader("val", batch_size, resize, self.device)

        # Load BioViL Model
        text_inference = get_bert_inference(BertEncoderType.CXR_BERT)
        image_inference = get_image_inference(ImageModelType.BIOVIL)
        model = ImageTextModel(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
            width=resize,
            height=resize,
        )
        
        val_dice = evaluate(model, val_loader, threshold, self.device)
        print("Test Evaluation: SUCCESS! Validation dice: ", val_dice)
    

if __name__ == "__main__":
    test = UnitTest()
    test.test_eval()