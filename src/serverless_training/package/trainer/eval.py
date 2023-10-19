import sys
import os
import pdb

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../..", "model"))

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from model import ImageTextModel
from dataset_mscxr import get_mscxr_dataloader


def xywh_to_x1y1x2y2(boxes):
    """Convert boxes in (x, y, w, h) format to (x1, y1, x2, y2) format."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def get_iou(pred_boxes, gt_boxes):
    """
    Computes the IoU using PyTorch's built-in function.
    pred_boxes: tensor of size (B, 4) in (x, y, w, h) format
    gt_boxes: tensor of size (B, 4) in (x, y, w, h) format
    """
    pred_boxes_x1y1x2y2 = xywh_to_x1y1x2y2(pred_boxes)
    gt_boxes_x1y1x2y2 = xywh_to_x1y1x2y2(gt_boxes)
    
    iou = box_iou(pred_boxes_x1y1x2y2, gt_boxes_x1y1x2y2).diag()
    
    return iou


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


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device):
    """
    Evaluates the model on the given loader and calculates the Dice score.

    Parameters:
    model -- PyTorch model to be evaluated
    loader -- DataLoader for the evaluation dataset

    Returns:
    average_dice -- Average Dice score over all samples in the evaluation dataset
    """
    tot_sample, tot_iou = 0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
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
            ious = get_iou(pred_boxes, ground_truth_boxes)

            # Calculate iou score for each sample in the batch
            tot_iou += ious.sum().item()
            tot_sample += ious.size(0)

            if batch_idx % 5 == 0:
                print(
                    f"[Evaluation] Batch {batch_idx}/{len(loader)}, val_dice: {tot_iou/tot_sample}"
                )

        average_dice = tot_iou/tot_sample

    return average_dice


class UnitTest:
    """Unit test for evaluation"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval(self):    
        # Load dataset
        batch_size = 32
        val_loader = get_mscxr_dataloader("val", batch_size, self.device)

        # Load BioViL Model
        text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        image_inference = get_image_inference(ImageModelType.BIOVIL_T)
        model = ImageTextModel(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
            width=1024,
            height=1024,
        )
        model.to(self.device)
        val_dice = evaluate(model, val_loader, self.device)
        print("Test Evaluation: SUCCESS! Validation iou: ", val_dice)

    def test_miou(self):
        pred_boxes = torch.tensor([[50, 50, 100, 100], [30, 30, 90, 90]]).to(self.device)
        gt_boxes = torch.tensor([[100, 100, 100, 100], [40, 40, 90, 90]]).to(self.device)
        print(get_iou(pred_boxes, gt_boxes))
    

if __name__ == "__main__":
    test = UnitTest()
    test.test_miou()
    test.test_eval()