import sys
import os
import pdb
import wandb
import matplotlib.pyplot as plt
from pathlib import Path

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../", "model"))

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
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


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, threshold=0.0, device="cuda", visualize=False):
    """
    Parameters:
    model (nn.Module): The PyTorch model to be evaluated.
    loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
    threshold (float, optional): Threshold for converting similarity maps to binary masks. Defaults to 0.0.
    device (str, optional): The device on which the model and data are placed. Defaults to "cuda".
    visualize (bool, optional): Whether to generate visualizations of grounding effects. Defaults to False.

    Returns:
    average_dice (float) -- Average Dice score over all samples in the evaluation dataset
    """

    with torch.no_grad():
        tot_sample, tot_dice = 0, 0
        for batch_idx, data in enumerate(loader):
            # Unpack data
            images = data["image"].to(device)
            text_prompt = data["text"]
            ground_truth_boxes = data["ground_truth_boxes"].to(device)
            dicom_id = data["dicom_id"]

            similarity_map = model.get_bbox_from_raw_data(
                images=images,
                query_text=text_prompt
            )

            similarity_map = torch.sigmoid(similarity_map)

            # Convert similarity map to a binary mask
            pred_masks = (similarity_map > threshold).float()

            # Convert bounding box to a binary mask with same size as similarity map
            masks = torch.zeros_like(similarity_map)
            for i in range(images.shape[0]):
                row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()
                masks[i][row_y : row_y + row_h, row_x : row_x + row_w] = 1

            # Calculate Dice score for each sample in the batch
            cur_dice = dice(pred_masks, masks)
            tot_dice += cur_dice.sum().item()
            tot_sample += cur_dice.size(0)

            if batch_idx % 5 == 0:
                print(
                    f"[Evaluation] Batch {batch_idx}/{len(loader)}, val_dice: {tot_dice/tot_sample}"
                )

            if visualize:
                path = Path(f"../../../../radiq-app-data/ms_cxr/val/" + dicom_id[0])
                fig = plot_phrase_grounding_similarity_map(
                    path, 
                    similarity_map[0].detach().cpu().numpy(), 
                    text_prompt[0],
                    cur_dice[0].item(),
                    [ground_truth_boxes[0].detach().cpu().numpy().tolist()]
                )

                save_directory = "../../../../radiq-app-data/visualize"
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                plt.savefig(f"{save_directory}/{dicom_id[0]}")
                
                plt.close()

        average_dice = tot_dice/tot_sample

    return average_dice


class UnitTest:
    """Unit test for evaluation"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval_thresholds(self, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
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

        # Load checkpoint
        model.box_head.load_state_dict(
            torch.load("./best_box_head.pth", map_location=self.device)
        )

        # Evaluate
        result = {}
        for thres in thresholds:
            val_dice = evaluate(model, val_loader, threshold=thres, device=self.device)
            print(f"Threshold: {thres}, Validation dice: {val_dice}")
            result[thres] = val_dice
        
        print(result)


    def test_eval(self, visualize=False):    
        # Load dataset
        batch_size = 16
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

        if visualize:
            # load best archecture from WandB
            wandb.login()
            run = wandb.init(project='AC215-RadIQ')

            best_sweep = 'likely-sweep-4'
            best_epoch = 3 # 0-index
            artifact_name = f'ac215-radiq/AC215-RadIQ/box_head_checkpoints_{best_sweep}:v{best_epoch}' 
            artifact = run.use_artifact(artifact_name)
            artifact_dir = artifact.download()

            box_head_state_dict = torch.load(f"{artifact_dir}/biovil_box_head_{best_epoch + 1}.pth", map_location=self.device)
            model.box_head.load_state_dict(box_head_state_dict)

            val_dice = evaluate(model, val_loader, device=self.device, visualize=True)
        else:
            val_dice = evaluate(model, val_loader, device=self.device, visualize=False)

        val_dice = evaluate(model, val_loader, device=self.device)
        print("Test Evaluation: SUCCESS! Validation dice: ", val_dice)


    def test_miou(self):
        pred_boxes = torch.tensor([[50, 50, 100, 100], [30, 30, 90, 90]]).to(self.device)
        gt_boxes = torch.tensor([[100, 100, 100, 100], [40, 40, 90, 90]]).to(self.device)
        print(get_iou(pred_boxes, gt_boxes))
    


if __name__ == "__main__":
    test = UnitTest()
    # test.test_miou()
    # test.test_eval(visualize=True)
    test.test_eval_thresholds()
