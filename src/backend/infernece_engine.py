import sys
import os
import wandb
import pdb
import numpy as np
from PIL import Image

# Add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../", "model"))

import torch
from torchvision.ops import box_iou

from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.common.visualization import overlay_similarity_map
from model import ImageTextModel
from dataset_mscxr import create_chest_xray_transform_for_inference, remap_to_uint8


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


class InferenceEngine:
    """Inference engine for BioViL model."""

    def __init__(self):
        """Initialization
        
        Load BioViL model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.download_model()
        model.box_head.load_state_dict(
            torch.load(self.ckpt_path, map_location=self.device)
        )
        self.model = model

    
    def download_model(self):
        """Download model from wandb."""
        # Specify which checkpoint to get
        best_sweep = 'likely-sweep-4'
        best_epoch = 3 # 0-index
        artifact_name = f'ac215-radiq/AC215-RadIQ/box_head_checkpoints_{best_sweep}:v{best_epoch}' 

        # Download model from wandb
        wandb.login()
        run = wandb.init(project='AC215-RadIQ')
        artifact = run.use_artifact(artifact_name)
        artifact_dir = artifact.download()
        wandb.finish()

        # Check ckpt path
        self.ckpt_path = f"{artifact_dir}/biovil_box_head_{best_epoch + 1}.pth"


    def preprocess(self, image, dim=1024):
        """Preprocess image."""
        # Resize
        image = image.resize((dim, dim))

        # Load image
        image = np.array(image)
        image = remap_to_uint8(image)
        image = Image.fromarray(image).convert("L")

        # Transform
        image = create_chest_xray_transform_for_inference()(image)
        image = image.unsqueeze(0)
        return image


    def inference(self, image, text_prompt, threshold=0.8):
        """Inference on a single image and text prompt."""
        # Preprocess
        transformed_image = self.preprocess(image)
        transformed_image = transformed_image.to(self.device)

        # Get similarity map
        with torch.no_grad():
            similarity_map = self.model.get_bbox_from_raw_data(
                images=transformed_image,
                query_text=text_prompt
            )
            similarity_map = torch.sigmoid(similarity_map)

            # Scale similarity_map to orignal image size
            similarity_map = torch.nn.functional.interpolate(
                similarity_map.unsqueeze(0),
                size=(image.size[1], image.size[0]), 
                mode="bilinear", 
                align_corners=False
            )
            similarity_map = similarity_map.squeeze()


        # Overlay on the original image
        fig = overlay_similarity_map(image, similarity_map.cpu().numpy())
        fig.savefig("overlay.png")
        return fig


if __name__ == "__main__":
    engine = InferenceEngine()
    image = Image.open("chest_xray.jpg")
    engine.inference(image, "cardiomegaly")
