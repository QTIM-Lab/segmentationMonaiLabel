import pdb
import torch
import torch.nn as nn

# BB
import numpy as np
from transformers import SamModel, SamProcessor
from PIL import Image

class MedSamNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the MedSAM model
        self._model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base", local_files_only=False)
        self._processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    def forward(self, dataset_instance):
        # model.to(device)
        # batch = next(iter(dataset_instance))
        # dataset_instance[0]["pixel_values"].shape
        # dataset_instance["input_boxes"]
        # forward pass
        # outputs = model(pixel_values=dataset_instance["pixel_values"].to(device),
        #                 input_boxes=dataset_instance["input_boxes"].to(device),
        #                 multimask_output=False)
        pdb.set_trace()
        device='cuda:0'
        self._model.to(device)
        outputs = self._model(pixel_values=dataset_instance[0]["pixel_values"].to(device), input_boxes=dataset_instance[0]["input_boxes"].to(device), multimask_output=False)
        outputs = self._model(dataset_instance[0], multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        return predicted_masks

