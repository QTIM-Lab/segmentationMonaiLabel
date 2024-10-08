import os, pdb
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class MedSamTransform(transforms.Compose):
    def __init__(self, is_grayscale=False, resize=(256, 256)):
        # print("!!!In MedSamTransform!!!\n")
        # pdb.set_trace()
        self.is_grayscale = is_grayscale
        self.resize = resize
        transformations = [
            transforms.Resize(self.resize),
            # transforms.Grayscale() if is_grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(), # it will convert to tensor and scale between 0-1
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            # transforms.Lambda(lambda x: torch.from_numpy(np.array(x))),
        ] 
        
        super().__init__(transformations)

    def __call__(self, image):
        image = image.convert('RGB')
        return super().__call__(image)