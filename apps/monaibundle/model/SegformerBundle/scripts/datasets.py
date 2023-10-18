from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_is_oned=False, include_original=False, img_shape=(224,224)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_shape)
        ])
        self.mask_is_oned = mask_is_oned
        self.include_original = include_original

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx].split('/projects/Fundus_Segmentation')[1]
        img_path = '/scratch/alpine/skinder@xsede.org/glaucomachris' + img_path
        mask_path = self.mask_paths[idx].split('/projects/Fundus_Segmentation')[1]
        mask_path = '/scratch/alpine/skinder@xsede.org/glaucomachris' + mask_path
        image_original = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        if self.mask_is_oned:
            mask = Image.open(mask_path).convert('L')

        # if self.transform:
        #     image = self.transform(image)
        #     mask = self.transform(mask)

        # if self.mask_is_oned:
        #     return image, mask
        # else:
        #     return image, mask[1:2, :, :]
        # Mask transform just to tensor
        mask = self.mask_transform(mask)
        # Image may be altered
        if self.transform:
            image = self.transform(image_original)
        else:
            # or also kept same like mask (just to tensor)
            image = self.mask_transform(image_original)
        
        if self.include_original:
            return image, mask, self.mask_transform(image_original)
        else:
            return image, mask

class AdvaithCustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_is_oned=False, include_original=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mask_is_oned = mask_is_oned
        self.include_original = include_original

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # img_path = self.image_paths[idx].split('/data/retina_datasets_preprocessed/Dynamic_Cropped/')[1]
        img_path_middle = '/Organized_Datasets/Rim_One_r1_Organized/Images'
        img_path = '/scratch/alpine/skinder@xsede.org/glaucomachris' + img_path_middle + '/' + self.image_paths[idx].split('/projects/Fundus_Segmentation/Organized_Datasets/Rim_One_r1_Organized/Images/')[1].split('.')[0] + '.png'
        image_original = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image_original)
        else:
            image = self.mask_transform(image_original)
        
        if self.include_original:
            return image, image, self.mask_transform(image_original)
        else:
            return image, image
