import torch
import torchvision

# from SegformerBundle.scripts.datasets import CustomDataset
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None, mask_is_oned=False, include_original=False, img_shape=(512,512)):
        self.data_list = data_list
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_shape)
        ])
        self.mask_is_oned = mask_is_oned
        self.include_original = include_original

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        image_original = Image.open(sample['image']).convert('RGB')  # Assuming 'image' is the key for image data
        mask = Image.open(sample['label']).convert('RGB')
        # if self.mask_is_oned:
        #     mask = Image.open(sample['label']).convert('L')

        # TODO: Figure out why convert RGB isnt converting to RGB... and therefore I need to do here
        # Worried image_original might have issue then too
        mask = torch.round(self.mask_transform(mask)[[2,1,0], :, :])
        print("MASK: ", mask)
        # Image may be altered
        if self.transform:
            image = self.transform(image_original)
            print("IMAGE: ", image)
        else:
            # or also kept same like mask (just to tensor)
            image = self.mask_transform(image_original)
        
        if self.include_original:
            return image, mask, self.mask_transform(image_original)
        else:
            return image, mask


batch_size = 1

# TODO: fix MednistBundleTrainTask to make proper train_ds (something that CustomDataset can handle?)
def datastore_dataloader(train_ds, val_ds, transform):
    image_label_dict_list = train_ds
    trainset = CustomDataset(image_label_dict_list, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
    return trainloader
