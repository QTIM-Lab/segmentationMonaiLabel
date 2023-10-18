import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from monailabel.interfaces.datastore import Datastore

from typing import List, Dict, Any

from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], transform=None):
        self.data_list = data_list
        self.transform = transform
        self.label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]

        image = Image.open(sample['image'])  # Assuming 'image' is the key for image data
        label = sample['label']  # Assuming 'label' is the key for label data
        # Open the JSON file in read mode.
        with open(label, 'r') as file:
            # Load the JSON data from the file and store it in a Python dictionary.
            label_json = json.load(file)
            label_value = self.label_names.index(label_json['annotation_label'])
            print("LABEL VALUE: ", label_value)
        if self.transform:
            image = self.transform(image)

        return image, label_value


batch_size = 1

def get_dataloader(is_training, transform):
    batch_size = 4
    if is_training:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        return trainloader
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        return testloader
    

# TODO: fix MednistBundleTrainTask to make proper train_ds (something that CustomDataset can handle?)
def datastore_dataloader(train_ds, val_ds, transform):
    image_label_dict_list = train_ds
    trainset = CustomDataset(image_label_dict_list, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
    return trainloader
