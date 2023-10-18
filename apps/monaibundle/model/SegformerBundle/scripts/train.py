import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import os
import time
import cv2
import csv
import numpy as np
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url

from datasets import CustomDataset
from models import ModifiedDeepLabV3, UNet
from losses import DiceLoss

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import SimpleITK as sitk

def dice_coefficient_torch(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (torch.sum(y_true) + torch.sum(y_pred))
    return dice

# Calculate pairwise Dice coefficient (Dpw)
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

# Function to find maximum diameter and draw the line
def find_and_draw_max_diameter(contour, image, color_tuple):
    max_diameter = 0
    max_diameter_points = None
    # print("CONTOUR: ", contour)
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            dist = np.linalg.norm(contour[i][0] - contour[j][0])  # Euclidean distance
            # print("DIST: ", dist)
            if dist > max_diameter:
                max_diameter = dist
                max_diameter_points = (tuple(contour[i][0]), tuple(contour[j][0]))

    if max_diameter_points is not None:
        cv2.line(image, max_diameter_points[0], max_diameter_points[1], color_tuple, 2)

    return max_diameter


def train(model):
    dropout_prob = 0.1

    batch_size = 8
    num_epochs = 40
    learning_rate = 0.00001

    # Train can be augmented, and then normalized under train statistics
    # train_transform = transforms.Compose([
    #     # transforms.RandomVerticalFlip(),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
    #     # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
    #     # transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
    # ])

    # Load the CSV file
    csv_file_path = '/scratch/alpine/skinder@xsede.org/glaucomachris/Cropped_Three_Seg_Dri_Riga_RDL_Ref_R3_Spec_Removed.csv'  # Replace with the actual path
    data_df = pd.read_csv(csv_file_path)

    train_image_paths = data_df['Cropped_Image'].tolist()  # Replace with the actual column name
    train_mask_paths = data_df['Cropped_Three'].tolist()  # Replace with the actual column name


    # Datasets and dataloaders
    train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_batches = len(train_dataloader)

    savedir = '/scratch/alpine/skinder@xsede.org/chrisglaucoma/segmentation_results_' + experiment_name
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    train_losses = []

    best_model_state = None

    # define model
    pretrained_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    # Get the model's configuration
    config = pretrained_model.config
    # Modify the configuration to add dropout
    config.hidden_dropout_prob = dropout_prob
    config.attention_probs_dropout_prob = dropout_prob
    config.classifier_dropout_prob = dropout_prob
    config.drop_path_rate = dropout_prob

    # Initialize a new model with the modified configuration
    # model = SegformerForSemanticSegmentation(config=config)
    # model.load_state_dict(pretrained_model.state_dict())
    # model.decode_head.classifier = nn.Conv2d(768, 3, kernel_size=1)

    # print(model)
    # model.to(device)

    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0  # Track total loss for the epoch
        
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.logits
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            # Calculate loss
            # loss = (dice_loss(outputs['aux'], masks) * aux_weight) + dice_loss(outputs['out'], masks)
            # loss = (criterion(outputs['aux'], masks) * aux_weight) + criterion(outputs['out'], masks)
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{num_batches}] - Loss: {loss.item():.4f}")
        
        average_loss = total_loss / num_batches
        train_losses.append(average_loss)

        # Step the scheduler to update the learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {average_loss:.4f}")

        # only save on last iter
        if epoch == num_epochs - 1:
            best_model_state = model.state_dict()
            checkpoint_path = os.path.join(savedir, 'best_model_bce.pth')
            torch.save(best_model_state, checkpoint_path)
            print("Best model saved!")

    print("Training ended, producing Loss Curve graph to: ", savedir + 'loss_curves.png')

    plt.figure()
    plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(savedir, 'loss_curves.png'))
    plt.close()
    print("Training over, fig saved, evaling")
    print("All done!")


