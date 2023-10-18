import torchvision.transforms as transforms
from monai.transforms import LoadImaged

train_transform = [
        LoadImaged(image_only=True),
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
        # transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
    ]

test_transform = [
        LoadImaged(image_only=True),
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
        # transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
    ]
