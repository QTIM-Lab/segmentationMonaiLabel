import torchvision.transforms as td_transforms
import monai

train_transform = td_transforms.Compose([
        # LoadImage(image_only=True),
        td_transforms.ToTensor(),
        td_transforms.Resize((512,512)),
        td_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        td_transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
        # td_transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
    ])

test_transform = td_transforms.Compose([
        # monai.transforms.LoadImage(image_only=True),
        td_transforms.ToTensor(),
        td_transforms.Resize((512,512)),
        td_transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
        # td_transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
    ])

# test_transform = [
#     monai.transforms.LoadImaged(keys="image", image_only=True),
#     monai.transforms.EnsureChannelFirstd(keys="image"),
#     monai.transforms.Resized(keys="image", spatial_size=[512,512]),
#     monai.transforms.ScaleIntensityd(keys="image"),
#     monai.transforms.NormalizeIntensityd(keys="image", subtrahend=[0.522, 0.300, 0.167], divisor=[0.240, 0.189, 0.147], channel_wise=True)
# ]

