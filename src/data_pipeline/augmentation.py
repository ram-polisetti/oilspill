from typing import Dict
import albumentations as A

def get_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    """Create augmentation pipeline for SAR imagery.
    
    Args:
        p: Probability of applying each augmentation
        
    Returns:
        Composed augmentation pipeline
    """
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.Rotate(limit=10, p=p),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=p),
        A.GridDistortion(p=p/2),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

def get_validation_pipeline() -> A.Compose:
    """Create validation pipeline with only essential transforms.
    
    Returns:
        Composed validation pipeline
    """
    return A.Compose([
        A.Normalize(mean=[0.485], std=[0.229]),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))