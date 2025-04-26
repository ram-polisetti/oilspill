import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import ZENONODODataset, get_augmentation_pipeline
from models.hybrid_model import HybridModel, HybridLoss
from training.trainer import Trainer
from configs.config import get_training_config, get_model_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train oil spill detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the ZENONODO dataset')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the wandb experiment')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
    
    # Validate dataset structure
    required_dirs = ['train/images', 'train/masks', 'train/labels',
                    'val/images', 'val/masks', 'val/labels']
    for dir_path in required_dirs:
        full_path = os.path.join(args.data_dir, dir_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Required directory {full_path} not found")
    
    # Load configurations
    train_config = get_training_config()
    model_config = get_model_config()
    
    # Setup data paths
    data_dir = Path(args.data_dir)
    train_images = sorted(list((data_dir / 'train' / 'images').glob('*.tif')))
    train_masks = sorted(list((data_dir / 'train' / 'masks').glob('*.png')))
    train_bboxes = sorted(list((data_dir / 'train' / 'labels').glob('*.txt')))
    
    val_images = sorted(list((data_dir / 'val' / 'images').glob('*.tif')))
    val_masks = sorted(list((data_dir / 'val' / 'masks').glob('*.png')))
    val_bboxes = sorted(list((data_dir / 'val' / 'labels').glob('*.txt')))
    
    # Create datasets
    train_transform = get_augmentation_pipeline(p=train_config['augmentation_prob'])
    train_dataset = ZENONODODataset(
        image_paths=train_images,
        mask_paths=train_masks,
        bbox_paths=train_bboxes,
        transform=train_transform,
        image_size=train_config['image_size']
    )
    
    val_dataset = ZENONODODataset(
        image_paths=val_images,
        mask_paths=val_masks,
        bbox_paths=val_bboxes,
        transform=None,
        image_size=train_config['image_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    # Initialize model and loss with error handling
    try:
        model = HybridModel(pretrained=model_config['use_pretrained'])
        criterion = HybridLoss(
            det_weight=train_config['det_weight'],
            seg_weight=train_config['seg_weight']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model or loss: {str(e)}")
        
    # Validate model configuration
    if not all(k in train_config for k in ['det_weight', 'seg_weight', 'device']):
        raise ValueError("Missing required configuration parameters")
    
    # Move to device
    device = torch.device(train_config['device'])
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=train_config
    )
    
    # Create checkpoint directory
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    
    # Start training
    trainer.train(experiment_name=args.experiment_name)

if __name__ == '__main__':
    main()