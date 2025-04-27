# Oil Spill Detection System

A deep learning-based system for detecting and segmenting oil spills in SAR (Synthetic Aperture Radar) imagery using a hybrid architecture combining YOLOv8 and U-Net++.

## Features

- Hybrid architecture combining YOLOv8 for detection and U-Net++ for segmentation
- SAR-specific data preprocessing with speckle reduction and calibration
- Feature fusion mechanism for improved performance
- Comprehensive training pipeline with Metal Performance Shaders (MPS) support
- Modular and extensible codebase

## Project Structure

```
oilspill/
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── dataset.py        # ZENONODO dataset handler
│   │   ├── augmentation.py   # Data augmentation pipeline
│   │   └── zenodo_loader.py  # Dataset loading utilities
│   ├── models/
│   │   └── hybrid_model.py   # YOLOv8 + U-Net++ architecture
│   ├── training/
│   │   └── trainer.py        # Training loop and validation
│   ├── configs/
│   │   └── config.py         # Configuration parameters
│   └── train.py              # Main training script
```

## Model Architecture

### Hybrid Model

The system uses a hybrid architecture that combines:
- **YOLOv8**: For object detection and localization of oil spills
- **U-Net++**: For precise segmentation of oil spill regions
- **Feature Fusion**: Custom module to combine features from both networks

```python
class HybridModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Initialize YOLO backbone
        self.detector = YOLO('yolov8n.pt')
        if not pretrained:
            self.detector = self.detector.model
            
        # Initialize U-Net++ components
        self.fusion = FeatureFusion(1024)  # Adjust channels based on YOLO feature maps
        self.decoder = UNetPlusPlusDecoder(512)
```

### Key Components

1. **Feature Fusion Module**
   - Combines feature maps from YOLO and U-Net++
   - Uses convolutional layers with batch normalization
   - Reduces channel dimensions for efficient processing

```python
class FeatureFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1)
        )
```

2. **U-Net++ Decoder**
   - Bilinear upsampling for feature maps
   - Sequential convolution blocks for refinement
   - Final sigmoid activation for binary segmentation

3. **Hybrid Loss Function**
   - Weighted combination of detection and segmentation losses
   - Configurable weights (default: 0.4 for detection, 0.6 for segmentation)

## Data Pipeline

### Dataset Handler

The `ZENONODODataset` class provides:
- SAR-specific preprocessing including speckle reduction
- Support for both image masks and bounding boxes
- Dynamic data augmentation pipeline
- Efficient batch processing

```python
class ZENONODODataset(Dataset):
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: Optional[List[str]] = None,
                 bbox_paths: Optional[List[str]] = None,
                 transform=None,
                 image_size: int = 256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.bbox_paths = bbox_paths
        self.transform = transform
        self.image_size = image_size
```

### Preprocessing Steps

1. Speckle reduction using Gaussian filtering
2. Image calibration and normalization
3. Resizing to configured dimensions (256x256)
4. Optional data augmentation using albumentations

```python
def get_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.Rotate(limit=10, p=p),
        A.RandomBrightnessContrast(p=p),
        A.GaussNoise(var_limit=(10.0, 50.0), p=p),
        A.GridDistortion(p=p/2)
    ])
```

## Training Pipeline

### Configuration

Key training parameters (configurable in `config.py`):
- Epochs: 100
- Batch size: 16 (auto-scaled for distributed training)
- Initial learning rate: 1e-4
- Validation interval: 5 epochs
- Image size: 256x256
- Loss weights: 0.4 (detection), 0.6 (segmentation)

### Model Architecture & Training

1. **Transfer Learning Implementation**
   ```python
   class Trainer(L.LightningModule):
       def __init__(self, model, train_loader, val_loader, criterion, config):
           self.model = model  # Pre-trained backbone with custom heads
           self.criterion = criterion
           self.automatic_optimization = True
           
           # Metrics initialization
           self.precision = Precision(task='binary', sync_dist=True)
           self.recall = Recall(task='binary', sync_dist=True)
           self.f1_score = F1Score(task='binary', sync_dist=True)
   ```

2. **Training Features**
   - Automatic mixed precision (AMP) for faster training
   - Vectorized IoU calculation for efficient batch processing
   - Advanced metrics tracking (Precision, Recall, F1-Score)
   - Cosine annealing learning rate scheduling

3. **Performance Optimization**
   - Distributed training with DDP strategy
   - Gradient scaling for numerical stability
   - Early stopping with configurable patience
   - Automatic device detection and GPU optimization

4. **Monitoring & Checkpointing**
   - Top-k model checkpointing
   - WandB integration for experiment tracking
   - Real-time metrics visualization
   - Comprehensive validation metrics

### Training Configuration & Deployment

1. **Hardware Setup**
   ```python
   # Configure hardware in trainer initialization
   trainer = Trainer(
       model=model,
       accelerator="gpu",  # or "cpu", "tpu", "auto"
       devices="auto",    # or specific number like 2
       strategy="ddp",    # or "deepspeed", "fsdp"
       config={
           'initial_lr': 1e-4,
           'epochs': 100,
           'patience': 10
       }
   )
   ```

2. **Training Launch**
   ```bash
   # Single GPU training
   python train.py --accelerator gpu --devices 1
   
   # Multi-GPU training
   python train.py --accelerator gpu --devices 4 --strategy ddp
   
   # CPU training
   python train.py --accelerator cpu
   ```

3. **Performance Tips**
   - Use `strategy="ddp"` for distributed data parallel training
   - Enable `automatic_optimization=True` for Lightning-managed training
   - Adjust batch size based on available GPU memory
   - Monitor GPU utilization and adjust learning rate accordingly

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Ultralytics (YOLOv8)
- OpenCV
- Albumentations

Refer to `requirements.txt` for complete dependencies.