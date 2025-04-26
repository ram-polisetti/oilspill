# Oil Spill Detection System

A deep learning-based system for detecting and segmenting oil spills in SAR (Synthetic Aperture Radar) imagery using a hybrid architecture combining YOLOv8 and U-Net++.

## Features

- Hybrid architecture combining YOLOv8 for detection and U-Net++ for segmentation
- SAR-specific data preprocessing and augmentation pipeline
- Feature fusion mechanism for improved performance
- Comprehensive training pipeline with Weights & Biases integration
- Modular and extensible codebase

## Project Structure

```
oilspill/
├── src/
│   ├── data_pipeline/
│   │   ├── dataset.py        # ZENODO dataset handler
│   │   └── augmentation.py   # Data augmentation pipeline
│   ├── models/
│   │   └── hybrid_model.py   # YOLOv8 + U-Net++ architecture
│   ├── training/
│   │   └── trainer.py        # Training loop and validation
│   ├── configs/
│   │   └── config.py         # Configuration parameters
│   └── train.py              # Main training script
└── requirements.txt          # Project dependencies
```

## Model Architecture

### Hybrid Model

The system uses a hybrid architecture that combines:
- **YOLOv8**: For object detection and localization of oil spills
- **U-Net++**: For precise segmentation of oil spill regions
- **Feature Fusion**: Custom module to combine features from both networks

```python
# models/hybrid_model.py
class OilSpillHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.yolo = YOLOv8('yolov8n.pt')
        self.unetpp = UNetPlusPlus(in_channels=3, out_channels=1)
        self.fusion = FeatureFusionModule()
        
    def forward(self, x):
        # YOLO detection branch
        yolo_features = self.yolo.backbone(x)
        detection_output = self.yolo.head(yolo_features)
        
        # U-Net++ segmentation branch
        seg_features = self.unetpp.encoder(x)
        
        # Feature fusion
        fused_features = self.fusion(yolo_features, seg_features)
        segmentation_output = self.unetpp.decoder(fused_features)
        
        return detection_output, segmentation_output
```

### Key Components

1. **Feature Fusion Module**
   - Combines feature maps from YOLO and U-Net++
   - Uses convolutional layers with batch normalization
   - Reduces channel dimensions for efficient processing

```python
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        
    def forward(self, yolo_feat, unet_feat):
        # Concatenate features along channel dimension
        x = torch.cat([yolo_feat, unet_feat], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
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

The `ZenodoDataset` class provides:
- SAR-specific preprocessing including speckle reduction
- Support for both image masks and bounding boxes
- Dynamic data augmentation pipeline
- Efficient batch processing

```python
# data_pipeline/dataset.py
class ZenodoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_dataset()
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]['image']
        mask_path = self.samples[idx]['mask']
        bbox = self.samples[idx]['bbox']
        
        # Load and preprocess SAR image
        image = self._load_sar_image(img_path)
        mask = self._load_mask(mask_path)
        
        if self.transform:
            image, mask, bbox = self.transform(image, mask, bbox)
            
        return {
            'image': image,
            'mask': mask,
            'bbox': bbox
        }
```

### Preprocessing Steps

1. Speckle reduction using Gaussian filtering
2. Image calibration and normalization
3. Resizing to configured dimensions
4. Optional data augmentation

```python
# data_pipeline/augmentation.py
class SARPreprocessing:
    def __init__(self, img_size=256):
        self.img_size = img_size
        
    def __call__(self, image):
        # Speckle reduction
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Calibration and normalization
        image = self._calibrate_sar(image)
        image = (image - image.mean()) / image.std()
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image
        
    def _calibrate_sar(self, image):
        # SAR-specific calibration
        return 10 * np.log10(image + 1e-10)
```

## Training Pipeline

### Configuration

Key training parameters (configurable in `config.py`):
- Epochs: 100
- Batch size: 16
- Initial learning rate: 1e-4
- Validation interval: 5 epochs
- Image size: 256x256
- Loss weights: 0.4 (detection), 0.6 (segmentation)

```python
# configs/config.py
class Config:
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    VAL_INTERVAL = 5
    
    # Model parameters
    IMG_SIZE = 256
    DETECTION_WEIGHT = 0.4
    SEGMENTATION_WEIGHT = 0.6
    
    # Dataset parameters
    DATA_ROOT = 'path/to/zenodo/dataset'
    NUM_WORKERS = 4
    
    # Optimizer parameters
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
```

### Training Features

1. **Optimizer and Scheduler**
   - AdamW optimizer with weight decay
   - Cosine annealing learning rate scheduler

2. **Metrics Tracking**
   - Training and validation loss
   - Detection accuracy
   - Segmentation IoU score

3. **Checkpointing**
   - Saves best model based on validation loss
   - Stores optimizer state for training resumption

4. **Weights & Biases Integration**
   - Real-time metric logging
   - Experiment tracking and comparison

## Usage

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Download the Zenodo SAR dataset
   - Update dataset path in config.py

3. **Training**
   ```bash
   python src/train.py --config src/configs/config.py
   ```

4. **Monitoring**
   - Access training metrics through Weights & Biases dashboard
   - Monitor validation metrics every 5 epochs

## Model Performance

- Detection Accuracy: [To be updated]
- Segmentation IoU: [To be updated]
- Inference Speed: [To be updated]

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Ultralytics (YOLOv8)
- OpenCV
- Weights & Biases

Refer to `requirements.txt` for complete dependencies.