from typing import Dict, Tuple
import torch
import torch.nn as nn
from ultralytics import YOLO

class FeatureFusion(nn.Module):
    """Feature fusion module to combine YOLO and U-Net++ features."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNetPlusPlusDecoder(nn.Module):
    """U-Net++ decoder for semantic segmentation."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return torch.sigmoid(self.conv(x))

class HybridModel(nn.Module):
    """Hybrid architecture combining YOLOv8 and U-Net++ for oil spill detection using transfer learning."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Initialize YOLO backbone with transfer learning
        self.detector = YOLO('yolov8n.pt')
        if not pretrained:
            self.detector = self.detector.model
        
        # Freeze YOLO backbone layers
        for param in self.detector.model.parameters():
            param.requires_grad = False
        
        # Only fine-tune the detection head
        for param in self.detector.model.head.parameters():
            param.requires_grad = True
            
        # Initialize U-Net++ decoder for segmentation
        self.decoder = UNetPlusPlusDecoder(512)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get YOLO features and predictions
        det_output = self.detector(x)
        det_features = self.detector.model.backbone(x)[-1]  # Extract backbone features
        
        # Segmentation using backbone features
        mask_pred = self.decoder(det_features)
        
        return det_output, mask_pred

class HybridLoss(nn.Module):
    """Custom loss combining detection and segmentation losses."""
    
    def __init__(self, detector: YOLO, det_weight: float = 0.4, seg_weight: float = 0.6):
        super().__init__()
        self.detector = detector
        self.det_weight = det_weight
        self.seg_weight = seg_weight
        self.seg_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                det_pred: torch.Tensor,
                mask_pred: torch.Tensor,
                det_target: torch.Tensor,
                mask_target: torch.Tensor) -> torch.Tensor:
        det_loss = self.detector.loss(det_pred, det_target)
        seg_loss = self.seg_criterion(mask_pred, mask_target)
        
        return self.det_weight * det_loss + self.seg_weight * seg_loss