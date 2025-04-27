from typing import Dict, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import lightning as L
from lightning.fabric.strategies import DDPStrategy
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Precision, Recall, F1Score
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

class Trainer(L.LightningModule):
    """Trainer class for the hybrid oil spill detection model with Lightning integration."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: torch.nn.Module,
                 config: Dict,
                 accelerator: str = "auto",
                 strategy: str = "ddp",
                 devices: int = -1):
        super().__init__()
        
        # Lightning specific configurations
        self.automatic_optimization = True
        self.save_hyperparameters(config)
        
        # Training strategy setup
        self.accelerator = accelerator
        self.strategy = DDPStrategy() if strategy == "ddp" else strategy
        self.devices = devices
        
        # Initialize metrics
        self.precision = Precision(task='binary', sync_dist=True)
        self.recall = Recall(task='binary', sync_dist=True)
        self.f1_score = F1Score(task='binary', sync_dist=True)
        self.scaler = GradScaler()
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        
        # Initialize model and components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        
        # Setup callbacks for cloud training
        self.checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='oilspill-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
        
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min'
        )
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['initial_lr'],
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
    def training_step(self, batch, batch_idx) -> Dict:
        """Lightning training step with mixed precision support."""
        # Move batch to device and optimize memory
        images = batch['image']
        bboxes = batch['bboxes']
        masks = batch['mask']
        
        # Forward pass with mixed precision
        with autocast():
            det_output, mask_pred = self.model(images)
            loss = self.criterion(
                det_output,
                mask_pred,
                bboxes,
                masks
            )
        
        # Calculate metrics
        det_accuracy = self._calculate_detection_accuracy(det_output, bboxes)
        seg_iou = self._calculate_segmentation_iou(mask_pred, masks)
        
        # Calculate additional metrics
        precision = self.precision(mask_pred, masks)
        recall = self.recall(mask_pred, masks)
        f1 = self.f1_score(mask_pred, masks)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_det_accuracy', det_accuracy, prog_bar=True)
        self.log('train_seg_iou', seg_iou, prog_bar=True)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_f1', f1)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx) -> Dict:
        """Lightning validation step."""
        images = batch['image']
        bboxes = batch['bboxes']
        masks = batch['mask']
        
        det_output, mask_pred = self.model(images)
        loss = self.criterion(
            det_output,
            mask_pred,
            bboxes,
            masks
        )
        
        det_accuracy = self._calculate_detection_accuracy(det_output, bboxes)
        seg_iou = self._calculate_segmentation_iou(mask_pred, masks)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_det_accuracy', det_accuracy, prog_bar=True)
        self.log('val_seg_iou', seg_iou, prog_bar=True)
        
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers with Lightning-specific settings."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config['initial_lr'],
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs']
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch"
            }
        }
        
    def train_dataloader(self):
        """Return the training dataloader."""
        return self.train_loader
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.val_loader
    
    def _calculate_detection_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate detection accuracy using IoU threshold with vectorized operations."""
        iou_threshold = 0.5
        pred_boxes = pred.detach()
        target_boxes = target.detach()
        
        # Vectorized IoU calculation
        x1 = torch.max(pred_boxes[:, None, 0], target_boxes[None, :, 0])  # [N,M]
        y1 = torch.max(pred_boxes[:, None, 1], target_boxes[None, :, 1])  # [N,M]
        x2 = torch.min(pred_boxes[:, None, 2], target_boxes[None, :, 2])  # [N,M]
        y2 = torch.min(pred_boxes[:, None, 3], target_boxes[None, :, 3])  # [N,M]
        
        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)  # [N,M]
        
        # Calculate areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])  # [N]
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])  # [M]
        
        # Calculate union area
        union = pred_area[:, None] + target_area[None, :] - intersection  # [N,M]
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)  # [N,M]
        
        # Count correct detections
        correct_detections = torch.sum(torch.max(iou, dim=1)[0] >= iou_threshold).item()
        total_detections = len(pred_boxes)
                max_iou = max(max_iou, iou)
            if max_iou >= iou_threshold:
                correct_detections += 1
        
        return correct_detections / (total_detections + 1e-6)
    
    def _calculate_segmentation_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate segmentation IoU score with morphological post-processing."""
        from torch.nn.functional import max_pool2d, avg_pool2d
        
        # Apply morphological operations
        # 1. Remove small noise using average pooling followed by max pooling
        kernel_size = 3
        pred = (pred > 0.5).float()
        pred = max_pool2d(avg_pool2d(pred, kernel_size, stride=1, padding=1), 
                         kernel_size, stride=1, padding=1)
        
        # 2. Calculate IoU
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + 1e-6) / (union + 1e-6)