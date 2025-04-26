from typing import Dict, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Precision, Recall, F1Score

class Trainer:
    """Trainer class for the hybrid oil spill detection model."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: torch.nn.Module,
                 config: Dict):
        # Initialize metrics
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1_score = F1Score(task='binary')
        self.scaler = GradScaler()
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        
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
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using mixed precision."""
        self.model.train()
        total_loss = 0
        total_det_accuracy = 0
        total_seg_iou = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast():
                    det_output, mask_pred = self.model(batch['image'])
                    loss = self.criterion(
                        det_output,
                        mask_pred,
                        batch['bboxes'],
                        batch['mask']
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                total_loss += loss.item()
                # Calculate detection accuracy and segmentation IoU
                det_accuracy = self._calculate_detection_accuracy(det_output, batch['bboxes'])
                seg_iou = self._calculate_segmentation_iou(mask_pred, batch['mask'])
                
                # Calculate additional metrics
                precision = self.precision(mask_pred, batch['mask'])
                recall = self.recall(mask_pred, batch['mask'])
                f1 = self.f1_score(mask_pred, batch['mask'])
                
                total_det_accuracy += det_accuracy
                total_seg_iou += seg_iou
                
                pbar.set_postfix({'loss': loss.item()})
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_det_accuracy': total_det_accuracy / len(self.train_loader),
            'train_seg_iou': total_seg_iou / len(self.train_loader),
            'train_precision': self.precision.compute(),
            'train_recall': self.recall.compute(),
            'train_f1': self.f1_score.compute()
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_det_accuracy = 0
        total_seg_iou = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                det_output, mask_pred = self.model(batch['image'])
                loss = self.criterion(
                    det_output,
                    mask_pred,
                    batch['bboxes'],
                    batch['mask']
                )
                
                total_loss += loss.item()
                det_accuracy = self._calculate_detection_accuracy(det_output, batch['bboxes'])
                seg_iou = self._calculate_segmentation_iou(mask_pred, batch['mask'])
                
                total_det_accuracy += det_accuracy
                total_seg_iou += seg_iou
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_det_accuracy': total_det_accuracy / len(self.val_loader),
            'val_seg_iou': total_seg_iou / len(self.val_loader)
        }
        
        return metrics
    
    def train(self, experiment_name: Optional[str] = None):
        """Full training loop."""
        # Initialize wandb
        if experiment_name:
            wandb.init(project="oil_spill_detection", name=experiment_name)
            wandb.config.update(self.config)
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            if (epoch + 1) % self.config['validation_interval'] == 0:
                val_metrics = self.validate()
                
                # Early stopping check
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': self.best_val_loss,
                    }, f'checkpoints/best_model.pth')
                
                # Log metrics
                if experiment_name:
                    wandb.log({**train_metrics, **val_metrics})
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Update learning rate
            self.scheduler.step()
        
        if experiment_name:
            wandb.finish()
    
    def _calculate_detection_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate detection accuracy using IoU threshold."""
        iou_threshold = 0.5
        pred_boxes = pred.detach().cpu()
        target_boxes = target.detach().cpu()
        
        # Calculate IoU for each prediction-target pair
        def box_iou(box1, box2):
            # Calculate intersection coordinates
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Calculate intersection area
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            
            # Calculate union area
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / (union + 1e-6)
        
        correct_detections = 0
        total_detections = len(pred_boxes)
        
        for pred_box in pred_boxes:
            max_iou = 0
            for target_box in target_boxes:
                iou = box_iou(pred_box, target_box)
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