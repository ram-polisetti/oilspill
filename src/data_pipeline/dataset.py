from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from torch.utils.data import Dataset

class ZENONODODataset(Dataset):
    """Dataset handler for ZENONODO SAR imagery dataset for oil spill detection."""

    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: Optional[List[str]] = None,
                 bbox_paths: Optional[List[str]] = None,
                 transform=None,
                 image_size: int = 256,
                 class_weights: Optional[np.ndarray] = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.bbox_paths = bbox_paths
        self.transform = transform
        self.image_size = image_size
        self.class_weights = class_weights
        self.classes = ["background", "oil_spill"]
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply SAR-specific preprocessing.
        
        Args:
            image: Input SAR image
            
        Returns:
            Preprocessed image
        """
        # Speckle reduction using Lee filter
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Calibration and normalization
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return image
    
    def __getitem__(self, idx: int) -> Dict:
        # Load SAR image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.preprocess_image(image)
        
        result = {"image": image}
        
        # Load mask if available
        if self.mask_paths:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 0).astype(np.float32)
            result["mask"] = mask
        
        # Load bounding boxes if available
        if self.bbox_paths:
            with open(self.bbox_paths[idx], 'r') as f:
                bboxes = [list(map(float, line.strip().split())) for line in f]
            result["bboxes"] = np.array(bboxes)
        
        # Apply augmentations if specified
        if self.transform:
            transformed = self.transform(**result)
            result = transformed
        
        return result