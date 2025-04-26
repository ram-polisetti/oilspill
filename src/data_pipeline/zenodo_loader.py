from pathlib import Path
from typing import List, Tuple, Dict
import json
import numpy as np

class ZenodoDataLoader:
    """Handles loading and preprocessing of Zenodo SAR oil spill dataset."""
    
    def __init__(self, dataset_path: str):
        """Initialize the data loader.
        
        Args:
            dataset_path: Path to the Zenodo dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_file = self.dataset_path / 'metadata.json'
        
    def load_dataset_splits(self) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
        """Load train/val/test splits from Zenodo dataset.
        
        Returns:
            Dictionary containing image_paths, mask_paths, and bbox_paths for each split
        """
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
            
        splits = {}
        for split in ['train', 'val', 'test']:
            split_data = metadata[split]
            
            image_paths = [str(self.dataset_path / 'images' / img_name) 
                          for img_name in split_data['images']]
            mask_paths = [str(self.dataset_path / 'masks' / mask_name)
                         for mask_name in split_data['masks']]
            bbox_paths = [str(self.dataset_path / 'annotations' / bbox_name)
                         for bbox_name in split_data['bboxes']]
                         
            splits[split] = (image_paths, mask_paths, bbox_paths)
            
        return splits
    
    def get_class_weights(self) -> np.ndarray:
        """Calculate class weights based on pixel distribution in masks.
        
        Returns:
            Array of class weights for background and oil spill
        """
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        bg_pixels = metadata['statistics']['background_pixels']
        oil_pixels = metadata['statistics']['oil_spill_pixels']
        total_pixels = bg_pixels + oil_pixels
        
        weights = np.array([1.0 / (bg_pixels / total_pixels),
                           1.0 / (oil_pixels / total_pixels)])
        weights = weights / np.sum(weights)  # normalize
        
        return weights