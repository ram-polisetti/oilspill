from typing import Dict

def get_training_config() -> Dict:
    """Get training configuration parameters.
    
    Returns:
        Dictionary containing training configuration
    """
    return {
        'epochs': 100,
        'batch_size': 16,
        'initial_lr': 1e-4,
        'validation_interval': 5,
        'image_size': 256,
        'num_classes': 2,  # background and oil_spill
        'det_weight': 0.4,
        'seg_weight': 0.6,
        'augmentation_prob': 0.5,
        'pretrained': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'pin_memory': True,
        'checkpoint_dir': 'checkpoints',
        'log_interval': 100,
        # Zenodo dataset configuration
        'dataset_path': '/path/to/zenodo/dataset',  # Update with actual path
        'use_class_weights': True,
        'cache_data': True  # Enable data caching for faster training
    }

def get_model_config() -> Dict:
    """Get model architecture configuration.
    
    Returns:
        Dictionary containing model configuration
    """
    return {
        'backbone': 'yolov8n',
        'feature_fusion_channels': 1024,
        'decoder_channels': 512,
        'use_pretrained': True
    }