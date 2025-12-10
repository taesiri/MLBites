"""
Custom Dataset and DataLoader - Starting Point

Create a custom Dataset class for loading data from CSV files.
Fill in the TODO sections to complete the implementation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Callable, Optional


class CSVDataset(Dataset):
    """Custom Dataset for loading data from CSV files."""
    
    def __init__(
        self, 
        csv_path: str, 
        feature_cols: list[str],
        label_col: str,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            feature_cols: List of column names to use as features
            label_col: Column name for labels
            transform: Optional transform to apply to features
        """
        # TODO: Load CSV using pandas
        
        # TODO: Extract features and labels
        
        # TODO: Store transform
        
        pass
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        # TODO: Return the number of samples
        pass
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label) as tensors
        """
        # TODO: Get features and label at index
        
        # TODO: Convert to tensors
        
        # TODO: Apply transform if provided
        
        pass


class ImageFolderDataset(Dataset):
    """Custom Dataset for loading images from folders."""
    
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset from folder structure.
        
        Expects folder structure:
        root_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
        """
        # TODO: Walk through directory and collect image paths
        
        # TODO: Create class-to-index mapping
        
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, idx: int):
        # TODO: Load image and return with label
        pass


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Use this when samples have different sizes and need special handling.
    """
    # TODO: Implement custom batching logic
    pass


if __name__ == "__main__":
    # Create sample CSV data for testing
    sample_data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('/tmp/sample_data.csv', index=False)
    
    # Test custom dataset
    dataset = CSVDataset(
        csv_path='/tmp/sample_data.csv',
        feature_cols=['feature1', 'feature2', 'feature3'],
        label_col='target'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    features, label = dataset[0]
    print(f"Single sample - Features: {features.shape}, Label: {label}")
    
    # Test with DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_features, batch_labels in dataloader:
        print(f"Batch - Features: {batch_features.shape}, Labels: {batch_labels.shape}")
        break
