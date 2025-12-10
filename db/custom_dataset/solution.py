"""
Custom Dataset and DataLoader - Solution

Complete implementation of custom Dataset classes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Callable, Optional
from pathlib import Path


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
        # Load CSV using pandas
        self.df = pd.read_csv(csv_path)
        
        # Extract features and labels
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[label_col].values
        
        # Handle label types
        if self.labels.dtype == np.float64:
            self.labels = self.labels.astype(np.float32)
        elif self.labels.dtype == np.int64:
            self.labels = self.labels.astype(np.int64)
        
        # Store transform
        self.transform = transform
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        features = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        
        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)
        
        return features, label


class ImageFolderDataset(Dataset):
    """Custom Dataset for loading images from folders."""
    
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset from folder structure.
        
        Expects:
        root_dir/class1/img1.jpg, root_dir/class2/img2.jpg, etc.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        
        # Load image (using PIL if available)
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except ImportError:
            # Fallback: return path if PIL not available
            image = img_path
        
        return image, label


class SequenceDataset(Dataset):
    """Dataset for variable-length sequences."""
    
    def __init__(self, sequences: list[list], labels: list):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])


def collate_sequences(batch):
    """Custom collate function for variable-length sequences."""
    sequences, labels = zip(*batch)
    
    # Pad sequences to max length in batch
    max_len = max(len(seq) for seq in sequences)
    
    padded = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    labels = torch.stack(labels)
    
    return padded, labels, lengths


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
    
    # Test sequence dataset with custom collate
    print("\nTesting variable-length sequences:")
    seq_dataset = SequenceDataset(
        sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        labels=[0, 1, 0]
    )
    seq_loader = DataLoader(seq_dataset, batch_size=2, collate_fn=collate_sequences)
    
    for padded, labels, lengths in seq_loader:
        print(f"Padded: {padded.shape}, Lengths: {lengths}")
        break
