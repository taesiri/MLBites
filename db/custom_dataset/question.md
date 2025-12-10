# Custom Dataset and DataLoader

## Problem Statement

Implement a **custom Dataset** class that loads data from a CSV file and use it with PyTorch's DataLoader. Understanding how to create custom datasets is essential for working with real-world data.

Your task is to:

1. Create a custom Dataset class that inherits from `torch.utils.data.Dataset`
2. Implement the required methods: `__init__`, `__len__`, `__getitem__`
3. Handle data transformations and preprocessing
4. Use the Dataset with DataLoader for batching and shuffling

## Requirements

- Inherit from `torch.utils.data.Dataset`
- Implement all required methods
- Support optional transforms
- Handle both features and labels
- Work with DataLoader for batching

## Function Signature

```python
class CSVDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        feature_cols: list[str],
        label_col: str,
        transform: callable = None
    ):
        """Initialize dataset from CSV file."""
        pass
    
    def __len__(self) -> int:
        """Return number of samples."""
        pass
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample (features, label)."""
        pass
```

## Example

```python
import torch
from torch.utils.data import DataLoader

# Create dataset
dataset = CSVDataset(
    csv_path="data.csv",
    feature_cols=["feature1", "feature2", "feature3"],
    label_col="target"
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for features, labels in dataloader:
    print(f"Batch features shape: {features.shape}")
    print(f"Batch labels shape: {labels.shape}")
```

## Hints

- Use pandas to read CSV: `pd.read_csv(csv_path)`
- Convert to tensors in `__getitem__`
- Apply transforms if provided
- DataLoader handles batching automatically
- Consider using `num_workers` for parallel data loading
