"""
AlexNet from Scratch - Solution

Complete AlexNet implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResponseNorm(nn.Module):
    """Local Response Normalization."""
    
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 2.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Normalize across channels
        square = x.pow(2)
        # Use avg_pool3d to sum over neighboring channels
        square = square.unsqueeze(1)  # (B, 1, C, H, W)
        padding = self.size // 2
        square = F.pad(square, (0, 0, 0, 0, padding, padding))
        square = F.avg_pool3d(square, (self.size, 1, 1), stride=1)
        square = square.squeeze(1) * self.size  # Undo averaging
        
        return x / (self.k + self.alpha * square).pow(self.beta)


class AlexNet(nn.Module):
    """AlexNet architecture."""
    
    def __init__(self, num_classes: int = 1000, use_lrn: bool = True):
        super().__init__()
        
        self.use_lrn = use_lrn
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5) if use_lrn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5) if use_lrn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling for flexible input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetBN(nn.Module):
    """Modern AlexNet with BatchNorm instead of LRN."""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = AlexNet(num_classes=1000)
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Modern version
    model_bn = AlexNetBN(num_classes=1000)
    print(f"\nAlexNet-BN parameters: {sum(p.numel() for p in model_bn.parameters()):,}")
