"""
GradCAM - Starting Point

Implement GradCAM for CNN interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: CNN model
            target_layer: Layer to compute GradCAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # TODO: Register hooks
        # Forward hook to capture activations
        # Backward hook to capture gradients
        pass
    
    def _save_activation(self, module, input, output):
        """Hook to save activations during forward pass."""
        # TODO: Save output (activations)
        pass
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        # TODO: Save grad_output (gradients)
        pass
    
    def __call__(self, input_image: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """
        Generate GradCAM heatmap.
        
        Args:
            input_image: (1, C, H, W) normalized image
            target_class: Class to generate heatmap for (None = predicted class)
        
        Returns:
            Heatmap of shape (H, W)
        """
        self.model.eval()
        
        # TODO: Forward pass
        
        # TODO: If no target class, use predicted class
        
        # TODO: Backward pass on target class score
        
        # TODO: Get activations and gradients
        
        # TODO: Compute weights (global average pooling of gradients)
        # weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # TODO: Weighted combination
        # cam = (weights * activations).sum(dim=1)
        
        # TODO: Apply ReLU
        
        # TODO: Normalize to [0, 1]
        
        pass
    
    def remove_hooks(self):
        """Remove registered hooks."""
        pass


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4):
    """
    Overlay heatmap on image.
    
    Args:
        image: (C, H, W) image
        heatmap: (H, W) heatmap
        alpha: Blend factor
    """
    # TODO: Resize heatmap to image size
    # TODO: Apply colormap to heatmap
    # TODO: Blend with original image
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create simple CNN
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    # Get target layer (last conv)
    target_layer = model[2]
    
    # Create GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Test image
    image = torch.randn(1, 3, 32, 32)
    
    # Generate heatmap
    heatmap = gradcam(image, target_class=5)
    
    print(f"Image shape: {image.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
