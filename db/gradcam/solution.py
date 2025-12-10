"""
GradCAM - Solution

Complete GradCAM implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_image: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """Generate GradCAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # If no target class, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass on target class score
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # (B, C, H, W)
        gradients = self.gradients      # (B, C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Squeeze batch and channel
        cam = cam.squeeze()
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """GradCAM++ - improved version with better localization."""
    
    def __call__(self, input_image: torch.Tensor, target_class: int = None) -> torch.Tensor:
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        activations = self.activations
        gradients = self.gradients
        
        # GradCAM++ weights
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)
        alpha = alpha * F.relu(gradients)
        
        weights = alpha.sum(dim=(2, 3), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4):
    """Overlay heatmap on image."""
    # Resize heatmap to image size
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=image.shape[-2:],
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Simple RGB colormap (blue -> red)
    heatmap_rgb = torch.zeros(3, *heatmap.shape)
    heatmap_rgb[0] = heatmap  # Red channel
    heatmap_rgb[2] = 1 - heatmap  # Blue channel
    
    # Blend
    result = (1 - alpha) * image + alpha * heatmap_rgb
    return result.clamp(0, 1)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    target_layer = model[2]
    gradcam = GradCAM(model, target_layer)
    
    image = torch.randn(1, 3, 32, 32)
    heatmap = gradcam(image, target_class=5)
    
    print(f"Image shape: {image.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Test overlay
    vis = overlay_heatmap(image[0], heatmap)
    print(f"Visualization shape: {vis.shape}")
    
    gradcam.remove_hooks()
