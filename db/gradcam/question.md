# GradCAM for Model Explainability

## Problem Statement

Implement **Gradient-weighted Class Activation Mapping (GradCAM)** to visualize which regions of an image a CNN focuses on for classification.

Your task is to:

1. Hook into a target convolutional layer
2. Capture activations during forward pass
3. Capture gradients during backward pass
4. Compute weighted combination for heatmap

## Requirements

- Use PyTorch hooks to capture activations and gradients
- Support any target layer in the network
- Generate heatmaps for any class
- Overlay heatmap on original image

## Function Signature

```python
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        pass
    
    def __call__(self, input_image: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """Generate GradCAM heatmap."""
        pass
```

## GradCAM Algorithm

```
1. Forward pass: save feature maps A from target layer
2. Backward pass: save gradients ∂y/∂A
3. Compute importance weights: α = global_avg_pool(gradients)
4. Weighted combination: L = ReLU(Σ α * A)
5. Upsample to input size
```

## Example

```python
model = torchvision.models.resnet50(pretrained=True)
gradcam = GradCAM(model, target_layer=model.layer4[-1])

image = load_image("cat.jpg")
heatmap = gradcam(image, target_class=281)  # tabby cat

# Overlay on image
visualization = overlay_heatmap(image, heatmap)
```

## Hints

- Use `register_forward_hook` to capture activations
- Use `register_backward_hook` or `register_full_backward_hook` for gradients
- Apply ReLU to filter out negative influences
- Normalize heatmap to [0, 1] for visualization
