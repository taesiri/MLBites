# Knowledge Distillation

## Problem Statement

Implement **Knowledge Distillation** to train a smaller student model to mimic a larger teacher model. The student learns from both hard labels and soft targets (teacher's softened predictions).

Your task is to:

1. Compute soft targets from teacher with temperature
2. Implement distillation loss (KL divergence on soft targets)
3. Combine hard and soft losses
4. Train student to match teacher

## Distillation Loss

```
L = α * L_hard(student, labels) + (1 - α) * T² * L_soft(student, teacher)

L_soft = KL(softmax(student_logits/T) || softmax(teacher_logits/T))
```

Temperature T softens the probability distribution, revealing more information about class similarities.

## Function Signature

```python
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """Combined distillation and hard label loss."""
    pass

def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    epochs: int,
    temperature: float = 4.0,
    alpha: float = 0.5
):
    pass
```

## Example

```python
teacher = load_pretrained_resnet50()
student = SmallCNN()

# Train student with distillation
for images, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(images)
    
    student_logits = student(images)
    loss = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5)
    loss.backward()
```

## Hints

- Temperature T > 1 makes probabilities softer
- Higher α means more weight on hard labels
- Multiply soft loss by T² to maintain gradient magnitudes
- Teacher should be frozen (no gradients)
