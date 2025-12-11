"""
Knowledge Distillation - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    KL divergence between softened student and teacher distributions.
    """
    # TODO: Compute soft targets from teacher
    # TODO: Compute soft predictions from student
    # TODO: Return KL divergence
    pass


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Combined distillation loss.
    
    L = α * CE(student, labels) + (1-α) * T² * KL(soft_student || soft_teacher)
    
    Args:
        student_logits: Student predictions (batch, num_classes)
        teacher_logits: Teacher predictions (batch, num_classes)
        labels: Ground truth labels (batch,)
        temperature: Softening temperature
        alpha: Weight for hard label loss
    """
    # TODO: Hard label loss (cross entropy)
    
    # TODO: Soft label loss (KL divergence with temperature)
    
    # TODO: Combine losses
    
    pass


class StudentModel(nn.Module):
    """Small student model."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO: Small CNN or MLP
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TeacherModel(nn.Module):
    """Larger teacher model."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO: Larger model
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    epochs: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.5,
    lr: float = 0.001
):
    """
    Train student model with knowledge distillation.
    """
    # TODO: Freeze teacher
    
    # TODO: Create optimizer for student
    
    # TODO: Training loop
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create models
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)
    
    # Dummy forward
    x = torch.randn(32, 3, 32, 32)
    
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    labels = torch.randint(0, 10, (32,))
    
    loss = distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5)
    
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Distillation loss: {loss.item():.4f}")
