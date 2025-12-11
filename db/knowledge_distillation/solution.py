"""
Knowledge Distillation - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """Combined distillation loss."""
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft label loss (KL divergence)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    
    # Combine (multiply soft loss by T² to maintain gradient scale)
    loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss
    
    return loss


def distillation_loss_mse(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """Distillation using MSE on logits (alternative)."""
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.mse_loss(student_logits, teacher_logits)
    return alpha * hard_loss + (1 - alpha) * soft_loss


class StudentModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        return self.classifier(self.features(x))


class TeacherModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        return self.classifier(self.features(x))


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    epochs: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.5,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """Train student with distillation."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)
    
    x = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    
    loss = distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5)
    
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.1f}x")
    print(f"Distillation loss: {loss.item():.4f}")
