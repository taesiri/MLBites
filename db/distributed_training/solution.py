"""
Distributed Training (DDP) - Solution

Complete DDP implementation.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed process group."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def create_model_ddp(model: nn.Module, rank: int) -> DDP:
    """Wrap model with DDP."""
    model = model.to(rank)
    return DDP(model, device_ids=[rank])


def create_distributed_dataloader(dataset: Dataset, batch_size: int, rank: int, world_size: int):
    """Create DataLoader with DistributedSampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader, sampler


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


def train_distributed(rank: int, world_size: int, epochs: int = 5):
    """Main training function for each process."""
    # Setup
    setup_distributed(rank, world_size)
    
    # Create model and wrap with DDP
    model = SimpleModel()
    model = create_model_ddp(model, rank)
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 5, (1000,))
    dataset = TensorDataset(X, y)
    
    # Create distributed dataloader
    dataloader, sampler = create_distributed_dataloader(dataset, batch_size=32, rank=rank, world_size=world_size)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Only print from rank 0
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save checkpoint (only from rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), '/tmp/ddp_model.pt')
    
    cleanup_distributed()


# Alternative: Using torchrun launcher
def main():
    """Entry point when using torchrun launcher."""
    # torchrun sets these environment variables
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # ... training code ...
    
    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size < 2:
        print("Running single GPU demo...")
        model = SimpleModel()
        x = torch.randn(32, 10)
        y = model(x)
        print(f"Output: {y.shape}")
    else:
        import torch.multiprocessing as mp
        mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
