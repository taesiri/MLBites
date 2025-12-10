"""
Distributed Training (DDP) - Starting Point

Implement distributed training with PyTorch DDP.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed process group.
    
    Args:
        rank: Unique process ID
        world_size: Total number of processes
        backend: 'nccl' for GPU, 'gloo' for CPU
    """
    # TODO: Set environment variables if not set
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    
    # TODO: Initialize process group
    # dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # TODO: Set CUDA device
    # torch.cuda.set_device(rank)
    
    pass


def cleanup_distributed():
    """Clean up distributed process group."""
    # TODO: Destroy process group
    pass


def create_model_ddp(model: nn.Module, rank: int) -> DDP:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Neural network model
        rank: Process rank (GPU to use)
    
    Returns:
        DDP-wrapped model
    """
    # TODO: Move model to device
    # TODO: Wrap with DDP
    pass


def create_distributed_dataloader(dataset: Dataset, batch_size: int, rank: int, world_size: int):
    """
    Create DataLoader with DistributedSampler.
    """
    # TODO: Create DistributedSampler
    
    # TODO: Create DataLoader with sampler
    pass


def train_one_epoch(model, dataloader, optimizer, criterion, rank, epoch, sampler):
    """Train for one epoch."""
    # TODO: Set epoch on sampler for proper shuffling
    # sampler.set_epoch(epoch)
    
    # TODO: Training loop
    pass


def train_distributed(rank: int, world_size: int, epochs: int = 10):
    """
    Main training function for each process.
    
    Args:
        rank: Process rank
        world_size: Total processes
        epochs: Number of epochs
    """
    # TODO: Setup distributed
    
    # TODO: Create model and wrap with DDP
    
    # TODO: Create distributed dataloader
    
    # TODO: Training loop
    
    # TODO: Cleanup
    pass


if __name__ == "__main__":
    # Check for GPUs
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size < 2:
        print("Need at least 2 GPUs for distributed training demo")
        print("Showing single-GPU training instead...")
        
        # Single GPU demo
        model = nn.Linear(10, 5)
        x = torch.randn(32, 10)
        y = model(x)
        print(f"Output shape: {y.shape}")
    else:
        import torch.multiprocessing as mp
        
        mp.spawn(
            train_distributed,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
