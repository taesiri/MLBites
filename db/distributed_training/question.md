# Distributed Training (DDP)

## Problem Statement

Implement **Distributed Data Parallel (DDP)** training in PyTorch to train models across multiple GPUs or machines.

Your task is to:

1. Initialize distributed process group
2. Wrap model with DistributedDataParallel
3. Use DistributedSampler for data loading
4. Handle proper synchronization

## Requirements

- Initialize process group with `init_process_group`
- Wrap model with `torch.nn.parallel.DistributedDataParallel`
- Use `DistributedSampler` for DataLoader
- Properly set device for each process

## Function Signature

```python
def setup_distributed(rank: int, world_size: int):
    """Initialize distributed process group."""
    pass

def cleanup_distributed():
    """Clean up process group."""
    pass

def train_distributed(rank: int, world_size: int, model: nn.Module, dataset):
    """Training function for each process."""
    pass
```

## DDP Workflow

```
1. Spawn N processes (one per GPU)
2. Each process:
   - Initialize process group
   - Create model and wrap with DDP
   - Create DataLoader with DistributedSampler
   - Train (gradients are synchronized automatically)
3. Cleanup
```

## Example

```python
import torch.multiprocessing as mp

world_size = torch.cuda.device_count()

mp.spawn(
    train_distributed,
    args=(world_size, model, dataset),
    nprocs=world_size,
    join=True
)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `rank` | Unique ID for each process (0 to world_size-1) |
| `world_size` | Total number of processes |
| `local_rank` | GPU index on current machine |
| `backend` | Communication backend (nccl for GPU) |

## Hints

- Use `nccl` backend for GPU training
- Set `MASTER_ADDR` and `MASTER_PORT` environment variables
- `sampler.set_epoch(epoch)` for proper shuffling
- Only save checkpoints from rank 0
