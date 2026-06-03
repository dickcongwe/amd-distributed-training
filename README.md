# 🌐 Distributed Training on AMD Cluster

Multi-node distributed training framework for AMD ROCm GPUs.

## Overview

Train massive models across multiple AMD GPU nodes using:
- PyTorch DistributedDataParallel (DDP)
- DeepSpeed ZeRO-3
- FSDP (Fully Sharded Data Parallel)
- RCCL (ROCm Communication Collectives)

## Architecture

```
┌─────────────────────────────────────────────┐
│              Master Node                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │    │
│  │ MI250X  │  │ MI250X  │  │ MI250X  │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
│       └────────────┼────────────┘           │
│              InfiniBand / RoCE              │
├─────────────────────────────────────────────┤
│              Worker Node                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │    │
│  │ MI250X  │  │ MI250X  │  │ MI250X  │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
│       └────────────┼────────────┘           │
└─────────────────────────────────────────────┘
```

## Requirements

- 2+ nodes with AMD Instinct GPUs
- InfiniBand or RoCE interconnect
- ROCm 6.0+ on all nodes
- Shared filesystem (NFS/GPFS)

## Usage

```bash
# 2-node, 4 GPUs each
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 \
    train.py --model llama-70b --distributed deepspeed
```

## Scaling Results

Llama 3 70B training on MI250X cluster:

| Nodes | GPUs | Throughput | Scaling Efficiency |
|-------|------|------------|-------------------|
| 1 | 4 | 4,800 tok/s | 100% |
| 2 | 8 | 9,200 tok/s | 95.8% |
| 4 | 16 | 17,600 tok/s | 91.7% |
| 8 | 32 | 33,000 tok/s | 85.9% |

## License

MIT
