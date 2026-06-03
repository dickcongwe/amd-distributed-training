#!/usr/bin/env python3
"""Distributed Training on AMD ROCm GPU Clusters"""
import torch, torch.nn as nn, torch.distributed as dist, argparse, os
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_method("nccl")
    torch.cuda.set_device(rank)

def train(rank, world_size, args):
    setup(rank, world_size)
    
    model = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10)).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        x = torch.randn(args.batch, 1024, device=rank)
        y = torch.randint(0, 10, (args.batch,), device=rank)
        
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"GPUs: {world_size}")
    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size)
