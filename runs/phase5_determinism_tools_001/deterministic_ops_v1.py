"""Explicit CPU-FP32 deterministic operators for a NEW numerical protocol.

Copied from the preserved V27 sampler before editing. Device copies remain in
its autograd graph; no memory/grid gradient is detached or suppressed. CPU vs
CUDA floating point results need not be bitwise equal. Pooling converts to
FP32 on CPU and casts back to the input dtype before returning to its device.
"""
import torch
from torch.nn import functional as F


def sample_joint_neighborhood(memory, grid, valid, radius=1.):
    """Sample 3x3 feature neighborhoods; each invalid location contributes zero."""
    height, width = memory.shape[-2:]
    offsets = grid.new_tensor([(x*2./width, y*2./height)
                              for y in (-1,0,1) for x in (-1,0,1)]) * radius
    locations = grid[:,:,None,:] + offsets
    mask = valid[:,:,None] & (locations.abs() <= 1).all(-1)
    locations = torch.where(mask[...,None], locations, torch.zeros_like(locations))
    # FP32 grid_sample supports both mixed precision and CPU regression checks.
    samples = F.grid_sample(memory.float().to('cpu'), locations.float().to('cpu'), mode='bilinear',
                            padding_mode='border', align_corners=False).to(memory.device)
    samples = samples.permute(0,2,3,1) * mask[...,None]
    return samples, mask


def adaptive_avg_pool2d_cpu(value, output_size):
    """Preserve adaptive bins and output dtype, explicitly execute in CPU FP32."""
    return F.adaptive_avg_pool2d(value.float().to('cpu'), output_size).to(
        device=value.device, dtype=value.dtype)
