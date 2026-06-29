# FA2 Triton

This repository contains a CS336 systems project exploring FlashAttention-2 style attention kernels in Triton, plus benchmark and distributed-training experiments from the same assignment work.

## Layout

- `src/fa2_triton/core.py`: main Triton FlashAttention-2 kernels, autograd wrappers, PyTorch baseline, and a simple attention benchmark.
- `src/fa2_triton/reference.py`: a Python/PyTorch reference implementation kept for readability and debugging.
- `benchmarks/`: MHA benchmark scripts, including fused backward and mixed-precision variants.
- `experiments/`: CS336 side experiments for model benchmarking, optimizer/DDP, and kernel drafts.
- `scripts/check_cuda.py`: quick CUDA/PyTorch environment check.
- `profiles/` and `outputs/`: local profiler/Hydra outputs. These are ignored by git.

## Install

```bash
pip install -e .
```

For benchmark-only dependencies:

```bash
pip install -e ".[bench]"
```

## Quick Usage

```python
import torch
from fa2_triton import triton_fa2

q = torch.randn(2, 1024, 64, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 64, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 64, device="cuda", requires_grad=True)

out = triton_fa2(q, k, v, is_causal=True)
out.sum().backward()
```

## Benchmarks

Run the standalone attention benchmark in the core module:

```bash
python -m fa2_triton.core
```

Run the MHA benchmark variants:

```bash
python benchmarks/mha_benchmark.py
python benchmarks/mha_benchmark_fused_bwd.py
python benchmarks/mha_benchmark_fused_bwd_mixed_precision.py
```

These scripts require a CUDA GPU and a PyTorch/Triton environment that supports the target architecture.

## Notes

The original working directory contained generated profiler reports, sqlite exports, Hydra outputs, and Python cache files. They are intentionally excluded from version control so the GitHub repository stays focused on source code and reproducible scripts.
