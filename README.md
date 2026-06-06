# tensor-core-from-scratch

From a naive matmul to tensor cores on NVIDIA Blackwell — step by step.

Each kernel is a **self-contained `.cu` file** that compiles with a single `nvcc` command, benchmarks itself against cuBLAS, and verifies correctness. No frameworks, no dependencies beyond CUDA.

## Performance Progression

Measured on **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 188 SMs) with CUDA 12.8.
Problem size: 4096×4096×4096 SGEMM (137.4 GFLOP).

```
Kernel                    TFLOPS   % cuBLAS   What You Learn
─────────────────────────────────────────────────────────────────
01 Naive                    5.67      9.7%     Baseline: one thread per output element
02 Memory Coalescing        5.67      9.7%     Thread-to-memory mapping (concept)
03 Shared Memory Tiling     8.15     13.9%     SMEM as a software-managed cache
04 1D Block Tiling         21.68     37.1%     Thread-level work granularity
05 2D Block Tiling         33.92     58.0%     Register tiling — the key to arithmetic intensity
06 Vectorized Memory       28.22     48.2%     float4 loads (not always faster — see comments)
07 WMMA Tensor Cores       48.92     83.6%     First tensor core kernel (WMMA API)
08 PTX mma.sync            45.29     77.4%     Raw PTX: register layout, mma.sync instruction
─────────────────────────────────────────────────────────────────
   cuBLAS (reference)      58.55    100.0%
```

## Quick Start

```bash
# Build all kernels
make ARCH=sm_120   # or sm_90 for Hopper, sm_89 for Ada Lovelace

# Run a specific kernel
make run K=01_naive

# Run all kernels in sequence
make run-all
```

Requirements: CUDA Toolkit 12.8+ and an NVIDIA GPU with compute capability 7.0+ (Volta or later for tensor cores).

## What Each Kernel Teaches

### CUDA Core Fundamentals (01–06)

| # | File | Key Concept |
|---|------|-------------|
| 01 | `01_naive.cu` | Baseline matmul: 1 thread → 1 output element. The simplest possible GPU kernel. |
| 02 | `02_global_memory_coalescing.cu` | Thread-to-memory mapping concept. On modern GPUs with large L2 caches, the effect is subtle — but the principle is critical for later kernels. |
| 03 | `03_shared_memory_tiling.cu` | Using shared memory as a software-managed cache. Load once from DRAM, reuse BLOCKSIZE times. |
| 04 | `04_1d_blocktiling.cu` | Each thread computes TM elements instead of 1. First taste of increased arithmetic intensity. |
| 05 | `05_2d_blocktiling.cu` | Register tiling: each thread computes a TM×TN sub-tile. This is what gets you to 50%+ of cuBLAS. |
| 06 | `06_vectorized_memory.cu` | `float4` loads and bank-conflict padding. May regress vs kernel 05 on some GPUs due to register pressure — a real-world lesson. |

### Tensor Core Kernels (07–08)

| # | File | Key Concept |
|---|------|-------------|
| 07 | `07_wmma_tensor_cores.cu` | Your first tensor core kernel using the WMMA C++ API. 16×16×16 matrix fragments in FP16→FP32. |
| 08 | `08_mma_ptx.cu` | Drop to inline PTX assembly. `mma.sync.aligned.m16n8k8` with manual fragment loading. Full control over register layout. |

## Architecture

```
                    CUDA Core Path              Tensor Core Path
                    ─────────────               ────────────────
                    01 → 02 → 03                07 (WMMA API)
                          ↓                          ↓
                    04 → 05 → 06                08 (PTX mma.sync)
                                                     ↓
                                                [coming soon]
                                                09 WMMA + double buffering
                                                10 mma.sync optimized
```

## Project Structure

```
tensor-core-from-scratch/
├── src/
│   ├── 01_naive.cu
│   ├── 02_global_memory_coalescing.cu
│   ├── 03_shared_memory_tiling.cu
│   ├── 04_1d_blocktiling.cu
│   ├── 05_2d_blocktiling.cu
│   ├── 06_vectorized_memory.cu
│   ├── 07_wmma_tensor_cores.cu
│   └── 08_mma_ptx.cu
├── Makefile
└── README.md
```

## How to Read This Project

Start with `01_naive.cu` and read sequentially. Each kernel builds on the previous one. The progression is designed so that each step introduces exactly one new concept:

1. **01→02**: Same algorithm, different thread mapping → teaches memory coalescing
2. **02→03**: Add shared memory → teaches tiling
3. **03→04**: More work per thread (1D) → teaches arithmetic intensity
4. **04→05**: 2D register tile → teaches the outer-product formulation
5. **05→06**: Wider loads → teaches memory-level parallelism
6. **06→07**: Switch from CUDA cores to tensor cores → teaches WMMA
7. **07→08**: Replace WMMA with raw PTX → teaches register layout

## Acknowledgments

Inspired by [Andrej Karpathy](https://github.com/karpathy)'s "from scratch" educational philosophy (micrograd, nanoGPT, llm.c) and [Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)'s step-by-step CUDA matmul optimization guide.

## License

MIT
