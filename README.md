# tensor-core-from-scratch

**84% of cuBLAS in 8 self-contained CUDA files. No frameworks. No dependencies. No magic.**

A step-by-step progression from a naive matmul to NVIDIA tensor cores on Blackwell, with every kernel benchmarked against cuBLAS and verified for correctness. ~2,000 lines of CUDA total. Read the code top-to-bottom — that's the whole point.

![Performance Progression](assets/performance.png)

## Why this exists

[Simon Boehm's CUDA matmul guide](https://siboehm.com/articles/22/CUDA-MMM) is the gold standard for CUDA-core optimization — but it stops before tensor cores. [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) has 200+ kernels but reads like a reference library, not a tutorial. This project fills the gap: a progressive, numbered sequence where each kernel introduces **exactly one new concept** and you can see the TFLOPS jump in real time.

## Performance Progression

Measured on **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 188 SMs) with CUDA 12.8.
Problem size: 4096x4096x4096 SGEMM (137.4 GFLOP).

```
Kernel                    TFLOPS   % cuBLAS   What You Learn
---------------------------------------------------------------
01 Naive                    5.67      9.7%    Baseline: 1 thread = 1 output element
02 Memory Coalescing        5.67      9.7%    Thread-to-memory mapping (concept)
03 Shared Memory Tiling     8.15     13.9%    SMEM as a software-managed cache
04 1D Block Tiling         21.68     37.1%    More work per thread
05 2D Block Tiling         33.92     58.0%    Register tiling (the big leap)
06 Vectorized Memory       28.22     48.2%    float4 loads — and why they can hurt
07 WMMA Tensor Cores       48.92     83.6%    First tensor core kernel
08 PTX mma.sync            45.29     77.4%    Raw PTX: full control over registers
---------------------------------------------------------------
   cuBLAS (reference)      58.55    100.0%
```

Yes, kernel 06 is slower than kernel 05. Not every "optimization" helps on every GPU — welcome to performance engineering.

## Quick Start

```bash
git clone https://github.com/waynehacking8/tensor-core-from-scratch.git
cd tensor-core-from-scratch

make ARCH=sm_120      # or sm_90 (Hopper), sm_89 (Ada Lovelace)
make run K=07_wmma_tensor_cores
```

You should see something like this:

```
=== Kernel 07: WMMA Tensor Cores (FP16 input, FP32 accum) ===

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (SM 12.0)
Tensor cores available: YES

Problem size: M=4096, N=4096, K=4096
FLOPs per matmul: 137.44 GFLOP
WMMA tile: 16x16x16
Warps per block: 4x4 = 16

cuBLAS FP32:             2.35 ms    58.50 TFLOPS  (FP32 SGEMM reference)
07_wmma:                 2.83 ms    48.52 TFLOPS  ( 82.9% of cuBLAS FP32)

  07_wmma vs cuBLAS FP32         max=0.0527 avg=0.024089 [PASS]
```

Every kernel verifies element-wise against cuBLAS and prints `[PASS]` or `[FAIL]`. On your GPU, TFLOPS will scale proportionally to your hardware's peak throughput.

Requirements: CUDA Toolkit 12.8+ and an NVIDIA GPU with compute capability 7.0+ (Volta or later for tensor cores).

## What Each Kernel Teaches

### CUDA Core Fundamentals (01-06)

| # | File | Key Concept |
|---|------|-------------|
| 01 | `01_naive.cu` | Baseline matmul. One thread computes one output element. The simplest possible GPU kernel. |
| 02 | `02_global_memory_coalescing.cu` | Thread-to-memory mapping. On Blackwell's large L2, the effect is subtle — but the principle matters in every kernel that follows. |
| 03 | `03_shared_memory_tiling.cu` | Load a tile from DRAM into shared memory once, reuse it BLOCKSIZE times. The fundamental GPU optimization. |
| 04 | `04_1d_blocktiling.cu` | Each thread computes TM=8 elements instead of 1. Arithmetic intensity goes up, memory traffic stays flat. |
| 05 | `05_2d_blocktiling.cu` | Register tiling: each thread computes a TM x TN sub-tile via outer products. This is what gets you past 50% of cuBLAS. |
| 06 | `06_vectorized_memory.cu` | `float4` loads and bank-conflict padding. Can *regress* vs kernel 05 due to register pressure — an honest lesson in GPU optimization. |

### Tensor Core Kernels (07-08)

| # | File | Key Concept |
|---|------|-------------|
| 07 | `07_wmma_tensor_cores.cu` | Your first tensor core kernel. WMMA C++ API, 16x16x16 fragments, FP16 compute with FP32 accumulation. One warp instruction = 8192 FLOPs. |
| 08 | `08_mma_ptx.cu` | Drop to inline PTX assembly. `mma.sync.aligned.m16n8k8` with manual fragment loading. You see exactly how data flows from shared memory to registers to tensor cores. |

## How to Read This Project

Start with `01_naive.cu` and read sequentially. Each step introduces one concept:

1. **01 -> 02**: Same algorithm, different thread mapping -> coalescing concept
2. **02 -> 03**: Add shared memory -> tiling
3. **03 -> 04**: More work per thread (1D) -> arithmetic intensity
4. **04 -> 05**: 2D register tile -> outer-product formulation
5. **05 -> 06**: Wider loads -> why "faster" loads can slow you down
6. **06 -> 07**: CUDA cores to tensor cores -> WMMA
7. **07 -> 08**: WMMA to raw PTX -> register layout exposed

## Roadmap

```
CUDA Core Path              Tensor Core Path
--------------              ----------------
01 -> 02 -> 03              07 (WMMA API)          <- you are here
      |                          |
04 -> 05 -> 06              08 (PTX mma.sync)      <- you are here
                                 |
                            [coming soon]
                            09 Double-buffered WMMA
                            10 Optimized mma.sync
                            11 FP8 precision
```

## Acknowledgments

Inspired by [Andrej Karpathy](https://github.com/karpathy)'s "from scratch" philosophy (micrograd, nanoGPT, llm.c) and [Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)'s CUDA matmul optimization guide. This project starts where Boehm's ends — at tensor cores.

## License

MIT
