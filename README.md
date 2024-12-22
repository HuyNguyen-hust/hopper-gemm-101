# cute-gemm-101

## Introduction

This project is for self-learning purposes, aiming to understand and implement GEMM optimizations on Nvidia Hopper GPU using Cutlass CuTe. Through a series of optimizations, this project has successfully achieved performance levels matching or exceeding cuBLAS for half-precision (half) computations in specific configurations.

## Detail Implementation
Utilizes Cutlass CuTe for efficient GEMM implementation, providing a step-by-step approach to GEMM optimization

0. First 4 Cutlass CuTe kernels are specific for Nvidia Ampere GPUs, which underperform on Hopper GPUs
1. Hopper kernel V00: Use TMA Load/Store and WGMMA 
2. Hopper Kernel V01: Add Pipeline
3. Hopper Kernel V02: Add Async Store (from Smem to Gmem)
4. Hopper Kernel V03: Persistent Kernel
5. Hopper Kernel V04: Take advantage of Multicast Load/Store

## Building and Running

```
git submodule init
git submodule update
cmake -B build
cmake --build build
./build/src/profile_cuda_gemm_fp16
```

## Performance

Performance tests were conducted on an NVIDIA H100 with 79 GB of global memory and a peak memory bandwidth of 3352.32 GB/s. The problem size used for testing was MxNxK = 8192 x 8192 x 8192.

### Half Precision (FP16) Performance

This table summarizes the performance of various GEMM kernels compared to cuBLAS on matrix size `8192 x 8192 x 8192`.

| Kernel Version                            | Latency (ms) | Effective Bandwidth (GB/s) | Effective TFLOPs | % of cuBLAS |
|-------------------------------------------|--------------|----------------------------|------------------|-------------|
| **cuBLAS**                                | 1.31533      | 612.248                    | 835.922          | 100%        |
| **Custom Cute GEMM Kernel V00**           | 13.9279      | 57.8195                    | 78.9429          | 9.44%       |
| **Custom Cute GEMM Kernel V01**           | 2.51184      | 320.604                    | 437.732          | 52.31%      |
| **Custom Cute GEMM Kernel V02**           | 2.12285      | 379.352                    | 517.942          | 62.09%      |
| **Custom Cute GEMM Kernel V03**           | 2.30998      | 348.62                     | 475.982          | 56.87%      |
| **Custom Cute Hopper GEMM Kernel V00**    | 1.77392      | 453.97                     | 619.82           | 74.11%      |
| **Custom Cute Hopper GEMM Kernel V01**    | 1.44074      | 558.955                    | 763.16           | 91.45%      |
| **Custom Cute Hopper GEMM Kernel V02**    | 1.42285      | 565.982                    | 772.754          | 92.55%      |
| **Custom Cute Hopper GEMM Kernel V03**    | 1.31757      | 611.207                    | 834.501          | 99.63%      |
| **Custom Cute Hopper GEMM Kernel V03 (No Epilogue)** | 1.54502      | 521.226                    | 711.647          | 85.15%      |
| **Custom Cute Hopper GEMM Kernel V04**    | 1.30429      | 617.43                     | 842.998          | 100.96%     |

## Notes
- All performance metrics are compared against the **cuBLAS** kernel as the baseline (100%).
- **Latency** is the time taken to execute the kernel.
- **Effective Bandwidth** and **Effective TFLOPs** represent the kernel's execution efficiency.
- **Custom Cute Hopper GEMM Kernel V04** outperforms cuBLAS, achieving **100.96%** of cuBLAS Effective TFLOPs.

## Acknowledgements

This project is based on the work of Colfax Research: [Colfax Research Repository](https://github.com/ColfaxResearch/cfx-article-src). The implementation heavily relies on the Cutlass CuTe library, and I encourage readers to refer to the [official Cutlass CuTe documentation](https://github.com/NVIDIA/cutlass/tree/master/media/docs/cute) for more detailed information on the techniques used.
