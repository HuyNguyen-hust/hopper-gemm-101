#pragma once
#include <cuda_runtime.h>

namespace gemm_v00
{
template <typename T>
void launch_cute_gemm_kernel_v00(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_v01
{
template <typename T>
void launch_cute_gemm_kernel_v01(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_v02
{
template <typename T>
void launch_cute_gemm_kernel_v02(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_v03
{
template <typename T>
void launch_cute_gemm_kernel_v03(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_hopper_v00
{
template <typename T>
void launch_cute_hopper_gemm_kernel_v00(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_hopper_v01
{
template <typename T>
void launch_cute_hopper_gemm_kernel_v01(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_hopper_v02
{
template <typename T>
void launch_cute_hopper_gemm_kernel_v02(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_hopper_v03
{
template <typename T>
void launch_cute_hopper_gemm_kernel_v03(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

namespace gemm_hopper_v04
{
template <typename T>
void launch_cute_hopper_gemm_kernel_v04(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream);
}

