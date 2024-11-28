#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cute/tensor.hpp>

#include "cuda_utils.hpp"

template <typename T,
    typename std::enable_if<
    (std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, cute::half_t>::value),
    bool>::type = true>
constexpr cudaDataType_t get_cuda_data_type()
{
    if (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else if (std::is_same<T, cute::half_t>::value)
    {
        return CUDA_R_16F;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }
}

template <typename T,
    typename std::enable_if<
    (std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, cute::half_t>::value),
    bool>::type = true>
void inititalize_random_matrix(T* A , size_t m, size_t n, size_t lda, unsigned int seed = 0U)
{   
    // create a generator
    // std::default_random_engine engine(seed);
    // std::uniform_int_distribution<int> distribution(0, 5);
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    auto const generator = [&engine, &distribution]() {
        return distribution(engine);
    };

    // initialize the matrix
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = static_cast<T>(generator());
        }
    }
}

template <typename T,
    typename std::enable_if<
    (std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, cute::half_t>::value),
    bool>::type = true>
void initialize_zero_matrix(T* A, size_t m, size_t n, size_t lda)
{
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = static_cast<T>(0.0f);
        }
    }
}

// ----------------------------- cublas and cpu -------------------------------
#define CHECK_CUBLASS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
void check_cublas(cublasStatus_t err, const char *const func, const char *const file, int const line);
//  const char *const: a fix pointer (*const) to a fix string (const char)

template <typename T, typename std::enable_if<
    (std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, cute::half_t>::value),
    bool>::type = true>
void launch_gemm_cublas(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cublasHandle_t handle)
{
    constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
    constexpr cudaDataType_t data_type{get_cuda_data_type<T>()};

    CHECK_CUBLASS_ERROR(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            alpha,
            A, data_type, lda,
            B, data_type, ldb,
            beta,
            C, data_type, ldc,
            data_type,
            algo
        )
    );

}


// ------------------------------ print utils ---------------------------------
void print_device_info();

template <typename T>
float compute_effective_bandwidth(size_t m, size_t n, size_t k, float latency)
{
    return (m * n + n * k + m * k) * sizeof(T) / (latency * 1e-3) / 1e9;
    // latency * 1e-3 (ms --> s)
    // 1e9 (bytes --> GB)
}

float compute_effective_tflops(size_t m, size_t n, size_t k, float latency);
void print_performance_results(size_t m, size_t n, size_t k, float latency);
// -----------------------------------------------------------------------------


// ------------------------------- measure utils -------------------------------
float measure_performance(std::function<void(cudaStream_t)> gemm_kernel_launch_function, cudaStream_t stream, size_t num_repeats, size_t num_warmups);

template <typename T>
bool all_close(T const* C, T const* C_ref, size_t m, size_t n, size_t ldc,
               T abs_tol, double rel_tol)   
{   
    // casting all to double to make the comparison accurate
    bool status{true};

    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            double const C_val{static_cast<double>(C[i * ldc + j])};
            double const C_ref_val{static_cast<double>(C_ref[j * ldc + i])};
            double const diff_val{std::abs(C_val - C_ref_val)};
            // if (diff_val > max(abs_tol, rel_tol * abs(C_ref_val)))
            if (diff_val > std::max(static_cast<double>(abs_tol), rel_tol * static_cast<double>(std::abs(C_ref_val))))
            {
                std::cout << "C[" << i << ", " << j << "] = " << C_val << " != " << C_ref_val << std::endl;
                std::cout << "diff = " << diff_val << " > " << std::max(static_cast<double>(abs_tol), rel_tol * static_cast<double>(std::abs(C_ref_val))) << std::endl;
                status = false;
                return status;
            }
        }
    }

    return status;
}
// -----------------------------------------------------------------------------

template <typename T, 
    typename std::enable_if< 
    (std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, cute::half_t>::value),
    bool>::type = true>
std::pair<float, float> profile_gemm(size_t m, size_t n, size_t k,
    size_t lda, size_t ldb, size_t ldc,
    std::function<void(
        size_t, size_t, size_t, 
        const T*, 
        const T*, size_t,
        const T*, size_t, 
        const T*,
        T*, size_t, 
        cudaStream_t
    )> gemm_kernel_launch_function,
    T abs_tol, double rel_tol,
    size_t num_repeats, size_t num_warmups, unsigned int seed = 0U)
{
    // create cuda stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // initialize values for alpha and beta
    T const alpha{static_cast<T>(1.0)};
    T const beta{static_cast<T>(0.0)};

    //  allocate host memory for A, B, C
    T* A_host{nullptr};
    T* B_host{nullptr};
    T* C_host{nullptr};
    T* C_host_ref{nullptr};
    T* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, n * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_ref, m * ldc * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device, m * ldc * sizeof(T)));

    // initialize A, B, C
    inititalize_random_matrix(A_host, m, k, lda, 1);
    // inititalize_random_matrix<T>(A_host, k, n, ldb, seed);
    // explicit spefify T, but we don't need it because the compiler will deduce it from the type of A_host
    inititalize_random_matrix(B_host, n, k, ldb, 11);
    // inititalize_random_matrix(C_host, m, n, ldc, 111);
    initialize_zero_matrix(C_host, m, n, ldc);

    // allocate device memory for A, B, C
    T* A_device{nullptr};
    T* B_device{nullptr};
    T* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, n * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(T)));

    // copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, n * ldb * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(T), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_host, m * ldc * sizeof(T), cudaMemcpyHostToDevice));

    // // create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLASS_ERROR(cublasCreate(&handle));
    CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

    // allclose sanity check
    // compute reference output using cuBLAS
    launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc, handle);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // copy C matrix from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_device, m * ldc * sizeof(T), cudaMemcpyDeviceToHost));


    // compute output using our gemm kernel
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream)); 
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(T), cudaMemcpyHostToDevice));
    gemm_kernel_launch_function(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // copy C matrix from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device, m * ldc * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // check allclose
    assert(all_close<T>(C_host_from_device, C_host_ref, m, n, ldc, abs_tol, rel_tol));

    // speed measurement
    float const cublas_latency{measure_performance(
            [&](cudaStream_t stream) {
            launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc, handle);
            return;
        },
        stream, num_repeats, num_warmups
    )};

    float const gemm_kernel_latency{measure_performance(
            [&](cudaStream_t stream) {
            gemm_kernel_launch_function(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc, stream);
            return;
        }, 
        stream, num_repeats, num_warmups
    )};

    // release resources
    CHECK_CUDA_ERROR(cudaFreeHost(A_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_ref));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    CHECK_CUBLASS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // print results
    std::cout << "cuBLAS gemm kernel performance: " << std::endl;
    print_performance_results(m, n, k, cublas_latency);
    std::cout << "custom gemm kernel performance: " << std::endl;
    print_performance_results(m, n, k, gemm_kernel_latency);
    std::cout << "custom gemm kernel vs. cuBLAS gemm kernel" << std::endl;
    std::cout << (cublas_latency / gemm_kernel_latency * 100.0f) << "%" << std::endl;

    return std::pair<float, float>{cublas_latency, gemm_kernel_latency};
}    