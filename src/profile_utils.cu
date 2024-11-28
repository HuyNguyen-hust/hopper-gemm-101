#include <cuda_runtime.h>

#include "profile_utils.cuh"
#include "cuda_utils.hpp"

float measure_performance(std::function<void(cudaStream_t)> gemm_kernel_launch_function, cudaStream_t stream, size_t num_repeats, size_t num_warmups)
{   
    // initialize for measurements
    float time;
    cudaEvent_t start, stop;

    // create events
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warm up
    for (size_t i{0}; i < num_warmups; ++i)
    {
        gemm_kernel_launch_function(stream);
    }

    // synchronize after warm up
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // measure
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i) {
        gemm_kernel_launch_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    
    // synchronize after measurement 
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();

    // get time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time / static_cast<float>(num_repeats);
}

void print_device_info() {
    // get device
    int device_id{0}; // because cuda API requires device id to be int, init to 0 is safe
    cudaGetDevice(&device_id);

    // get device properties
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    // print device name
    std::cout << "Device name: " << device_prop.name << std::endl;

    // print global memory size in GB
    float const global_memory_size{static_cast<float>(device_prop.totalGlobalMem / (1 << 30))}; // 1 << 30 bytes = 1 GiB ~ 1 GB 
    std::cout << "Global memory size: " << global_memory_size << " GB" << std::endl;

    // print peak memory bandwidth
    float const peak_bandwidth{static_cast<float>((2.0f * device_prop.memoryClockRate * device_prop.memoryBusWidth / 8) / 1.0e6)}; // 1.0e6 from kHz to Ghz
    std::cout << "Peak memory bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std::endl;
}

float compute_effective_tflops(size_t m, size_t n, size_t k, float latency) { 
    return (2.0 * m * n * k) / (latency * 1e-3) / 1e12;
    // latency * 1e-3 (ms --> s)
    // 1e12 (flops --> tflops)
}

void print_performance_results(size_t m, size_t n, size_t k, float latency) {
    float const effective_bandwidth{
        compute_effective_bandwidth<float>(m, n, k, latency)
    };
    float const effective_tflops{
        compute_effective_tflops(m, n, k, latency)
    };
    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << effective_bandwidth << " GB/s" << std::endl;
    std::cout << "Effective TFLOPs: " << effective_tflops << " tflops" << std::endl;
    std::cout << std::endl;
}

void check_cublas(cublasStatus_t err, const char *const func, const char *const file, int const line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cublas runtime error at:" << file << ":" << line << std::endl;
        std::cerr << "Error Message:" << cublasGetStatusString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }    
}