#include <iostream>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

// cuda_utils
void check_cuda_error(cudaError_t err, const char *const func, const char *const file, int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Cuda runtime error at:" << file << ":" << line << std::endl;
        std::cerr << "Error Message:" << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_last_cuda_error(const char *const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "Cuda runtime error at:" << file << ":" << line << std::endl;
        std::cerr << "Error Message:" << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}