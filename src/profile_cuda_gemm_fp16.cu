#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "profile_utils.cuh"
#include "cuda_gemm.hpp"

int main() 
{

    // print device information
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{2U};

    cute::half_t const fp16_abs_tol{__float2half(5.0e-2f)};
    double const fp16_rel_tol{1.0e-1f};

    // constexpr size_t m{4096U};
    // constexpr size_t n{4096U};
    // constexpr size_t k{4096U};

    constexpr size_t m{8192U};
    constexpr size_t n{8192U};
    constexpr size_t k{8192U};

    // cublas covenience for TN gemm
    // all matrices are in column major
    // A (m,k) --> transpose --> A(k, m) --> cute layout: A (m, k) : (k, 1) --> lda = k
    // B (k,n) --> cute layout: B (n, k) : (k, 1) --> ldb = k
    // C (m,n) --> cute layout: C (m, n) : (1, m) --> ldc = m

    constexpr size_t lda{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(m + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= k);
    static_assert(ldc >= m);

    std::cout << "Matrix size: " << m << " x " << n << " x " << k << std::endl;
    std::cout << "Matrix A: " << m << " x " << k << " Leading Dimension Size " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n << " Leading Dimension Size " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n << " Leading Dimension Size " << ldc << std::endl;

    // Define all the gemm kernel launch functions
    std::vector<
        std::pair<
            std::string, 
            std::function<void(size_t, size_t, size_t, 
                                const cute::half_t*,
                                const cute::half_t*, size_t, 
                                const cute::half_t*, size_t,
                                const cute::half_t*,
                                cute::half_t*, size_t,
                                cudaStream_t)>>> const gemm_kernel_launch_functions {
                                    // {"custom cute gemm kernel V00", gemm_v00::launch_cute_gemm_kernel_v00<cute::half_t>},
                                    // {"custom cute gemm kernel V01", gemm_v01::launch_cute_gemm_kernel_v01<cute::half_t>},
                                    // {"custom cute gemm kernel V02", gemm_v02::launch_cute_gemm_kernel_v02<cute::half_t>},
                                    // {"custom cute gemm kernel V03", gemm_v03::launch_cute_gemm_kernel_v03<cute::half_t>},
                                    // {"custom cute hopper gemm kernel V00", gemm_hopper_v00::launch_cute_hopper_gemm_kernel_v00<cute::half_t>},
                                    // {"custom cute hopper gemm kernel V01", gemm_hopper_v01::launch_cute_hopper_gemm_kernel_v01<cute::half_t>},
                                    {"custom cute hopper gemm kernel V02", gemm_hopper_v02::launch_cute_hopper_gemm_kernel_v02<cute::half_t>},
                                    {"custom cute hopper gemm kernel V03", gemm_hopper_v03::launch_cute_hopper_gemm_kernel_v03<cute::half_t>}
                                };

    for (auto gemm_kernel_launch_function : gemm_kernel_launch_functions) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> gemm_kernel_profile_result{
            profile_gemm<cute::half_t>(m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                                fp16_abs_tol, fp16_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }
    return 0;
}

