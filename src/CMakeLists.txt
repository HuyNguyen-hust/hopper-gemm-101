# set cmake version
cmake_minimum_required(VERSION 3.20.0)

# set project name
project(cute-gemm-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# find cuda toolkit
find_package(CUDAToolkit REQUIRED)

# find include dir of this project
find_path(CUDA_GEMM_INCLUDE_DIR cuda_gemm.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
set(CUTLASS_CUTE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/cutlass/include)
set(CUTLASS_UTIL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/cutlass/tools/util/include)
# CUDA_GEMM_INCLUDE_DIR == /include

# add libraries
add_library(cuda_gemm
            SHARED
            cutlass_cute_00_block_tiling_thread_tiling_without_shared_memory.cu
            cutlass_cute_01_block_tiling_thread_tiling_with_swizzle_shared_memory.cu
            cutlass_cute_02_block_tiling_thread_tiling_with_swizzle_shared_memory_pipeline.cu
            cutlass_cute_03_block_tiling_thread_tiling_with_swizzle_shared_memory_pipeline_epilogue.cu
            cutlass_cute_hopper_00.cu
            cutlass_cute_hopper_01_pipeline.cu
            cutlass_cute_hopper_02_pipeline_epilogue.cu
            profile_utils.cu
            cuda_utils.cu)
# include directories for cuda_gemm
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIR} ${CUTLASS_CUTE_INCLUDE_DIR}, ${CUTLASS_UTIL_INCLUDE_DIR})
# set gpu properties
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES 90a)

# add executables
add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
# link libraries
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)
# set gpu properties
set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES 90a)