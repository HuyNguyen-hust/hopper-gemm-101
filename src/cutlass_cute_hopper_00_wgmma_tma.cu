#include "cuda_gemm.hpp"

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>



using namespace cute;

namespace gemm_hopper_v00
{
template <typename T>
struct Params 
{
    int M, N, K;
    T *C;
    const T alpha;
    const T beta;
};

// shared storage
template <
    typename T,
    typename SmemLayoutA,
    typename SmemLayoutB
>
struct SharedStorage
{
    // data storage
    array_aligned<T, cosize_v<SmemLayoutA>, 128> smem_A;
    array_aligned<T, cosize_v<SmemLayoutB>, 128> smem_B;

    // barrier
    uint64_t smem_A_barrier;
    uint64_t smem_B_barrier;
};

// kernel traits
template <
    typename T,
    int kBlockM_,
    int kBlockN_,
    int kBlockK_
>
struct KernelTraits
{
    using Element = T;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;

    // TiledMMA
    using mma_op = decltype(
        SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}
    );
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    // thread repetition
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    // thread workload repetition
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int MmaVN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    // this is for problem shape (64x1x1) x (64x1x1) x (16x1x1) = 64x64x16
    
    // Thread repetition 1x1x1 --> 128 threads
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    // Thread workload repetition 1x1x1
    // Each mode of this shape can be a layout to do permutation on the corresponding layout mode
    using MMA_V_RepeatT = decltype(
        make_shape(Int<MmaVM>{}, Int<MmaVN>{}, Int<MmaVK>{})
    );

    using TiledMMA = decltype(
        make_tiled_mma(
            mma_atom{},
            MMA_EU_RepeatT{},
            MMA_V_RepeatT{}
        )
    );

    // Shared memory layout
    // Start from Layout Atom, e.g., Layout_K_SW128_Atom
    // K: leading dimension is K, M: leading dimension is N
    // SW128 float: Sw<3,4,3> o smem_ptr[32b](unset) o (_8,_32):(_32,_1)
    // SW64  float: Sw<2,4,3> o smem_ptr[32b](unset) o (_8,_16):(_16,_1)
    // SW32  float: Sw<1,4,3> o smem_ptr[32b](unset) o (_8,_8):(_8,_1)
    // SW128 half:  Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
    // SW64  half:  Sw<2,4,3> o smem_ptr[16b](unset) o (_8,_32):(_32,_1)
    // SW32  half:  Sw<1,4,3> o smem_ptr[16b](unset) o (_8,_16):(_16,_1)

    // 32b here means 32 bits = 4 bytes = 1 float
    // 16b here means 16 bits = 2 bytes = 1 half

    // One thing to be careful is the Swizzle config B,M,S
    // For example with SW64 half
    // print would display:         Sw<2,4,3> o smem_ptr[16b](unset) o (_8,_32):(_32,_1)
    // print_layout would display:  Sw<2,3,3> o _0 o (_8,_32):(_32,_1)
    // The difference is the above config 2^M is the number of bytes, not the number of halfs
    // 2^4 bytes = 2^3 halfs
    // Similarly, see the layout for SW128 float
    // print displays:              Sw<3,4,3> o smem_ptr[32b](unset) o (_8,_32):(_32,_1)
    // print_layout displays:       Sw<3,2,3> o _0 o (_8,_32):(_32,_1)
    // 2^4 bytes = 2^2 floats
    // From this we know that these builtin swizzle configs always treat consecutive 16 bytes (8 halfs or 4 floats) as one unit (bc M = 4)
    // And it also views 1 row as of having 8 units (bc S = 3)

    // Now let's see how we have that swizzle layout
    // Take SW64 half as an example
    // Why is it SW<2,4,3> and (_8,_32):(_32,_1)?
    // Here 64 bytes is the swizzle width:
    // 64 bytes = 4 x 16 bytes
    // This will do swizzle on 4 consecutive 16-byte segments, or 4 consecutive of (8 halfs), which is 32 halfs, which is exactly the width of the atom
    // So why is the number or rows is 8?
    // This comes from the number of physical rows of the swizzle config
    // SW<2,4,3> means 2^2 rows, or 4 rows. Why 4?
    // Bc the pattern repeats every 4 rows, with row n having the same pattern as row (n-4).
    // --> the number of physical rows = 4, and the number of logical rows = 8

    // Read the appendix C.4,5,6 from this paper: https://arxiv.org/pdf/2410.20399
    // to see the 32-byte, 64-byte, and 128-byte swizzle for halfs (you can view it as float by combining 2 halfs as 1 float)

    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<T>;

    using SmemLayoutA = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<kBlockM>{}, Int<kBlockK>{})
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<kBlockN>{}, Int<kBlockK>{})
        )
    );

    // SharedStorage
    using SharedStorage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;

    // smem_size
    static constexpr int smem_size = sizeof(SharedStorage);
};

// kernel
template <
    typename ParamsT,
    typename Kernel_traits,
    typename TmaLoadA,
    typename TmaLoadB

>
__global__ void cute_hopper_gemm_v00(
    ParamsT params,
    CUTE_GRID_CONSTANT TmaLoadA const tma_load_A,
    CUTE_GRID_CONSTANT TmaLoadB const tma_load_B
)
{   
    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kBlockK = Kernel_traits::kBlockK;

    // global full tensor
    Tensor mA = tma_load_A.get_tma_tensor(make_shape(params.M, params.K));
    Tensor mB = tma_load_B.get_tma_tensor(make_shape(params.N, params.K));
    Tensor mC = make_tensor(
        make_gmem_ptr(params.C),
        make_shape(params.M, params.N),
        // make_stride(params.N, _1{})
        make_stride(_1{}, params.M)
    );

    // tiling
    auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    Tensor gA = local_tile(
        mA,
        cta_tiler,
        cta_coord,
        Step<_1, X, _1>{}
    ); // kBlockM x kBlockK x NUM_TILES_K
    Tensor gB = local_tile(
        mB,
        cta_tiler,
        cta_coord,
        Step<X, _1, _1>{}
    );  // kBlockN x kBlockK x NUM_TILES_K
    Tensor gC = local_tile(
        mC,
        cta_tiler,
        cta_coord,
        Step<_1, _1, X>{}
    );  // kBlockM x kBlockN

    // smem
    using SharedStorage = typename Kernel_traits::SharedStorage;
    extern __shared__ char smem_[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    // smem tensors
    Tensor sA = make_tensor(
        make_smem_ptr(shared_storage.smem_A.data()),
        SmemLayoutA{}
    );  // kBlockM x kBlockK
    Tensor sB = make_tensor(
        make_smem_ptr(shared_storage.smem_B.data()),
        SmemLayoutB{}
    ); // kBlockN x kBlockK

    // // debug
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {

    //     // print params
    //     printf("M = %d, N = %d, K = %d\n", params.M, params.N, params.K);
    //     printf("kBlockM = %d, kBlockN = %d, kBlockK = %d\n", kBlockM, kBlockN, kBlockK);
    
    //     print(mA);
    //     printf("\n");
    //     print(mB);
    //     printf("\n");
    //     print(mC);
    //     printf("\n");
    //     print(gA);
    //     printf("\n");
    //     print(gB);
    //     printf("\n");
    //     print(gC);
    //     printf("\n");
    //     print(sA);
    //     printf("\n");
    //     print(sB);
    //     printf("\n");
    // }

    // copy parition
    auto [tAgA, tAsA] = tma_partition(
        tma_load_A, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sA), group_modes<0,2>(gA)
    ); // (TMA,k) and (TMA)
    auto [tBgB, tBsB] = tma_partition(
        tma_load_B, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sB), group_modes<0,2>(gB)
    );  // (TMA,k) and (TMA)

    // mma partition
    typename Kernel_traits::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // allocate "fragments"
    // note that in Ampere, the fragments are physically stored in registers
    // but in Hopper, the fragments are not, the tCrA and tCrB are actually the iterators not the data, but still in registers
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // iterator (1, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // iterator (1, MMA_N, MMA_K)

    // debug
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     print(tCsA);
    //     printf("\n");
    //     print(tCsB);
    //     printf("\n");
    //     print(tCrA);
    //     printf("\n");
    //     print(tCrB);
    //     printf("\n");
    //     print(tCgC);
    //     printf("\n");
    //     print(tCrC);
    //     printf("\n");
    // }

    // Initialize barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;
    using T = typename Kernel_traits::Element;
    constexpr int kTmaTransactionBytesA = sizeof(ArrayEngine<T, size(SmemLayoutA{})>); // must be a multiple of 16
    constexpr int kTmaTransactionBytesB = sizeof(ArrayEngine<T, size(SmemLayoutB{})>); // must be a multiple of 16

    uint64_t& smem_A_barrier = shared_storage.smem_A_barrier;
    uint64_t& smem_B_barrier = shared_storage.smem_B_barrier;

    if (warp_idx == 0 && lane_predicate)
    {
        TransactionBarrier::init(&smem_A_barrier, 1);
        TransactionBarrier::init(&smem_B_barrier, 1);
    }

    __syncthreads();

    auto NUM_TILES_K = size<2>(gA);

    // // debug
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("kTmaTransactionBytesA = %d\n", kTmaTransactionBytesA);
    //     printf("kTmaTransactionBytesB = %d\n", kTmaTransactionBytesB);
    //     printf("NUM_TILES_K = %d\n", NUM_TILES_K);
    //     print(tAgA);
    //     printf("\n");
    //     print(tAsA);
    //     printf("\n");
    //     print(tBgB);
    //     printf("\n");
    //     print(tBsB);
    //     printf("\n");
    // }
    
    #pragma unroll 1
    for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile) {

        // load A and B tiles
        if (warp_idx == 0 && lane_predicate)
        {
            TransactionBarrier::arrive_and_expect_tx(&smem_A_barrier, kTmaTransactionBytesA);
            TransactionBarrier::arrive_and_expect_tx(&smem_B_barrier, kTmaTransactionBytesB);
            copy(tma_load_A.with(smem_A_barrier), tAgA(_, k_tile), tAsA);
            copy(tma_load_B.with(smem_B_barrier), tBgB(_, k_tile), tBsB);
        }

        TransactionBarrier::wait(&smem_A_barrier, k_tile%2);
        TransactionBarrier::wait(&smem_B_barrier, k_tile%2);
        // phase flips between 0 and 1 every load

        // Fence to ensure inputs are ready, just like syncthreads bef
        cute::warpgroup_arrive();

        // compute
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        // Commit batch group of gemm
        cute::warpgroup_commit_batch();
        // - Groups all previous uncommitted WGMMAs
        // - One group per warpgroup

        // Wait for completion
        cute::warpgroup_wait<0>();
        // - 0 means wait for ALL previous groups
        // - Can use other values for overlap

        // the commit and wait are kinda similar to the async copy complete mechanism on SM80
    }

    axpby(params.alpha, tCrC, params.beta, tCgC);
}

// launch
template<typename T>
void launch_cute_hopper_gemm_kernel_v00(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream
)
{   
    using ParamsT = Params<T>;
    ParamsT params = {int(m), int(n), int(k), C, *alpha, *beta};

    // Block shape and cta tiler
    constexpr int kBlockM_ = 256;
    constexpr int kBlockN_ = 128;
    constexpr int kBlockK_ = 64;

    using Kernel_traits = KernelTraits<T, kBlockM_, kBlockN_, kBlockK_>;

    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    using TiledMMA = typename Kernel_traits::TiledMMA;

    // TMA copy (G2S): SM90_TMA_LOAD{} + gmem tensor + smem layout

    // Global memory tensor
    // Stride
    // Cublas covenience for TN gemm 
    // At first cublas has A (m,k), B (k,n), C (m,n)
    // All matrices are in column major
    // A (m,k) --> transpose --> A(k, m) --> cute layout: A (m, k) : (k, 1) --> lda = k
    // B (k,n) --> cute layout: B (n, k) : (k, 1) --> ldb = k
    // C (m,n) --> cute layout: C (m, n) : (1, m) --> ldc = m

    Tensor mA = make_tensor(
        make_gmem_ptr(A),
        make_shape(params.M, params.K),
        make_stride(lda, _1{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(B),
        make_shape(params.N, params.K),
        make_stride(ldb, _1{})
    );

    // Finally we create tma_load
    auto tma_load_A = make_tma_copy(
        SM90_TMA_LOAD{},
        mA,
        SmemLayoutA{}
    );

    auto tma_load_B = make_tma_copy(
        SM90_TMA_LOAD{},
        mB,
        SmemLayoutB{}
    );

    // Launch parameter setup
    constexpr int smem_size = Kernel_traits::smem_size;
    dim3 block{cute::size(TiledMMA{}), 1U, 1U};
    dim3 cluster{1, 1, 1};
    dim3 grid{
        cute::size(ceil_div(params.N, kBlockN_)),
        cute::size(ceil_div(params.M, kBlockM_)),
        1U
    };

    cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size, stream};

    void const* kernel = reinterpret_cast<void const*>(&cute_hopper_gemm_v00 <
            ParamsT,
            Kernel_traits,
            decltype(tma_load_A),
            decltype(tma_load_B)
        >
    );

    if (smem_size >= 48 * 1024) // 48KB
    {
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // kernel launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        kernel,
        params,
        tma_load_A,
        tma_load_B
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Kernel launch failed with status: " << std::endl;
    }

}

// explicit instantiation                      
template void launch_cute_hopper_gemm_kernel_v00<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);                                    

} // namespace gemm_hopper_v00