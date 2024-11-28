#include <cute/tensor.hpp>

#include "cuda_gemm.hpp"

// kernel
template <
    class T, 
    class CtaTiler,
    class SmemLayoutA, class SmemLayoutB,
    class TiledMMA,
    class G2STiledCopyA, class G2STiledCopyB,
    class S2RCopyAtomA, class S2RCopyAtomB
>
__global__ void cute_gemm_v01(
    unsigned int M, unsigned int N, unsigned int K,
    const T alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T beta,
    T *C, size_t ldc
)
{
    using namespace cute;

    // global full tensor
    // shape
    auto shape_MNK = make_shape(M, N, K);

    // stride
    // cublas covenience for TN gemm
    // all matrices are in column major
    // A (m,k) --> transpose --> A(k, m) --> cute layout: A (m, k) : (k, 1) --> lda = k
    // B (k,n) --> cute layout: B (n, k) : (k, 1) --> ldb = k
    // C (m,n) --> cute layout: C (m, n) : (1, m) --> ldc = m

    auto dA = make_stride(lda, _1{});
    auto dB = make_stride(ldb, _1{});
    auto dC = make_stride(ldc, _1{});
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M x K
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N x K
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M x N

    // global tile tensor
    auto cta_tiler = CtaTiler{};
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BLOCK_SIZE_M x BLOCK_SIZE_K x NUM_TILES_K
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BLOCK_SIZE_N x BLOCK_SIZE_K x NUM_TILES_K
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BLOCK_SIZE_M x BLOCK_SIZE_N

    // shared memory
    extern __shared__ T smem[];
    T* Asmem = smem;
    T* Bsmem = smem + cosize_v<SmemLayoutA>;

    Tensor sA = make_tensor(make_smem_ptr(Asmem), SmemLayoutA{}); // BLOCK_SIZE_M x BLOCK_SIZE_K
    Tensor sB = make_tensor(make_smem_ptr(Bsmem), SmemLayoutB{}); // BLOCK_SIZE_N x BLOCK_SIZE_K

    // MMA
    // use TiledMMA --> get one thread work
    auto tiled_mma = TiledMMA{};
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgC = thr_mma.partition_C(gC);                     // MMA x MMA_M x MMA_N
    
    // thread private memory for MMA
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));         // MMA x MMA_M x MMA_K
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));         // MMA x MMA_N x MMA_K
    
    // thread private memory for accumulator for MMA
    Tensor tCrC = thr_mma.partition_fragment_C(gC);                // MMA x MMA_M x MMA_N

    clear(tCrC);

    // initiate copy from global memory to shared memory
    // use G2S TiledCopy --> get one thread copy work
    auto g2s_tiled_copy_a = G2STiledCopyA{};
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
    const auto tAgA = g2s_thr_copy_a.partition_S(gA);             // CPY x CPY_M x CPY_K x NUM_TILES_K
    auto tAsA = g2s_thr_copy_a.partition_D(sA);             // CPY x CPY_M x CPY_K

    auto g2s_tiled_copy_b = G2STiledCopyB{};
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    const auto tBgB = g2s_thr_copy_b.partition_S(gB);             // CPY x CPY_N x CPY_K x NUM_TILES_K
    auto tBsB = g2s_thr_copy_b.partition_D(sB);             // CPY x CPY_N x CPY_K
    
    // initiate copy from shared memory to thread private memory
    // use S2R TiledCopy --> get one thread copy work
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    const auto tCsA = s2r_thr_copy_a.partition_S(sA);             // CPY x CPY_M x CPY_K
    auto tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);    // CPY x CPY_M x CPY_K

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    const auto tCsB = s2r_thr_copy_b.partition_S(sB);             // CPY x CPY_N x CPY_K x NUM_STAGES
    auto tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);    // CPY x CPY_N x CPY_K

    int ntile = size<3>(tAgA);

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile)
    {
        // copy  (CPY, CPY_M, CPY_K) , async
        copy(g2s_tiled_copy_a, tAgA(_, _, _, itile),
                tAsA(_, _, _));
        copy(g2s_tiled_copy_b, tBgB(_, _, _, itile),
                tBsB(_, _, _));

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        
        copy(s2r_tiled_copy_a, tCsA, tCrA_copy_view);
        copy(s2r_tiled_copy_b, tCsB, tCrB_copy_view);
        __syncthreads(); // this is super important (I just found it after a few months)
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    } // itile

    copy(tCrC, tCgC);
    // axpby(alpha, tCrC, beta, tCgC);
}

// launch
template <typename T>
void launch_cute_gemm_kernel_v01(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream
)
{
    using namespace cute;

    // problem shape
    unsigned int M = static_cast<unsigned int>(m);
    unsigned int N = static_cast<unsigned int>(n);
    unsigned int K = static_cast<unsigned int>(k);

    // block shape and cta tiler
    // additional dim: NUM_STAGES --> This is for later pipelining the k-slice GEMM
    auto BLOCK_SIZE_M = _256{};
    auto BLOCK_SIZE_N = _128{};
    auto BLOCK_SIZE_K = _64{};
    using CtaTiler = decltype(make_shape(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K));

    // smem layout
    // Swizzle parameters need to be chosen right
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;

    using SmemLayoutAtom = decltype(
        composition(
            Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
            make_layout(
                make_shape(_8{}, BLOCK_SIZE_K),
                LayoutRight{}
            )
        )
    );


    // what does this do?
    // with BLOCK_SIZE_K = 32, shape: (8, 32)
    // 2^M = 8, 1 new unit = 8 units --> 1 row contains 32/8 = 4 new units
    // 2^S = 8, it will treat 1 row = 8 new units --> do 8-unit swizzle
    // 2^B = 8, it will reset the swizzle pattern after 8 rows
    // print_layout(SmemLayoutAtom{});

    // tile_to_shape extends the layout in LayoutLeft order
    using SmemLayoutA = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(BLOCK_SIZE_M, BLOCK_SIZE_K)
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(BLOCK_SIZE_N, BLOCK_SIZE_K)
        )
    );

    // TiledMMA
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    // thread repetition
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    // thread workload repetition
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int MmaVN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    
    // Thread repetition 2x2x1 --> 128 threads
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    // Thread workload repetition 1x2x1
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

    // TiledCopy from global memory to shared memory
    // uint128_t is 16 bytes = 4 floats = 8 halfs
    static constexpr int NUM_VECTOR_UNITS = sizeof(cute::uint128_t) / sizeof(T);

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    // one block contains 128 threads
    // --> find the compatible thread layout
    using G2S_Copy_Thread_Layout = decltype(
        make_layout(
            make_shape(_32{}, _4{}),    // 32x4 = 128 threads
            LayoutRight{}               // A is in row-major
        )
    );

    using G2S_Copy_Value_Layout = decltype(
        make_layout(
            make_shape(_1{}, Int<NUM_VECTOR_UNITS>{})
        )
    );

    // This is for copy shape 32x4 of uint128_t
    using G2STiledCopyA = decltype(
        make_tiled_copy(
            g2s_copy_atom{},
            G2S_Copy_Thread_Layout{},
            G2S_Copy_Value_Layout{}
        )
    );

    // Both A and B are in row-major so use the same TiledCopy for B
    using G2STiledCopyB = G2STiledCopyA;

    // CopyAtom from shared memory to registers
    // Why no need to do tiling atom here? Because we will do it later with the information from TiledMMA
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // grid, block
    dim3 block{size(TiledMMA{}), 1U, 1U};
    dim3 grid{
        size(ceil_div(N, BLOCK_SIZE_N)),
        size(ceil_div(M, BLOCK_SIZE_M)),
        1U
    };

    static constexpr int smem_size = (cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB>) * sizeof(T);
    cudaFuncSetAttribute(
        cute_gemm_v01 <
            T,
            CtaTiler,
            SmemLayoutA, SmemLayoutB,
            TiledMMA,
            G2STiledCopyA, G2STiledCopyB,
            S2RCopyAtomA, S2RCopyAtomB
        >, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // kernel
    cute_gemm_v01 <
        T,
        CtaTiler,
        SmemLayoutA, SmemLayoutB,
        TiledMMA,
        G2STiledCopyA, G2STiledCopyB,
        S2RCopyAtomA, S2RCopyAtomB
    >
    <<<grid, block, smem_size, stream>>>(
        M, N, K,
        *alpha,
        A, lda,
        B, ldb,
        *beta,
        C, ldc
    );
}

// explicit instantiation
template void launch_cute_gemm_kernel_v01<cute::half_t>(
    size_t m, size_t n, size_t k,
    const cute::half_t *alpha,
    const cute::half_t *A, size_t lda,
    const cute::half_t *B, size_t ldb,
    const cute::half_t *beta,
    cute::half_t *C, size_t ldc,
    cudaStream_t stream
);