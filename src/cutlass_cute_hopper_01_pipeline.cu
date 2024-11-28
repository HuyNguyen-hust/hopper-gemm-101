#include "cuda_gemm.hpp"

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>

using namespace cute;

// shared storage
template <
    typename T,
    int PIPE,
    typename SmemLayoutA,
    typename SmemLayoutB
>
struct SharedStorage
{
    // data storage
    array_aligned<T, cosize_v<SmemLayoutA>, 128> smem_A;
    array_aligned<T, cosize_v<SmemLayoutB>, 128> smem_B;

    // pipeline
    typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline;
};

// kernel traits
template <
    typename T,
    int kWarps_,
    int kBlockM_,
    int kBlockN_,
    int kBlockK_,
    int kStages_
>
struct KernelTraits
{
    using Element = T;
    
    static constexpr int kWarps = kWarps_;
    static_assert(kWarps == 12, "Only support 12 warps now");
    static constexpr int kWarpGroups = kWarps / 4;
    static constexpr int kConsumerWGs = kWarpGroups - 1;
    static constexpr int kThreads = kWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kStages = kStages_;

    // TiledMMA
    // using mma_op = decltype(
    //     GMMA::ss_op_selector <
    //         Element, Element, Element,
    //         Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>
    //     >()
    // );
    using mma_op = decltype(
        SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}
    );
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    // thread repetition
    static constexpr int kMmaEURepeatM = kConsumerWGs;
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
    // From this we know that these builtin swizzle configs always treat consecutive 16 bytes (8 halfs or 4 floats) as one unit (bc M = 4 always)
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
            make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{})
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<kStages>{})
        )
    );

    // SharedStorage
    using SharedStorage = SharedStorage<T, kStages, SmemLayoutA, SmemLayoutB>;

    // smem_size
    static constexpr int smem_size = sizeof(SharedStorage);

    // MainloopPipeline
    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
};

// collective mainloop
template <typename Kernel_traits>
struct CollectiveMainloop
{
    // 1. extract Kernel_traits
    using Element = typename Kernel_traits::Element;

    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kBlockN = Kernel_traits::kBlockN;
    static constexpr int kBlockK = Kernel_traits::kBlockK;

    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;

    using TiledMMA = typename Kernel_traits::TiledMMA;

    // 2. decltype of TMA desc
    using ShapeT = Shape<int32_t, int32_t>;
    using StrideT = Shape<int32_t, _1>;
    // using StrideCT = Shape<_1, int32_t>;
    using StrideCT = Shape<int32_t, _1>;
    using LayoutT = Layout<ShapeT, StrideT>;
    using LayoutCT = Layout<ShapeT, StrideCT>;

    using TmaLoadA = decltype(
        make_tma_copy(
            SM90_TMA_LOAD{},
            make_tensor(
                make_gmem_ptr(static_cast<Element const*>(nullptr)),
                ShapeT{},
                StrideT{}
            ),
            SmemLayoutA{}(_, _, 0)
        )
    );
    using TmaLoadB = decltype(
        make_tma_copy(
            SM90_TMA_LOAD{},
            make_tensor(
                make_gmem_ptr(static_cast<Element const*>(nullptr)),
                ShapeT{},
                StrideT{}
            ),
            SmemLayoutB{}(_, _, 0)
        )
    );

    // 3. set the TMA transaction bytes
    static constexpr int kTmaTransactionBytesA = sizeof(ArrayEngine<Element, size(SmemLayoutA{}(_, _, 0))>);
    static constexpr int kTmaTransactionBytesB = sizeof(ArrayEngine<Element, size(SmemLayoutB{}(_, _, 0))>);
    static constexpr int kTmaTransactionBytes = kTmaTransactionBytesA + kTmaTransactionBytesB; // must be a multiple of 16

    // 4. setup the TMA desc (which replace the need of Params in hopper_00)
    // Host-side kernel arguments
    struct Arguments
    {
        Element const* A;
        Element const* B;
        Element *C;
        LayoutT gmemLayoutA;
        LayoutT gmemLayoutB;
        LayoutCT gmemLayoutC;
    };

    // Device-side kernel params
    struct Params
    {
        LayoutT gmemLayoutA;
        LayoutT gmemLayoutB;
        LayoutCT gmemLayoutC;
        TmaLoadA tma_load_A;
        TmaLoadB tma_load_B;
        Element *C;
    };

    static Params
    to_underlying_arguments(const Arguments& args)
    {
        Tensor mA = make_tensor(
            make_gmem_ptr(args.A),
            args.gmemLayoutA
        );

        Tensor mB = make_tensor(
            make_gmem_ptr(args.B),
            args.gmemLayoutB
        );

        TmaLoadA tma_load_A = make_tma_copy(
            SM90_TMA_LOAD{},
            mA,
            SmemLayoutA{}(_, _, 0)
        );
        TmaLoadB tma_load_B = make_tma_copy(
            SM90_TMA_LOAD{},
            mB,
            SmemLayoutB{}(_, _, 0)
        );

        return
        {
            args.gmemLayoutA,
            args.gmemLayoutB,
            args.gmemLayoutC,
            tma_load_A,
            tma_load_B,
            args.C
        };
    }

    // 5. prefetch TMA desc
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params)
    {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_A.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_B.get_tma_descriptor());
    }

    // 6. producer
    using MainloopPipeline = typename Kernel_traits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;

    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void load
    (
        Params const& mainloop_params,
        MainloopPipeline pipeline,
        PipelineState& write_state,
        SharedStorage& shared_storage,
        int NUM_TILES_K
    )
    {
        // gmem tensors
        Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(mainloop_params.gmemLayoutA.shape());
        Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(mainloop_params.gmemLayoutB.shape());

        // tiling
        auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
        auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
        Tensor gA = local_tile(
            mA,
            cta_tiler,
            cta_coord,
            Step<_1, X, _1>{}
        );  // kBlockM x kBlockK x NUM_TILES_K
        Tensor gB = local_tile(
            mB,
            cta_tiler,
            cta_coord,
            Step<X, _1, _1>{}
        );  // kBlockN x kBlockK x NUM_TILES_K

        // smem tensors
        Tensor sA = make_tensor(
            make_smem_ptr(shared_storage.smem_A.data()),
            SmemLayoutA{}
        );  // kBlockM x kBlockK x PIPE
        Tensor sB = make_tensor(
            make_smem_ptr(shared_storage.smem_B.data()),
            SmemLayoutB{}
        ); // kBlockN x kBlockK x PIPE

        // copy partition
        auto [tAgA, tAsA] = tma_partition(
            mainloop_params.tma_load_A,
            _0{}, Layout<_1>{},
            group_modes<0,2>(sA),
            group_modes<0,2>(gA)
        ); // (TMA, NUM_TILES_K) and (TMA, PIPE)
        auto [tBgB, tBsB] = tma_partition(
            mainloop_params.tma_load_B,
            _0{}, Layout<_1>{},
            group_modes<0,2>(sB),
            group_modes<0,2>(gB)
        ); // (TMA, NUM_TILES_K) and (TMA, PIPE)

        int lane_predicate = cute::elect_one_sync();

        // copy
        if (lane_predicate)
        {
            #pragma unroll 1
            for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile)
            {
                pipeline.producer_acquire(write_state);
                // empty_barrier.wait()
                // full_barrier.arrive_and_expect_tx()
                uint64_t* full_barrier = pipeline.producer_get_barrier(write_state);

                auto stage = write_state.index();

                copy(mainloop_params.tma_load_A.with(*full_barrier, 0), tAgA(_, k_tile), tAsA(_, stage));
                copy(mainloop_params.tma_load_B.with(*full_barrier, 0), tBgB(_, k_tile), tBsB(_, stage));

                ++write_state;
            }
        }
    }

    // 7. consumers
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void mma
    (
        Params const& mainloop_params,
        MainloopPipeline pipeline,
        PipelineState& read_state,
        SharedStorage& shared_storage,
        int NUM_TILES_K
    )
    {
        // gmem tensors
        Tensor mC = make_tensor(
            make_gmem_ptr(mainloop_params.C),
            mainloop_params.gmemLayoutC  
        );
        
        // tiling
        auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
        auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
        Tensor gC = local_tile(
            mC,
            cta_tiler,
            cta_coord,
            Step<_1, _1, X>{}
        );

        // smem tensors
        Tensor sA = make_tensor(
            make_smem_ptr(shared_storage.smem_A.data()),
            SmemLayoutA{}
        );  // kBlockM x kBlockK x PIPE
        Tensor sB = make_tensor(
            make_smem_ptr(shared_storage.smem_B.data()),
            SmemLayoutB{}
        ); // kBlockN x kBlockK x PIPE

        // partition
        TiledMMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);

        Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K, PIPE)
        Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K, PIPE)
        Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)
        Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
        clear(tCrC);

        // allocate "fragments"
        // note that in Ampere, the fragments are physically stored in registers
        // but in Hopper, the fragments are not, the tCrA and tCrB are actually the iterators not the data, but still in registers
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // iterator (1, MMA_M, MMA_K, PIPE)
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // iterator (1, MMA_N, MMA_K, PIPE)

         // MAINLOOP MMA
        #pragma unroll 1
        for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile) {
            // Wait for TMA to load this stage of the pipeline
            pipeline.consumer_wait(read_state);
            auto stage = read_state.index();
            warpgroup_arrive();
            // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
            gemm(tiled_mma, tCrC, tCrA(_,_,_,stage), tCrB(_,_,_,stage), tCrC);
            warpgroup_commit_batch();
        
            // Wait for all MMAs in a K_TILE to complete
            warpgroup_wait<0>();

            // Release the stage of the pipeline for TMA
            pipeline.consumer_release(read_state);
            ++read_state;
        }

        // Make sure all warpgroups have finished mma
        cutlass::arch::NamedBarrier::sync(Kernel_traits::kConsumerWGs * 32 * 4, 0);
        axpby(static_cast<Element>(1.0f), tCrC, static_cast<Element>(0.0f), tCgC);
    }
};

// kernel
template <
    typename Kernel_traits
>
__global__ void cute_hopper_gemm_v01(
    CUTE_GRID_CONSTANT typename CollectiveMainloop<Kernel_traits>::Params const mainloop_params
)
{   
    using CollectiveMainloop = CollectiveMainloop<Kernel_traits>;

    using MainloopPipeline = typename Kernel_traits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    // The Synchronization is orchestrated by the pipeline + pipeline state
    // each thread has its own pipeline state to control the synchronization
    // pipeline state = phase bit + stage + count
    // phase bit is initialized to 0 and flip between 0 and 1
    // stage increments by 1 each time but reset to 0 when it reaches kStages
    // count increments by 1 each time

    // shared memory for data + pipeline
    using SharedStorage = typename Kernel_traits::SharedStorage;
    extern __shared__ char smem_[];
    auto &shared_storage = *reinterpret_cast<SharedStorage*>(smem_);
   
    // Only one thread is elected to perfrom the prefetch
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    // prefetch TMA Descriptor
    if (warp_idx == 0 && lane_predicate)
    {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    }

    // pipeline initialization
    PipelineParams pipeline_params;

    // set the transaction size
    // Remember that the transaction size is passed to arrive_and_expect_tx of the barrier
    // However, the pipeline.producer_acquire_tx() will do that for us:
    // 1. empty_barrier.wait()
    // 2. full_barrier.arrive_and_expect_tx() (only Hopper does this 2.)
    pipeline_params.transaction_bytes = CollectiveMainloop::kTmaTransactionBytes;
    
    // set the role
    int warp_group_idx = cutlass::canonical_warp_group_idx();    
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;

    // set the thread leader (scope of warp_group)
    const int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup * Kernel_traits::kConsumerWGs;
    
    MainloopPipeline pipeline(
        shared_storage.pipeline,    // address of the pipeline
        pipeline_params,
        Shape<_1, _1, _1>{}         // todo
    );

    const int NUM_TILES_K = cutlass::ceil_div(
        shape<1>(mainloop_params.gmemLayoutA),
        Kernel_traits::kBlockK
    );

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    // This is similar to have the barrier visible to all threads in the Cluster
    cluster_sync();

    // Producer
    if (warp_group_idx == 0)
    {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_warpgroup = __shfl_sync(
            0xffffffff,                     // mask: all lanes in the warp
            (threadIdx.x / 32) % 4,         // value: warp_idx_in_warpgroup
            0                               // src lane
        );

        // only the first warp in the warp group will do the load
        // and only the elected thread inside the first warp will do the load
        if (warp_idx_in_warpgroup == 0)
        {
            PipelineState write_state = cutlass::make_producer_start_state<MainloopPipeline>();
            // phase bit = 1, stage = 0, count = 0
            // 1. the phase bit of write_state is to control the empty_barrier (sounds counterintuitive):
            // pipeline.producer_acquire(write_state) or empty_barrier.wait(): when all threads have arrived, the phase bit will be flipped to 1
            // 2. the stage of write_state is to control the full_barrier:
            // pipeline.producer_get_barrier(write_state): it gets the full_barrier_ptr_[stage]

            CollectiveMainloop::load(
                mainloop_params,
                pipeline,
                write_state,
                shared_storage,
                NUM_TILES_K
            );
        }
    }
    else    // Consumer
    {
        cutlass::arch::warpgroup_reg_alloc<240>();
        PipelineState read_state;

        CollectiveMainloop::mma(
            mainloop_params,
            pipeline,
            read_state,
            shared_storage,
            NUM_TILES_K
        );
    }
}

// launch
template<typename T>
void launch_cute_hopper_gemm_kernel_v01(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream
)
{   
    // Block shape and cta tiler
    constexpr int kWarps_ = 12;
    constexpr int kBlockM_ = 256;
    constexpr int kBlockN_ = 128;
    constexpr int kBlockK_ = 128;
    constexpr int kStages_ = 2;

    using Kernel_traits = KernelTraits<T, kWarps_,kBlockM_, kBlockN_, kBlockK_, kStages_>;

    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    using TiledMMA = typename Kernel_traits::TiledMMA;

    // setup TMA desc like v00 but using CollectiveMainloop
    int M = int(m); int N = int(n); int K = int(k);
    auto gmemLayoutA = make_layout(make_shape(M, K), make_stride(K, _1{}));
    auto gmemLayoutB = make_layout(make_shape(N, K), make_stride(K, _1{}));
    // auto gmemLayoutC = make_layout(make_shape(M, N), make_stride(_1{}, M));
    auto gmemLayoutC = make_layout(make_shape(M, N), make_stride(N, _1{}));
    using Collective_mainloop = CollectiveMainloop<Kernel_traits>;
    typename Collective_mainloop::Params mainloop_params = Collective_mainloop::to_underlying_arguments(
        {
            A, B, C,
            gmemLayoutA, gmemLayoutB, gmemLayoutC
        }
    );

    // Launch parameter setup
    constexpr int smem_size = Kernel_traits::smem_size;
    dim3 block{Kernel_traits::kThreads, 1U, 1U};
    dim3 cluster{1, 1, 1};
    dim3 grid{
        cute::size(ceil_div(n, kBlockN_)),
        cute::size(ceil_div(m, kBlockM_)),
        1U
    };
    cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size, stream};

    void const* kernel = reinterpret_cast<void const*>(&cute_hopper_gemm_v01 <Kernel_traits>);

    if (smem_size >= 48 * 1024) // 48KB
    {
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // kernel launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        kernel,
        mainloop_params
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Kernel launch failed with status: " << std::endl;
    }

}

// explicit instantiation                      
template void launch_cute_hopper_gemm_kernel_v01<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);                                    