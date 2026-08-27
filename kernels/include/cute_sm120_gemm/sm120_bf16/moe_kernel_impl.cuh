/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "cute_sm120_gemm/sm120_bf16/builder.cuh"
#include "cute_sm120_gemm/sm120_common/moe_scheduler.cuh"

namespace cute_sm120_gemm {
namespace sm120_bf16 {

using namespace cute;

template <typename KT>
struct SM120Bf16MoeGemmKernel {
  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers =
      KT::MMAConfig::kNumMathWarps + 1;
  static constexpr int MaxThreadsPerBlock =
      (KT::MMAConfig::kNumMathWarps + 2) * 32;
  static constexpr int MinBlocksPerMultiprocessor = 1;

  using Scheduler = sm120_common::SelectedMoeScheduler<
      KT::kSwapAB, KT::kTileM, KT::kTileN>;

  struct SharedStorage {
    typename KT::TensorStorage tensors;
    alignas(16) typename KT::BarrierStorage barriers;
    alignas(8) sm120_common::MoeSchedStorage<kNumSchedStages> sched;
  };

  static constexpr int kSmemSize = int(sizeof(SharedStorage));

  struct Arguments {
    typename KT::ElementA const* ptr_A;
    typename KT::ElementB const* ptr_B;
    typename KT::ElementD* ptr_D;
    int32_t const* token_offset;
    int M;
    int N;
    int K;
    int num_experts;
  };

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_load_a;
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::ElementD* ptr_D;
    int32_t const* grouped_layout;
    int M;
    int N;
    int K;
    int num_experts;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    using StrideA = typename KT::ABLoadConfig::StrideA;
    using StrideB = typename KT::ABLoadConfig::StrideB;
    auto ptr_physical_a = [&] {
      if constexpr (KT::kSwapAB) {
        return reinterpret_cast<typename KT::ElementA const*>(args.ptr_B);
      } else {
        return args.ptr_A;
      }
    }();
    auto ptr_physical_b = [&] {
      if constexpr (KT::kSwapAB) {
        return reinterpret_cast<typename KT::ElementB const*>(args.ptr_A);
      } else {
        return args.ptr_B;
      }
    }();
    StrideA dA{int64_t(args.K), Int<1>{},
               int64_t(KT::kSwapAB ? args.N : args.M) * args.K};
    StrideB dB{int64_t(args.K), Int<1>{},
               int64_t(KT::kSwapAB ? args.M : args.N) * args.K};
    auto [tma_load_a, tma_load_b] =
        sm120_common::utils::make_ab_tma_descriptors<KT>(
            ptr_physical_a, dA, ptr_physical_b, dB,
            args.M, args.N, args.K, args.num_experts);
    return {tma_load_a, tma_load_b, args.ptr_D, args.token_offset,
            args.M, args.N, args.K, args.num_experts};
  }

  CUTE_DEVICE
  static void load_ab(
      Params const& params, SharedStorage& shared,
      sm120_common::MoeWorkTile const& tile,
      int k_tile_count, int& write_stage,
      uint32_t& write_phase, uint32_t& store_phase) {
    auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
        tile.m_block, tile.n_block, tile.group);
    auto [tAgA, tBgB] = sm120_common::utils::tma_ab_partition<KT>(
        params.tma_load_a, params.tma_load_b,
        params.M, params.N, params.K, params.num_experts,
        blk_coord, tile.m_offset);
    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto sA_base = make_tensor(
        make_smem_ptr(shared.tensors.load.smem_A.begin()),
        typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_base = make_tensor(
        make_smem_ptr(shared.tensors.load.smem_B.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{});
    auto sA = as_position_independent_swizzle_tensor(sA_base);
    auto sB = as_position_independent_swizzle_tensor(sB_base);
    auto tAsA = block_tma_a.partition_D(sA);
    auto tBsB = block_tma_b.partition_D(sB);

    if constexpr (KT::kUnionSmem) {
      shared.barriers.store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      shared.barriers.ab_empty_mbar[write_stage].wait(write_phase);
      auto& full = shared.barriers.ab_full_mbar[write_stage];
      auto copy_a = params.tma_load_a.with(
          *recast_ptr<typename KT::ProducerBarrierType>(&full));
      auto copy_b = params.tma_load_b.with(
          *recast_ptr<typename KT::ProducerBarrierType>(&full));
      full.arrive_and_expect_tx(KT::ABLoadConfig::TmaABTransactionBytes);
      cute::copy(copy_a, tAgA(_, _, _, k_tile), tAsA(_, _, _, write_stage));
      cute::copy(copy_b, tBgB(_, _, _, k_tile), tBsB(_, _, _, write_stage));
      ++write_stage;
      if (write_stage == KT::AB_Stages) {
        write_stage = 0;
        write_phase ^= 1;
      }
    }
  }

  CUTE_DEVICE
  static void mma(
      Params const& params, SharedStorage& shared,
      sm120_common::MoeWorkTile const& tile,
      int k_tile_count, int thread_idx,
      int& read_stage, uint32_t& read_phase) {
    typename KT::MMAConfig::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto accum = partition_fragment_C(
        tiled_mma, make_shape(Int<KT::kTileM>{}, Int<KT::kTileN>{}));
    clear(accum);

    auto sA_base = make_tensor(
        make_smem_ptr(shared.tensors.load.smem_A.begin()),
        typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_base = make_tensor(
        make_smem_ptr(shared.tensors.load.smem_B.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{});
    auto sA = as_position_independent_swizzle_tensor(sA_base);
    auto sB = as_position_independent_swizzle_tensor(sB_base);
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
    auto s2r_copy_A = make_tiled_copy_A(
        typename KT::ABLoadConfig::SmemCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(
        typename KT::ABLoadConfig::SmemCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);
    constexpr int KBlockCount = decltype(size<2>(tCrA))::value;

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      shared.barriers.ab_full_mbar[read_stage].wait(read_phase);
      cute::for_each(cute::make_int_sequence<KBlockCount>{}, [&](auto k_block) {
        cute::copy(s2r_copy_A,
                   tXsA(_, _, k_block, read_stage), tXrA(_, _, k_block));
        cute::copy(s2r_copy_B,
                   tXsB(_, _, k_block, read_stage), tXrB(_, _, k_block));
        cute::gemm(tiled_mma,
                   tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      });
      shared.barriers.ab_empty_mbar[read_stage].arrive();
      ++read_stage;
      if (read_stage == KT::AB_Stages) {
        read_stage = 0;
        read_phase ^= 1;
      }
    }

    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    if constexpr (KT::kSwapAB) {
      typename KT::EmptyBarrier* no_store_barrier = nullptr;
      auto blk_coord = sm120_common::utils::make_blk_coord<true>(
          tile.m_block, tile.n_block, tile.group);
      sm120_common::utils::epi_pred_stg<KT>(
          params, accum, thread_idx,
          tile.m_offset, tile.m_boundary,
          get<0>(blk_coord), get<1>(blk_coord),
          no_store_barrier);
    } else {
      sm120_common::utils::epi_pred_r2g<KT>(
          params, shared, accum, thread_idx,
          tile.m_offset, tile.m_boundary,
          tile.m_block, tile.n_block, tile.group,
          &shared.barriers.store_empty_mbar[0]);
    }
  }

  CUTE_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    bool init_thread = warp_idx == 0 && lane_predicate;

    if (init_thread) {
      cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    }
    __syncthreads();

    if (init_thread) {
      CUTE_UNROLL
      for (int stage = 0; stage < KT::AB_Stages; ++stage) {
        shared.barriers.ab_full_mbar[stage].init(1);
        shared.barriers.ab_empty_mbar[stage].init(
            KT::MMAConfig::kNumMathThreads);
      }
      if constexpr (KT::kUnionSmem) {
        shared.barriers.store_empty_mbar[0].init(
            KT::MMAConfig::kNumMathThreads);
      }
      shared.sched.init_mbars(kNumSchedConsumers);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int k_tile_count = (params.K + KT::kTileK - 1) / KT::kTileK;
    constexpr int sched_warp = KT::MMAConfig::kNumMathWarps;
    constexpr int ab_warp = sched_warp + 1;

    if (warp_idx == sched_warp) {
      Scheduler scheduler(
          params.M, params.N, params.num_experts, params.grouped_layout);
      sm120_common::MoeSchedProducer<kNumSchedStages> pipeline{
          shared.sched, lane_predicate};
      int32_t m_block;
      int32_t n_block;
      while (scheduler.get_next_block(m_block, n_block)) {
        pipeline.publish(sm120_common::MoeWorkTile{
            m_block, n_block, scheduler.get_expert_idx(m_block),
            scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
      }
      pipeline.publish_sentinel();
    } else if (warp_idx == ab_warp) {
      int write_stage = 0;
      uint32_t write_phase = 1;
      uint32_t store_phase = 1;
      sm120_common::MoeSchedConsumer<kNumSchedStages> pipeline{
          shared.sched, lane_predicate};
      sm120_common::MoeWorkTile tile;
      while (pipeline.get_next_tile(tile)) {
        if (lane_predicate) {
          load_ab(params, shared, tile, k_tile_count,
                  write_stage, write_phase, store_phase);
        }
      }
    } else {
      int read_stage = 0;
      uint32_t read_phase = 0;
      sm120_common::MoeSchedConsumer<kNumSchedStages> pipeline{
          shared.sched, lane_predicate};
      sm120_common::MoeWorkTile tile;
      while (pipeline.get_next_tile(tile)) {
        mma(params, shared, tile, k_tile_count,
            int(threadIdx.x), read_stage, read_phase);
      }
    }
  }
};

}  // namespace sm120_bf16
}  // namespace cute_sm120_gemm
