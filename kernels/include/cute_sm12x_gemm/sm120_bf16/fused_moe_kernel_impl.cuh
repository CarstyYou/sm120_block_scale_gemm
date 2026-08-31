/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "cute_sm12x_gemm/sm120_bf16/fused_builder.cuh"
#include "cute_sm12x_gemm/sm120_common/moe_scheduler.cuh"

namespace cute_sm12x_gemm {
namespace sm120_bf16 {

using namespace cute;

template <typename KT_>
struct SM120Bf16FusedMoeGemmKernel {
  using KT = KT_;
  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers =
      KT::MMAConfig::kNumMathWarps + 1;
  static constexpr int MaxThreadsPerBlock =
      (KT::MMAConfig::kNumMathWarps + 2) * 32;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using Scheduler = sm120_common::MoeScheduler<KT::kTileM, KT::kTileN>;

  struct SharedStorage {
    typename KT::TensorStorageUnion tensors;
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
    int E;
  };

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_a;
    typename KT::ABLoadConfig::TMA_B tma_b;
    typename KT::ElementD* ptr_D;
    int32_t const* grouped_layout;
    int M;
    int N;
    int K;
    int num_experts;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    typename KT::ABLoadConfig::StrideA dA{
        int64_t(args.K), Int<1>{}, int64_t(args.M) * args.K};
    typename KT::ABLoadConfig::StrideB dB{
        int64_t(args.K), Int<1>{}, int64_t(2 * args.N) * args.K};
    auto [tma_a, tma_b] =
        sm120_common::utils::make_ab_tma_descriptors<KT>(
            args.ptr_A, dA, args.ptr_B, dB,
            args.M, 2 * args.N, args.K, args.E);
    return {tma_a, tma_b, args.ptr_D, args.token_offset,
            args.M, args.N, args.K, args.E};
  }

  CUTE_DEVICE
  static void load_ab(
      Params const& p, SharedStorage& s,
      sm120_common::MoeWorkTile const& tile,
      int k_tiles, int& stage, uint32_t& phase, uint32_t& store_phase) {
    auto mA_full = p.tma_a.get_tma_tensor(make_shape(p.M, p.K, 1));
    auto mA = domain_offset(make_coord(tile.m_offset, 0, 0), mA_full);
    auto mB = p.tma_b.get_tma_tensor(make_shape(2 * p.N, p.K, p.num_experts));
    auto mGate = domain_offset(make_coord(p.N, 0, 0), mB);
    auto gA = local_tile(mA, typename KT::TileShape{}, make_coord(_, _, _),
                        Step<_1, Underscore, _1>{});
    auto gUp = local_tile(mB, typename KT::TileShape{}, make_coord(_, _, _),
                         Step<Underscore, _1, _1>{});
    auto gGate = local_tile(mGate, typename KT::TileShape{}, make_coord(_, _, _),
                           Step<Underscore, _1, _1>{});
    auto ba = p.tma_a.get_slice(0);
    auto bb = p.tma_b.get_slice(0);
    auto srcA = ba.partition_S(gA(_, _, tile.m_block, _, 0));
    auto srcUp = bb.partition_S(gUp(_, _, tile.n_block, _, tile.group));
    auto srcGate = bb.partition_S(gGate(_, _, tile.n_block, _, tile.group));
    auto sA = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_A.begin()),
        typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sUp = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_B_up.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sGate = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_B_gate.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{}));
    auto dstA = ba.partition_D(sA);
    auto dstUp = bb.partition_D(sUp);
    auto dstGate = bb.partition_D(sGate);
    s.barriers.store_empty_mbar[0].wait(store_phase);
    store_phase ^= 1;
    for (int kt = 0; kt < k_tiles; ++kt) {
      s.barriers.ab_empty_mbar[stage].wait(phase);
      auto ca = p.tma_a.with(
          *recast_ptr<typename KT::ProducerBarrierType>(
              &s.barriers.ab_full_mbar[stage]));
      auto cb = p.tma_b.with(
          *recast_ptr<typename KT::ProducerBarrierType>(
              &s.barriers.ab_full_mbar[stage]));
      s.barriers.ab_full_mbar[stage].arrive_and_expect_tx(
          KT::TmaTransactionBytesFused);
      cute::copy(ca, srcA(_, _, _, kt), dstA(_, _, _, stage));
      cute::copy(cb, srcUp(_, _, _, kt), dstUp(_, _, _, stage));
      cute::copy(cb, srcGate(_, _, _, kt), dstGate(_, _, _, stage));
      if (++stage == KT::AB_Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  }

  CUTE_DEVICE
  static void mma(
      Params const& p, SharedStorage& s,
      sm120_common::MoeWorkTile const& tile,
      int k_tiles, int tid, int& stage, uint32_t& phase) {
    typename KT::MMAConfig::TiledMma tiled_mma;
    auto thr = tiled_mma.get_thread_slice(tid);
    auto accum_up = partition_fragment_C(
        tiled_mma, make_shape(Int<KT::kTileM>{}, Int<KT::kTileN>{}));
    auto accum_gate = partition_fragment_C(
        tiled_mma, make_shape(Int<KT::kTileM>{}, Int<KT::kTileN>{}));
    clear(accum_up);
    clear(accum_gate);
    auto sA = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_A.begin()),
        typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sUp = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_B_up.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sGate = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(s.tensors.load.smem_B_gate.begin()),
        typename KT::ABLoadConfig::SmemLayoutB{}));
    auto rA = thr.partition_fragment_A(sA(_, _, 0));
    auto rB = thr.partition_fragment_B(sUp(_, _, 0));
    auto copyA = make_tiled_copy_A(
        typename KT::ABLoadConfig::SmemCopyAtomA{}, tiled_mma);
    auto copyB = make_tiled_copy_B(
        typename KT::ABLoadConfig::SmemCopyAtomB{}, tiled_mma);
    auto ca = copyA.get_thread_slice(tid);
    auto cb = copyB.get_thread_slice(tid);
    auto srcA = ca.partition_S(sA);
    auto srcUp = cb.partition_S(sUp);
    auto srcGate = cb.partition_S(sGate);
    auto dstA = ca.retile_D(rA);
    auto dstB = cb.retile_D(rB);
    constexpr int KBlocks = decltype(size<2>(rA))::value;
    static_assert(KBlocks == 2);
    for (int kt = 0; kt < k_tiles; ++kt) {
      s.barriers.ab_full_mbar[stage].wait(phase);
      cute::for_each(cute::make_int_sequence<KBlocks>{}, [&](auto kb) {
        cute::copy(copyA, srcA(_, _, kb, stage), dstA(_, _, kb));
        cute::copy(copyB, srcUp(_, _, kb, stage), dstB(_, _, kb));
        cute::gemm(tiled_mma, rA(_, _, kb), rB(_, _, kb), accum_up);
        cute::copy(copyB, srcGate(_, _, kb, stage), dstB(_, _, kb));
        cute::gemm(tiled_mma, rA(_, _, kb), rB(_, _, kb), accum_gate);
      });
      s.barriers.ab_empty_mbar[stage].arrive();
      if (++stage == KT::AB_Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
    CUTE_UNROLL
    for (int i = 0; i < size(accum_gate); ++i) {
      accum_gate(i) = accum_gate(i) / (1.0f + ::expf(-accum_gate(i)));
      accum_up(i) *= accum_gate(i);
    }
    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    sm120_common::utils::epi_pred_r2g<KT>(
        p, s, accum_up, tid,
        tile.m_offset, tile.m_boundary,
        tile.m_block, tile.n_block, tile.group,
        &s.barriers.store_empty_mbar[0]);
  }

  CUTE_DEVICE
  void operator()(Params const& p, char* smem_buf) {
    auto& s = *reinterpret_cast<SharedStorage*>(smem_buf);
    int warp = cutlass::canonical_warp_idx_sync();
    int elected = cute::elect_one_sync();
    bool init = warp == 0 && elected;
    if (init) {
      cute::prefetch_tma_descriptor(p.tma_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(p.tma_b.get_tma_descriptor());
    }
    __syncthreads();
    if (init) {
      for (int i = 0; i < KT::AB_Stages; ++i) {
        s.barriers.ab_full_mbar[i].init(1);
        s.barriers.ab_empty_mbar[i].init(KT::MMAConfig::kNumMathThreads);
      }
      s.barriers.store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      s.sched.init_mbars(kNumSchedConsumers);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    int k_tiles = (p.K + KT::kTileK - 1) / KT::kTileK;
    constexpr int SchedWarp = KT::MMAConfig::kNumMathWarps;
    constexpr int ABWarp = SchedWarp + 1;
    if (warp == SchedWarp) {
      Scheduler scheduler(p.M, p.N, p.num_experts, p.grouped_layout);
      sm120_common::MoeSchedProducer<kNumSchedStages> pipe{s.sched, elected};
      int32_t mb, nb;
      while (scheduler.get_next_block(mb, nb)) {
        pipe.publish(sm120_common::MoeWorkTile{
            mb, nb, scheduler.get_expert_idx(mb),
            scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
      }
      pipe.publish_sentinel();
    } else if (warp == ABWarp) {
      int stage = 0;
      uint32_t phase = 1, store_phase = 1;
      sm120_common::MoeSchedConsumer<kNumSchedStages> pipe{s.sched, elected};
      sm120_common::MoeWorkTile tile;
      while (pipe.get_next_tile(tile)) {
        if (elected) {
          load_ab(p, s, tile, k_tiles, stage, phase, store_phase);
        }
      }
    } else {
      int stage = 0;
      uint32_t phase = 0;
      sm120_common::MoeSchedConsumer<kNumSchedStages> pipe{s.sched, elected};
      sm120_common::MoeWorkTile tile;
      while (pipe.get_next_tile(tile)) {
        mma(p, s, tile, k_tiles, int(threadIdx.x), stage, phase);
      }
    }
  }
};

}  // namespace sm120_bf16
}  // namespace cute_sm12x_gemm
