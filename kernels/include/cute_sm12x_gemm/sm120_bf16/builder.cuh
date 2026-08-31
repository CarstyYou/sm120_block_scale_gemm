/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "cute_sm12x_gemm/sm120_common/ab_tma_load.cuh"
#include "cute_sm12x_gemm/sm120_common/epilogue.cuh"
#include "cute_sm12x_gemm/sm120_common/scheduler.cuh"

namespace cute_sm12x_gemm {
namespace sm120_bf16 {

using namespace cute;

template <bool SwapAB, typename LoadStorage, typename R2GConfig>
struct Bf16TensorStorageSelector;

template <typename LoadStorage, typename R2GConfig>
struct Bf16TensorStorageSelector<false, LoadStorage, R2GConfig> {
  union Type {
    LoadStorage load;
    typename R2GConfig::SharedStorageR2G store;
  };
};

template <typename LoadStorage, typename R2GConfig>
struct Bf16TensorStorageSelector<true, LoadStorage, R2GConfig> {
  struct Type {
    LoadStorage load;
  };
};

template <sm120_common::GemmType GemmType_, bool SwapAB_,
          int TileM_, int TileN_>
struct SM120Bf16BuilderConfig {
  using ElementA = cute::bfloat16_t;
  using ElementB = cute::bfloat16_t;
  using ElementAccum = float;
  using ElementD = cute::bfloat16_t;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = 32;
  static constexpr int AB_Stages = 4;
  static constexpr sm120_common::GemmType kGemmType = GemmType_;
  static constexpr bool kFlat = sm120_common::is_flat_gemm(kGemmType);
  static constexpr bool kSwapAB = SwapAB_;
  static constexpr bool kPerBatchAB =
      kGemmType == sm120_common::GemmType::Batched ||
      kGemmType == sm120_common::GemmType::MGroupedMasked;
  static constexpr bool kUnionSmem = !kSwapAB;
  static constexpr bool kDefaultConfig =
      !kSwapAB && kTileM == 32 && kTileN == 128;
  static constexpr bool kLargeM64Config =
      !kSwapAB && kTileM == 64 && kTileN == 128 &&
      (kGemmType == sm120_common::GemmType::Normal ||
       kGemmType == sm120_common::GemmType::Batched ||
       kGemmType == sm120_common::GemmType::MGroupedMasked ||
       kGemmType ==
           sm120_common::GemmType::MGroupedContiguousWithZeroPadding);
  static constexpr bool kLargeM128Config =
      !kSwapAB && kTileM == 128 && kTileN == 128;
  static constexpr bool kSwapABConfig =
      kSwapAB && kTileM == 128 && kTileN == 8;
  static constexpr bool kValidPhysicalConfig =
      kDefaultConfig || kLargeM64Config || kLargeM128Config || kSwapABConfig;
  static_assert(kValidPhysicalConfig, "unsupported BF16 physical config");

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using MMAAtom = cute::MMA_Atom<
      cute::SM80_16x8x16_F32BF16BF16F32_TN>;

  struct MMAConfig {
    using WarpLayout = std::conditional_t<
        kSwapAB,
        Layout<Shape<_8, _1, _1>, Stride<_1, _8, _0>>,
        Layout<Shape<_2, _4, _1>, Stride<_1, _2, _0>>>;
    using TiledMma = cute::TiledMMA<
        MMAAtom,
        WarpLayout,
        Tile<Int<kTileM>, Int<kTileN>, _16>>;
    static constexpr int kNumMathWarps = 8;
    static constexpr int kNumMathThreads = kNumMathWarps * 32;
  };

  using ABLoadConfig = sm120_common::Sm120BlockScaledABLoadConfig<
      kTileM, kTileN, kTileK, AB_Stages, ElementA, ElementB>;
  using R2GStoreConfig = std::conditional_t<
      kSwapAB,
      void,
      sm120_common::Sm120BlockScaledR2GStoreConfig<kTileM, kTileN, ElementD>>;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<
        ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024) cute::ArrayEngine<
        ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
  };

  using TensorStorage = typename Bf16TensorStorageSelector<
      kSwapAB, SharedStorageLoad, R2GStoreConfig>::Type;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;

  struct BarrierStorageUnion {
    FullBarrier ab_full_mbar[AB_Stages];
    EmptyBarrier ab_empty_mbar[AB_Stages];
    EmptyBarrier store_empty_mbar[1];
  };
  struct BarrierStorageDirect {
    FullBarrier ab_full_mbar[AB_Stages];
    EmptyBarrier ab_empty_mbar[AB_Stages];
  };
  using BarrierStorage = std::conditional_t<
      kSwapAB, BarrierStorageDirect, BarrierStorageUnion>;
};

}  // namespace sm120_bf16
}  // namespace cute_sm12x_gemm
