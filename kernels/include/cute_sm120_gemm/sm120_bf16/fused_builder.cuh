/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cute_sm120_gemm/sm120_bf16/builder.cuh"

namespace cute_sm120_gemm {
namespace sm120_bf16 {

template <int TileM_>
struct SM120Bf16FusedMoeBuilderConfig
    : SM120Bf16BuilderConfig<
          sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
          false, TileM_, 128> {
  using Base = SM120Bf16BuilderConfig<
      sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
      false, TileM_, 128>;
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ABLoadConfig = typename Base::ABLoadConfig;
  using R2GStoreConfig = typename Base::R2GStoreConfig;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<
        ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024) cute::ArrayEngine<
        ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_up;
    alignas(1024) cute::ArrayEngine<
        ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_gate;
  };

  static constexpr uint32_t TmaTransactionBytesFused =
      ABLoadConfig::TmaTransactionBytesA +
      2 * ABLoadConfig::TmaTransactionBytesB;
  static_assert(TmaTransactionBytesFused == 64 * TileM_ + 16384);

  union TensorStorageUnion {
    SharedStorageLoad load;
    typename R2GStoreConfig::SharedStorageR2G store;
  };
  static_assert(sizeof(SharedStorageLoad) == 256 * TileM_ + 65536);
  static_assert(sizeof(TensorStorageUnion) == sizeof(SharedStorageLoad));
};

struct SM120Bf16FusedMoeBuilder
    : SM120Bf16FusedMoeBuilderConfig<32> {};

struct SM120Bf16FusedMoeSwapABBuilder
    : SM120Bf16BuilderConfig<
          sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
          true, 128, 8> {
  using Base = SM120Bf16BuilderConfig<
      sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
      true, 128, 8>;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<
        ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_up;
    alignas(1024) cute::ArrayEngine<
        ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_gate;
    alignas(1024) cute::ArrayEngine<
        ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
  };

  using TensorStorage = typename Bf16TensorStorageSelector<
      true, SharedStorageLoad, typename Base::R2GStoreConfig>::Type;
  static constexpr uint32_t TmaTransactionBytesFused =
      2 * ABLoadConfig::TmaTransactionBytesA +
      ABLoadConfig::TmaTransactionBytesB;
  static_assert(TmaTransactionBytesFused == 16896);
  static_assert(sizeof(SharedStorageLoad) == 67584);
  static_assert(sizeof(TensorStorage) == 67584);
};

}  // namespace sm120_bf16
}  // namespace cute_sm120_gemm
