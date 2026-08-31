/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cute_sm12x_gemm::sm120_bf16 {

enum class Bf16Tactic : uint8_t {
  M32N128K32,
  SwapABM8N128K32,
};

inline Bf16Tactic select_batched_tactic(
    int L, int M, int N, int K, int num_sms) {
  if (L <= 0 || M <= 0 || N <= 0 || K <= 0 || num_sms <= 0 ||
      N % 8 != 0 || K % 8 != 0) {
    return Bf16Tactic::M32N128K32;
  }
  int64_t const m_tiles = (int64_t(M) + 7) / 8;
  int64_t const n_tiles = (int64_t(N) + 127) / 128;
  if (M <= 8) {
    // With int32 inputs and m_tiles == 1, this product is bounded by
    // INT_MAX * ceil(INT_MAX / 128) and fits int64_t.
    int64_t const tiles = int64_t(L) * m_tiles * n_tiles;
    int64_t const last_wave_tiles = (tiles - 1) % num_sms + 1;
    if (last_wave_tiles <= 4 && M <= 1 && K > 2048) {
      return Bf16Tactic::M32N128K32;
    }
    return Bf16Tactic::SwapABM8N128K32;
  }
  bool const tiles_over_140 =
      int64_t(L) > 140 / m_tiles ||
      int64_t(L) * m_tiles > 140 / n_tiles;
  if (tiles_over_140 || K % 32 != 0 || (K <= 1536 && M > 24)) {
    return Bf16Tactic::M32N128K32;
  }
  return Bf16Tactic::SwapABM8N128K32;
}

}  // namespace cute_sm12x_gemm::sm120_bf16
