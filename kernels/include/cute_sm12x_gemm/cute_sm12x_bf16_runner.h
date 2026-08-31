/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cute_sm12x_gemm {

class CuteSm12xBf16GemmRunner {
public:
    void gemm_bf16(
        void* D, void const* A, void const* B,
        int M, int N, int K, int num_sms, cudaStream_t stream);

    void batch_gemm_bf16(
        void* D, void const* A, void const* B,
        int L, int M, int N, int K,
        int num_sms, cudaStream_t stream);

    void group_gemm_bf16_masked(
        void* D, void const* A, void const* B,
        int32_t const* masked_m,
        int E, int max_m, int N, int K,
        int num_sms, cudaStream_t stream);

    void group_gemm_bf16_contiguous(
        void* D, void const* A, void const* B,
        int32_t const* m_indices,
        int E, int M, int N, int K, bool use_psum_layout,
        int num_sms, cudaStream_t stream);

    void moe_gemm_bf16(
        void* D, void const* A, void const* B,
        int32_t const* token_offset,
        int num_experts, int M, int N, int K,
        int num_sms, cudaStream_t stream);

    void fused_moe_bf16(
        void* D, void const* A, void const* B,
        int32_t const* token_offset,
        int E, int M, int N, int K,
        int num_sms, cudaStream_t stream);
};

} // namespace cute_sm12x_gemm
