/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cute_sm12x_gemm/cute_sm12x_bf16_runner.h"
#include "cute_sm12x_gemm/sm120_bf16/launch.cuh"
#include "cute_sm12x_gemm/sm120_bf16/tile_selection.h"

namespace cute_sm12x_gemm {

void CuteSm12xBf16GemmRunner::gemm_bf16(
    void* D, void const* A, void const* B,
    int M, int N, int K, int num_sms, cudaStream_t stream)
{
    using KT = sm120_bf16::SM120Bf16BuilderConfig<
        sm120_common::GemmType::Normal, false, 32, 128>;
    sm120_bf16::launch_gemm<KT>(
        reinterpret_cast<KT::ElementA const*>(A),
        reinterpret_cast<KT::ElementB const*>(B),
        reinterpret_cast<KT::ElementD*>(D),
        M, N, K, num_sms, stream);
}

void CuteSm12xBf16GemmRunner::batch_gemm_bf16(
    void* D, void const* A, void const* B,
    int L, int M, int N, int K,
    int num_sms, cudaStream_t stream)
{
    constexpr auto kGT = sm120_common::GemmType::Batched;
    using KT_M32 = sm120_bf16::SM120Bf16BuilderConfig<
        kGT, false, 32, 128>;
    using KT_SWAPAB = sm120_bf16::SM120Bf16BuilderConfig<
        kGT, true, 128, 8>;
    auto const* ptr_A = reinterpret_cast<KT_M32::ElementA const*>(A);
    auto const* ptr_B = reinterpret_cast<KT_M32::ElementB const*>(B);
    auto* ptr_D = reinterpret_cast<KT_M32::ElementD*>(D);
    auto tactic = sm120_bf16::select_batched_tactic(L, M, N, K, num_sms);
    if (tactic == sm120_bf16::Bf16Tactic::SwapABM8N128K32) {
        sm120_bf16::launch_bmm<KT_SWAPAB>(
            ptr_A, ptr_B, ptr_D, M, N, K, L, num_sms, stream);
    } else {
        sm120_bf16::launch_bmm<KT_M32>(
            ptr_A, ptr_B, ptr_D, M, N, K, L, num_sms, stream);
    }
}

void CuteSm12xBf16GemmRunner::group_gemm_bf16_masked(
    void* D, void const* A, void const* B,
    int32_t const* masked_m,
    int E, int max_m, int N, int K,
    int num_sms, cudaStream_t stream)
{
    using KT = sm120_bf16::SM120Bf16BuilderConfig<
        sm120_common::GemmType::MGroupedMasked, false, 32, 128>;
    sm120_bf16::launch_masked_gemm<KT>(
        reinterpret_cast<KT::ElementA const*>(A),
        reinterpret_cast<KT::ElementB const*>(B),
        reinterpret_cast<KT::ElementD*>(D),
        max_m, N, K, E, masked_m, num_sms, stream);
}

void CuteSm12xBf16GemmRunner::group_gemm_bf16_contiguous(
    void* D, void const* A, void const* B,
    int32_t const* m_indices,
    int E, int M, int N, int K, bool use_psum_layout,
    int num_sms, cudaStream_t stream)
{
    if (use_psum_layout) {
        using KT = sm120_bf16::SM120Bf16BuilderConfig<
            sm120_common::GemmType::MGroupedContiguousWithPsumLayout,
            false, 32, 128>;
        sm120_bf16::launch_moe_gemm<KT>(
            reinterpret_cast<KT::ElementA const*>(A),
            reinterpret_cast<KT::ElementB const*>(B),
            reinterpret_cast<KT::ElementD*>(D),
            M, N, K, E, m_indices, num_sms, stream);
    } else {
        using KT = sm120_bf16::SM120Bf16BuilderConfig<
            sm120_common::GemmType::MGroupedContiguous,
            false, 32, 128>;
        sm120_bf16::launch_moe_gemm<KT>(
            reinterpret_cast<KT::ElementA const*>(A),
            reinterpret_cast<KT::ElementB const*>(B),
            reinterpret_cast<KT::ElementD*>(D),
            M, N, K, E, m_indices, num_sms, stream);
    }
}

void CuteSm12xBf16GemmRunner::moe_gemm_bf16(
    void* D, void const* A, void const* B,
    int32_t const* token_offset,
    int num_experts, int M, int N, int K,
    int num_sms, cudaStream_t stream)
{
    using KT = sm120_bf16::SM120Bf16BuilderConfig<
        sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
        false, 32, 128>;
    sm120_bf16::launch_moe_gemm<KT>(
        reinterpret_cast<KT::ElementA const*>(A),
        reinterpret_cast<KT::ElementB const*>(B),
        reinterpret_cast<KT::ElementD*>(D),
        M, N, K, num_experts, token_offset,
        num_sms, stream);
}

void CuteSm12xBf16GemmRunner::fused_moe_bf16(
    void* D, void const* A, void const* B,
    int32_t const* token_offset,
    int E, int M, int N, int K,
    int num_sms, cudaStream_t stream)
{
    using KT = sm120_bf16::SM120Bf16FusedMoeBuilder;
    sm120_bf16::launch_fused_moe<KT>(
        reinterpret_cast<KT::ElementA const*>(A),
        reinterpret_cast<KT::ElementB const*>(B),
        reinterpret_cast<KT::ElementD*>(D),
        M, N, K, E, token_offset, num_sms, stream);
}

namespace sm120_bf16 {

using NormalM64 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::Normal, false, 64, 128>;
using NormalM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::Normal, false, 128, 128>;
using BatchedM64 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::Batched, false, 64, 128>;
using BatchedM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::Batched, false, 128, 128>;
using MaskedM64 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedMasked, false, 64, 128>;
using MaskedM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedMasked, false, 128, 128>;
using ContiguousM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedContiguous, false, 128, 128>;
using PsumM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedContiguousWithPsumLayout,
    false, 128, 128>;
using MoeM64 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
    false, 64, 128>;
using MoeM128 = SM120Bf16BuilderConfig<
    sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
    false, 128, 128>;
using FusedM64 = SM120Bf16FusedMoeBuilderConfig<64>;
using FusedM128 = SM120Bf16FusedMoeBuilderConfig<128>;

template void launch_gemm<NormalM64>(
    NormalM64::ElementA const*, NormalM64::ElementB const*,
    NormalM64::ElementD*, int, int, int, int, cudaStream_t);
template void launch_gemm<NormalM128>(
    NormalM128::ElementA const*, NormalM128::ElementB const*,
    NormalM128::ElementD*, int, int, int, int, cudaStream_t);
template void launch_bmm<BatchedM64>(
    BatchedM64::ElementA const*, BatchedM64::ElementB const*,
    BatchedM64::ElementD*, int, int, int, int, int, cudaStream_t);
template void launch_bmm<BatchedM128>(
    BatchedM128::ElementA const*, BatchedM128::ElementB const*,
    BatchedM128::ElementD*, int, int, int, int, int, cudaStream_t);
template void launch_masked_gemm<MaskedM64>(
    MaskedM64::ElementA const*, MaskedM64::ElementB const*,
    MaskedM64::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_masked_gemm<MaskedM128>(
    MaskedM128::ElementA const*, MaskedM128::ElementB const*,
    MaskedM128::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_moe_gemm<ContiguousM128>(
    ContiguousM128::ElementA const*, ContiguousM128::ElementB const*,
    ContiguousM128::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_moe_gemm<PsumM128>(
    PsumM128::ElementA const*, PsumM128::ElementB const*,
    PsumM128::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_moe_gemm<MoeM64>(
    MoeM64::ElementA const*, MoeM64::ElementB const*,
    MoeM64::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_moe_gemm<MoeM128>(
    MoeM128::ElementA const*, MoeM128::ElementB const*,
    MoeM128::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_fused_moe<FusedM64>(
    FusedM64::ElementA const*, FusedM64::ElementB const*,
    FusedM64::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);
template void launch_fused_moe<FusedM128>(
    FusedM128::ElementA const*, FusedM128::ElementB const*,
    FusedM128::ElementD*, int, int, int, int, int32_t const*,
    int, cudaStream_t);

} // namespace sm120_bf16

} // namespace cute_sm12x_gemm
