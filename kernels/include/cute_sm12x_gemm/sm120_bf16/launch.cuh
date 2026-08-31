/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cuda_runtime.h>

#include <cutlass/device_kernel.h>

#include "cute_sm12x_gemm/sm120_bf16/kernel_impl.cuh"
#include "cute_sm12x_gemm/sm120_bf16/fused_moe_kernel_impl.cuh"
#include "cute_sm12x_gemm/sm120_bf16/fused_moe_swapab_kernel_impl.cuh"
#include "cute_sm12x_gemm/sm120_bf16/moe_kernel_impl.cuh"

namespace cute_sm12x_gemm {
namespace sm120_bf16 {

template <typename Kernel>
__forceinline__ void launch_kernel(
    typename Kernel::Params const& params,
    int num_sms,
    cudaStream_t stream = 0)
{
    auto kernel_ptr = &cutlass::device_kernel<Kernel>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Kernel::kSmemSize));

    cudaLaunchConfig_t launch_config{};
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    launch_config.gridDim = dim3(num_sms, 1, 1);
    launch_config.blockDim = dim3(Kernel::MaxThreadsPerBlock, 1, 1);
    launch_config.dynamicSmemBytes = Kernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    CUTE_CHECK_ERROR(cudaLaunchKernelEx(&launch_config, kernel_ptr, params));
    CUTE_CHECK_ERROR(cudaGetLastError());
}

template <typename KT>
void launch_moe_gemm(
    typename KT::ElementA const* ptr_A,
    typename KT::ElementB const* ptr_B,
    typename KT::ElementD* ptr_D,
    int M, int N, int K, int num_experts,
    int32_t const* grouped_layout,
    int num_sms,
    cudaStream_t stream = 0)
{
    using Kernel = std::conditional_t<
        KT::kGemmType ==
            sm120_common::GemmType::MGroupedContiguousWithZeroPadding,
        SM120Bf16MoeGemmKernel<KT>,
        SM120Bf16GemmKernel<KT>>;
    typename Kernel::Arguments args{
        ptr_A, ptr_B, ptr_D, grouped_layout,
        M, N, K, num_experts};
    launch_kernel<Kernel>(
        Kernel::to_underlying_arguments(args), num_sms, stream);
}

template <typename KT>
void launch_gemm(
    typename KT::ElementA const* ptr_A,
    typename KT::ElementB const* ptr_B,
    typename KT::ElementD* ptr_D,
    int M, int N, int K,
    int num_sms,
    cudaStream_t stream = 0)
{
    using Kernel = SM120Bf16GemmKernel<KT>;
    typename Kernel::Arguments args{
        ptr_A, ptr_B, ptr_D, nullptr, M, N, K, 1};
    launch_kernel<Kernel>(
        Kernel::to_underlying_arguments(args), num_sms, stream);
}

template <typename KT>
void launch_bmm(
    typename KT::ElementA const* ptr_A,
    typename KT::ElementB const* ptr_B,
    typename KT::ElementD* ptr_D,
    int M, int N, int K, int L,
    int num_sms,
    cudaStream_t stream = 0)
{
    using Kernel = SM120Bf16GemmKernel<KT>;
    typename Kernel::Arguments args{
        ptr_A, ptr_B, ptr_D, nullptr, M, N, K, L};
    launch_kernel<Kernel>(
        Kernel::to_underlying_arguments(args), num_sms, stream);
}

template <typename KT>
void launch_masked_gemm(
    typename KT::ElementA const* ptr_A,
    typename KT::ElementB const* ptr_B,
    typename KT::ElementD* ptr_D,
    int max_m, int N, int K, int num_groups,
    int32_t const* masked_m,
    int num_sms,
    cudaStream_t stream = 0)
{
    using Kernel = SM120Bf16GemmKernel<KT>;
    typename Kernel::Arguments args{
        ptr_A, ptr_B, ptr_D, masked_m,
        max_m, N, K, num_groups};
    launch_kernel<Kernel>(
        Kernel::to_underlying_arguments(args), num_sms, stream);
}

template <typename KT>
void launch_fused_moe(
    typename KT::ElementA const* ptr_A,
    typename KT::ElementB const* ptr_B,
    typename KT::ElementD* ptr_D,
    int M, int N, int K, int num_experts,
    int32_t const* grouped_layout,
    int num_sms,
    cudaStream_t stream = 0)
{
    using Kernel = std::conditional_t<
        KT::kSwapAB,
        SM120Bf16FusedMoeSwapABGemmKernel,
        SM120Bf16FusedMoeGemmKernel<KT>>;
    typename Kernel::Arguments args{
        ptr_A, ptr_B, ptr_D, grouped_layout,
        M, N, K, num_experts};
    launch_kernel<Kernel>(
        Kernel::to_underlying_arguments(args), num_sms, stream);
}

}  // namespace sm120_bf16
}  // namespace cute_sm12x_gemm
