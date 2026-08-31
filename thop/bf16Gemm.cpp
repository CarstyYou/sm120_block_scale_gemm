/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cute_sm12x_gemm/cute_sm12x_bf16_runner.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <limits>

namespace torch_ext
{

torch::Tensor gemm_bf16(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda(),
                "mat1 and mat2 must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device(),
                "mat1 and mat2 must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous(),
                "mat1 and mat2 must be contiguous");
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2,
                "mat1 and mat2 must be rank 2");
    auto const M = mat1.size(0);
    auto const K = mat1.size(1);
    auto const N = mat2.size(0);
    TORCH_CHECK(mat2.size(1) == K, "mat1 and mat2 K must match");
    TORCH_CHECK(N > 0 && N % 8 == 0,
                "N must be positive and a multiple of 8, got ", N);
    TORCH_CHECK(K > 0 && K % 8 == 0,
                "K must be positive and a multiple of 8, got ", K);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat1 and mat2 data pointers must be 16-byte aligned");

    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty(
        {M, N}, torch::TensorOptions()
                    .dtype(at::ScalarType::BFloat16)
                    .device(mat1.device()));
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.gemm_bf16(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        num_sms, stream);
    return out;
}

void check_batched_bf16_inputs(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda(),
                "mat1 and mat2 must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device(),
                "mat1 and mat2 must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous(),
                "mat1 and mat2 must be contiguous");
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3,
                "mat1 and mat2 must be rank 3");
    TORCH_CHECK(mat1.size(0) > 0 && mat1.size(0) == mat2.size(0),
                "mat1 and mat2 batch dimensions must match and be positive");
    TORCH_CHECK(mat1.size(2) == mat2.size(2),
                "mat1 and mat2 K must match");
    TORCH_CHECK(mat2.size(1) > 0 && mat2.size(1) % 8 == 0,
                "N must be positive and a multiple of 8");
    TORCH_CHECK(mat1.size(2) > 0 && mat1.size(2) % 8 == 0,
                "K must be positive and a multiple of 8");
    auto const int_max = std::numeric_limits<int>::max();
    TORCH_CHECK(mat1.size(0) <= int_max && mat1.size(1) <= int_max &&
                    mat2.size(1) <= int_max && mat1.size(2) <= int_max,
                "L, M, N and K must fit int32");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat1 and mat2 data pointers must be 16-byte aligned");
}

torch::Tensor launch_batch_gemm_bf16(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor& out) {
    auto const L = mat1.size(0);
    auto const M = mat1.size(1);
    auto const K = mat1.size(2);
    auto const N = mat2.size(1);
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "out data pointer must be 16-byte aligned");
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.batch_gemm_bf16(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int>(L), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), num_sms, stream);
    return out;
}

torch::Tensor batch_gemm_bf16_out(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor& out) {
    check_batched_bf16_inputs(mat1, mat2);
    auto const L = mat1.size(0);
    auto const M = mat1.size(1);
    auto const N = mat2.size(1);
    TORCH_CHECK(out.is_cuda() && out.device() == mat1.device(),
                "out must be CUDA and on the same device as mat1");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
                "out must be BFloat16");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({L, M, N}),
                "out must have shape [L,M,N]");
    return launch_batch_gemm_bf16(mat1, mat2, out);
}

torch::Tensor batch_gemm_bf16(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2) {
    check_batched_bf16_inputs(mat1, mat2);
    auto out = torch::empty(
        {mat1.size(0), mat1.size(1), mat2.size(1)},
        torch::TensorOptions().dtype(at::ScalarType::BFloat16)
                              .device(mat1.device()));
    return launch_batch_gemm_bf16(mat1, mat2, out);
}

torch::Tensor group_gemm_bf16_masked(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor const& masked_m) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && masked_m.is_cuda(),
                "mat1, mat2 and masked_m must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device() &&
                    mat1.device() == masked_m.device(),
                "mat1, mat2 and masked_m must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(masked_m.scalar_type() == at::ScalarType::Int,
                "masked_m must be Int32");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous() &&
                    masked_m.is_contiguous(),
                "mat1, mat2 and masked_m must be contiguous");
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3 && masked_m.dim() == 1,
                "expected mat1[E,maxM,K], mat2[E,N,K], masked_m[E]");
    auto const E = mat1.size(0);
    auto const max_m = mat1.size(1);
    auto const K = mat1.size(2);
    auto const N = mat2.size(1);
    TORCH_CHECK(E > 0 && mat2.size(0) == E && masked_m.size(0) == E,
                "expert dimensions must match and be positive");
    TORCH_CHECK(mat2.size(2) == K, "mat1 and mat2 K must match");
    TORCH_CHECK(N > 0 && N % 8 == 0,
                "N must be positive and a multiple of 8");
    TORCH_CHECK(K > 0 && K % 8 == 0,
                "K must be positive and a multiple of 8");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat1 and mat2 data pointers must be 16-byte aligned");
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty(
        {E, max_m, N}, torch::TensorOptions()
                           .dtype(at::ScalarType::BFloat16)
                           .device(mat1.device()));
    if (max_m == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.group_gemm_bf16_masked(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(masked_m.data_ptr()),
        static_cast<int>(E), static_cast<int>(max_m),
        static_cast<int>(N), static_cast<int>(K), num_sms, stream);
    return out;
}

torch::Tensor group_gemm_bf16_contiguous(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor const& m_indices,
    bool use_psum_layout) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && m_indices.is_cuda(),
                "mat1, mat2 and m_indices must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device() &&
                    mat1.device() == m_indices.device(),
                "mat1, mat2 and m_indices must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(m_indices.scalar_type() == at::ScalarType::Int,
                "m_indices must be Int32");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous() &&
                    m_indices.is_contiguous(),
                "mat1, mat2 and m_indices must be contiguous");
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 3 && m_indices.dim() == 1,
                "expected mat1[M,K], mat2[E,N,K], m_indices[metadata]");
    auto const M = mat1.size(0);
    auto const K = mat1.size(1);
    auto const E = mat2.size(0);
    auto const N = mat2.size(1);
    TORCH_CHECK(E > 0, "E must be positive");
    TORCH_CHECK(mat2.size(2) == K, "mat1 and mat2 K must match");
    TORCH_CHECK(N > 0 && N % 8 == 0,
                "N must be positive and a multiple of 8");
    TORCH_CHECK(K > 0 && K % 8 == 0,
                "K must be positive and a multiple of 8");
    TORCH_CHECK(
        m_indices.size(0) == (use_psum_layout ? E : M),
        "m_indices size must be E for psum layout and M otherwise");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat1 and mat2 data pointers must be 16-byte aligned");
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty(
        {M, N}, torch::TensorOptions()
                    .dtype(at::ScalarType::BFloat16)
                    .device(mat1.device()));
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.group_gemm_bf16_contiguous(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(m_indices.data_ptr()),
        static_cast<int>(E), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), use_psum_layout,
        num_sms, stream);
    return out;
}

torch::Tensor moe_gemm_bf16(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor const& token_offset) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && token_offset.is_cuda(),
                "mat1, mat2 and token_offset must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device() &&
                    mat1.device() == token_offset.device(),
                "mat1, mat2 and token_offset must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Int,
                "token_offset must be Int32");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous() &&
                    token_offset.is_contiguous(),
                "mat1, mat2 and token_offset must be contiguous");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be [M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [E, N, K]");
    TORCH_CHECK(token_offset.dim() == 1, "token_offset must be 1D");

    auto const M = mat1.size(0);
    auto const K = mat1.size(1);
    auto const E = mat2.size(0);
    auto const N = mat2.size(1);
    TORCH_CHECK(E > 0, "E must be positive");
    TORCH_CHECK(N > 0 && N % 8 == 0,
                "N must be positive and a multiple of 8, got ", N);
    TORCH_CHECK(K > 0 && K % 8 == 0,
                "K must be positive and a multiple of 8, got ", K);
    TORCH_CHECK(mat2.size(2) == K, "mat1 and mat2 K must match");
    TORCH_CHECK(token_offset.size(0) == E + 1,
                "token_offset size must be E+1");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0,
                "mat1 data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat2 data pointer must be 16-byte aligned");

    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty(
        {M, N}, torch::TensorOptions()
                    .dtype(at::ScalarType::BFloat16)
                    .device(mat1.device()));
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.moe_gemm_bf16(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(token_offset.data_ptr()),
        static_cast<int>(E), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), num_sms, stream);
    return out;
}

torch::Tensor fused_moe_bf16(
    torch::Tensor const& mat1,
    torch::Tensor const& mat2,
    torch::Tensor const& token_offset) {
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && token_offset.is_cuda(),
                "mat1, mat2 and token_offset must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device() &&
                    mat1.device() == token_offset.device(),
                "mat1, mat2 and token_offset must be on the same device");
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16 &&
                    mat2.scalar_type() == at::ScalarType::BFloat16,
                "mat1 and mat2 must be BFloat16");
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Int,
                "token_offset must be Int32");
    TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous() &&
                    token_offset.is_contiguous(),
                "mat1, mat2 and token_offset must be contiguous");
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 3 && token_offset.dim() == 1,
                "expected mat1[M,K], mat2[E,2N,K], token_offset[E+1]");
    auto const M = mat1.size(0);
    auto const K = mat1.size(1);
    auto const E = mat2.size(0);
    auto const physical_n = mat2.size(1);
    TORCH_CHECK(E > 0, "E must be positive");
    TORCH_CHECK(physical_n > 0 && physical_n % 2 == 0,
                "mat2 dimension 1 must be positive and even");
    auto const N = physical_n / 2;
    TORCH_CHECK(N % 8 == 0, "logical N must be a multiple of 8");
    TORCH_CHECK(K > 0 && K % 8 == 0,
                "K must be positive and a multiple of 8");
    TORCH_CHECK(mat2.size(2) == K, "mat1 and mat2 K must match");
    TORCH_CHECK(token_offset.size(0) == E + 1,
                "token_offset size must be E+1");
    constexpr auto int_max = std::numeric_limits<int>::max();
    TORCH_CHECK(M <= int_max && K <= int_max && E <= int_max &&
                    physical_n <= int_max,
                "fused MoE dimensions must fit int32");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat1 and mat2 data pointers must be 16-byte aligned");
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty(
        {M, N}, torch::TensorOptions()
                    .dtype(at::ScalarType::BFloat16)
                    .device(mat1.device()));
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int num_sms = at::cuda::getDeviceProperties(
        mat1.get_device())->multiProcessorCount;
    cute_sm12x_gemm::CuteSm12xBf16GemmRunner runner;
    runner.fused_moe_bf16(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(token_offset.data_ptr()),
        static_cast<int>(E), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), num_sms, stream);
    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(custom_ops, m) {
    m.def("gemm_bf16(Tensor mat1, Tensor mat2) -> Tensor");
    m.def("batch_gemm_bf16(Tensor mat1, Tensor mat2) -> Tensor");
    m.def("batch_gemm_bf16_out(Tensor mat1, Tensor mat2, "
          "Tensor(a!) out) -> Tensor(a!)");
    m.def("group_gemm_bf16_masked(Tensor mat1, Tensor mat2, "
          "Tensor masked_m) -> Tensor");
    m.def("group_gemm_bf16_contiguous(Tensor mat1, Tensor mat2, "
          "Tensor m_indices, bool use_psum_layout=False) -> Tensor");
    m.def("moe_gemm_bf16(Tensor mat1, Tensor mat2, "
          "Tensor token_offset) -> Tensor");
    m.def("fused_moe_bf16(Tensor mat1, Tensor mat2, "
          "Tensor token_offset) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CUDA, m) {
    m.impl("gemm_bf16", &torch_ext::gemm_bf16);
    m.impl("batch_gemm_bf16", &torch_ext::batch_gemm_bf16);
    m.impl("batch_gemm_bf16_out", &torch_ext::batch_gemm_bf16_out);
    m.impl("group_gemm_bf16_masked",
           &torch_ext::group_gemm_bf16_masked);
    m.impl("group_gemm_bf16_contiguous",
           &torch_ext::group_gemm_bf16_contiguous);
    m.impl("moe_gemm_bf16",
           &torch_ext::moe_gemm_bf16);
    m.impl("fused_moe_bf16",
           &torch_ext::fused_moe_bf16);
}
