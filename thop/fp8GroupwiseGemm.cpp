/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cute_sm120_gemm/cute_sm120_fp8_runner.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace cute_sm120_gemm;

namespace torch_ext
{

namespace
{

using Fp8BlockScaleGemmRunnerPtr =
    std::unique_ptr<CuteSm120Fp8GemmRunnerInterface>;

Fp8BlockScaleGemmRunnerPtr get_gemm_runner(
    at::ScalarType a_type, at::ScalarType b_type)
{
    if (a_type == at::ScalarType::Float8_e4m3fn &&
        b_type == at::ScalarType::Float8_e4m3fn)
    {
        return std::make_unique<CuteSm120Fp8GemmRunner<
            cute::float_e4m3_t, cute::bfloat16_t, float, float>>();
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input types: ", a_type, " and ",
                    b_type);
    }
}

} // namespace

int64_t ceil_div(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

void check_scale_granularity_mnk(
    int64_t scale_granularity_m, int64_t scale_granularity_n,
    int64_t scale_granularity_k)
{
    TORCH_CHECK(
        scale_granularity_m == 1 && scale_granularity_n == 128 &&
            scale_granularity_k == 128,
        "unsupported scale_granularity_mnk (", scale_granularity_m, ", ",
        scale_granularity_n, ", ", scale_granularity_k, ")",
        "; expected (1, 128, 128)");
}

void check_matrix(torch::Tensor const& mat, char const* name)
{
    TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
                name, " dtype must be Float8_e4m3fn");
    TORCH_CHECK(mat.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(mat.is_contiguous(), name, " must be contiguous");
}

void check_scale(torch::Tensor const& scale, char const* name)
{
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
                name, " dtype must be Float32");
    TORCH_CHECK(scale.is_cuda(), name, " must be CUDA");
}

void check_shape(torch::Tensor const& tensor, std::vector<int64_t> const& expected,
                 char const* name);

int64_t compute_padded_offset(int64_t offset, int64_t problem_idx)
{
    constexpr int64_t alignment = 4;
    return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

void check_sfa_layout(
    torch::Tensor const& sfa, int64_t groups, int64_t m, int64_t k,
    int64_t gran_m, int64_t gran_k, bool per_group)
{
    int64_t scale_m = ceil_div(m, gran_m);
    int64_t scale_k = ceil_div(k, gran_k);
    int64_t element_size = sfa.element_size();
    TORCH_CHECK(16 % element_size == 0,
                "mat1Scale element size must divide 16 bytes");
    int64_t alignment = 16 / element_size;
    int64_t aligned_m = ceil_div(scale_m, alignment) * alignment;
    int64_t k_dim = per_group ? 1 : 0;
    int64_t m_dim = per_group ? 2 : 1;
    TORCH_CHECK(
        sfa.stride(k_dim) == aligned_m && sfa.stride(m_dim) == 1,
        "mat1Scale must have TMA-aligned [Kb,M] stride");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(sfa.data_ptr()) % 16 == 0,
        "mat1Scale data pointer must be 16-byte aligned");
    if (per_group) {
        TORCH_CHECK(
            sfa.stride(0) == scale_k * aligned_m,
            "mat1Scale batch stride must cover padded M");
    }
    int64_t required_elements = sfa.storage_offset() +
        (per_group ? groups : 1) * scale_k * aligned_m;
    TORCH_CHECK(
        required_elements * element_size <=
            static_cast<int64_t>(sfa.storage().nbytes()),
        "mat1Scale backing storage is too small for padded M stride");
}

void check_zero_padding_sfa_layout(
    torch::Tensor const& sfa, int64_t num_experts, int64_t m, int64_t k,
    int64_t gran_m, int64_t gran_k)
{
    int64_t scale_m = ceil_div(m, gran_m);
    int64_t scale_k = ceil_div(k, gran_k);
    int64_t padded_m = compute_padded_offset(scale_m, num_experts);
    check_shape(sfa, {scale_k, padded_m}, "mat1Scale");
    TORCH_CHECK(sfa.is_contiguous(),
                "mat1Scale must be contiguous [Kb,MpE]");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(sfa.data_ptr()) % 16 == 0,
        "mat1Scale data pointer must be 16-byte aligned");
}

void check_sfb_layout(torch::Tensor const& sfb)
{
    TORCH_CHECK(sfb.is_contiguous(), "mat2Scale must be contiguous");
}

void check_layout(torch::Tensor const& layout, char const* name)
{
    TORCH_CHECK(layout.scalar_type() == at::ScalarType::Int,
                name, " dtype must be Int32");
    TORCH_CHECK(layout.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(layout.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const& ref, torch::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(ref.device() == tensor.device(), name, " must be on ", ref.device(),
                ", got ", tensor.device());
}

void check_shape(torch::Tensor const& tensor, std::vector<int64_t> const& expected,
                 char const* name)
{
    TORCH_CHECK(tensor.sizes().vec() == expected, name, " must have shape ",
                c10::IntArrayRef(expected),
                ", got ", tensor.sizes());
}

void check_mnk(int64_t n, int64_t k)
{
    TORCH_CHECK(k > 0, "K must be positive, got ", k);
    TORCH_CHECK(n > 0, "N must be positive, got ", n);
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, got ", k);
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, got ", n);
}

void check_scale_shapes(
    torch::Tensor const& sfa, torch::Tensor const& sfb,
    int64_t num_groups, int64_t m, int64_t n, int64_t k,
    int64_t scale_granularity_m, int64_t scale_granularity_n,
    int64_t scale_granularity_k,
    bool sfa_per_group, bool sfb_per_group)
{
    int64_t scale_m = ceil_div(m, scale_granularity_m);
    int64_t scale_n = ceil_div(n, scale_granularity_n);
    int64_t scale_k = ceil_div(k, scale_granularity_k);
    std::vector<int64_t> sfa_shape = sfa_per_group
        ? std::vector<int64_t>{num_groups, scale_k, scale_m}
        : std::vector<int64_t>{scale_k, scale_m};
    std::vector<int64_t> sfb_shape = sfb_per_group
        ? std::vector<int64_t>{num_groups, scale_k, scale_n}
        : std::vector<int64_t>{scale_k, scale_n};
    check_shape(sfa, sfa_shape, "mat1Scale");
    check_shape(sfb, sfb_shape, "mat2Scale");
}

void check_inputs(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1_scale, torch::Tensor const& mat2_scale)
{
    check_matrix(mat1, "mat1");
    check_matrix(mat2, "mat2");
    check_scale(mat1_scale, "mat1Scale");
    check_scale(mat2_scale, "mat2Scale");
    check_same_device(mat1, mat2, "mat2");
    check_same_device(mat1, mat1_scale, "mat1Scale");
    check_same_device(mat1, mat2_scale, "mat2Scale");
}

torch::Tensor gemm_fp8_nt_groupwise(torch::Tensor const& mat1, torch::Tensor const& mat2,
                                    torch::Tensor const& mat1Scale,
                                    torch::Tensor const& mat2Scale,
                                    int64_t scale_granularity_m,
                                    int64_t scale_granularity_n,
                                    int64_t scale_granularity_k)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1],
                "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0],
                "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x",
                mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    check_mnk(n, k);
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    check_scale_shapes(
        mat1Scale, mat2Scale, 1, m, n, k,
        scale_granularity_m, scale_granularity_n, scale_granularity_k, false, false);
    check_sfa_layout(
        mat1Scale, 1, m, k, scale_granularity_m, scale_granularity_k,
        false);
    check_sfb_layout(mat2Scale);

    c10::cuda::CUDAGuard device_guard(mat1.device());
    at::Tensor out = torch::empty({m, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));

    auto gemm_runner = get_gemm_runner(
        mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->gemm_fp8_nt_groupwise(
        out.data_ptr(),
        reinterpret_cast<void const*>(mat1.data_ptr()),
        reinterpret_cast<void const*>(mat2.data_ptr()),
        m, n, k,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        stream,
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k));

    return out;
}

torch::Tensor batch_gemm_fp8_nt_groupwise_out(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor& out, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be [L, M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [L, N, K]");
    TORCH_CHECK(mat1.size(0) == mat2.size(0), "mat1 and mat2 batch size must match");
    TORCH_CHECK(mat1.size(2) == mat2.size(2), "mat1 and mat2 K must match");
    auto const l = mat1.size(0);
    auto const m = mat1.size(1);
    auto const n = mat2.size(1);
    auto const k = mat1.size(2);
    TORCH_CHECK(l > 0, "batch size must be positive");
    check_mnk(n, k);
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    check_scale_shapes(
        mat1Scale, mat2Scale, l, m, n, k,
        scale_granularity_m, scale_granularity_n, scale_granularity_k, true, true);
    check_sfa_layout(
        mat1Scale, l, m, k, scale_granularity_m, scale_granularity_k,
        true);
    check_sfb_layout(mat2Scale);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out dtype must be BFloat16");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    check_same_device(mat1, out, "out");
    check_shape(out, {l, m, n}, "out");
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto gemm_runner = get_gemm_runner(
        mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->batch_gemm_fp8_nt_groupwise(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(), l, m, n, k,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        stream,
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k));
    return out;
}

torch::Tensor batch_gemm_fp8_nt_groupwise(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    int64_t scale_granularity_m, int64_t scale_granularity_n,
    int64_t scale_granularity_k)
{
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be [L, M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [L, N, K]");
    TORCH_CHECK(mat1.size(0) == mat2.size(0), "mat1 and mat2 batch size must match");
    auto out = torch::empty(
        {mat1.size(0), mat1.size(1), mat2.size(1)},
        torch::TensorOptions().dtype(at::ScalarType::BFloat16).device(mat1.device()));
    return batch_gemm_fp8_nt_groupwise_out(
        mat1, mat2, mat1Scale, mat2Scale, out,
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
}

torch::Tensor group_gemm_fp8_nt_groupwise_masked(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor const& masked_m, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);
    check_layout(masked_m, "masked_m");
    check_same_device(mat1, masked_m, "masked_m");
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be [L, max_M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [L, N, K]");
    TORCH_CHECK(masked_m.dim() == 1, "masked_m must be 1D");
    TORCH_CHECK(mat1.size(0) == mat2.size(0), "mat1 and mat2 group count must match");
    TORCH_CHECK(masked_m.size(0) == mat1.size(0), "masked_m size must match group count");
    TORCH_CHECK(mat1.size(2) == mat2.size(2), "mat1 and mat2 K must match");
    auto const num_groups = mat1.size(0);
    auto const max_m = mat1.size(1);
    auto const n = mat2.size(1);
    auto const k = mat1.size(2);
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    check_mnk(n, k);
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    check_scale_shapes(mat1Scale, mat2Scale, num_groups, max_m, n, k,
                       scale_granularity_m, scale_granularity_n,
                       scale_granularity_k, true, true);
    check_sfa_layout(
        mat1Scale, num_groups, max_m, k,
        scale_granularity_m, scale_granularity_k, true);
    check_sfb_layout(mat2Scale);
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty({num_groups, max_m, n}, torch::TensorOptions()
                                                     .dtype(at::ScalarType::BFloat16)
                                                     .device(mat1.device()));
    auto gemm_runner = get_gemm_runner(
        mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->group_gemm_fp8_nt_groupwise_masked(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(masked_m.data_ptr()), num_groups,
        max_m, n, k, stream,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k));
    return out;
}

torch::Tensor moe_gemm_fp8_nt_groupwise(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor const& token_offset, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);
    check_layout(token_offset, "token_offset");
    check_same_device(mat1, token_offset, "token_offset");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be [M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [E, N, K]");
    TORCH_CHECK(token_offset.dim() == 1, "token_offset must be 1D");
    TORCH_CHECK(token_offset.size(0) == mat2.size(0) + 1,
                "token_offset size must be num_experts + 1");
    TORCH_CHECK(mat1.size(1) == mat2.size(2), "mat1 and mat2 K must match");
    auto const m = mat1.size(0);
    auto const num_experts = mat2.size(0);
    auto const n = mat2.size(1);
    auto const k = mat1.size(1);
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    check_mnk(n, k);
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    int64_t scale_n = ceil_div(n, scale_granularity_n);
    int64_t scale_k = ceil_div(k, scale_granularity_k);
    check_zero_padding_sfa_layout(
        mat1Scale, num_experts, m, k,
        scale_granularity_m, scale_granularity_k);
    check_shape(
        mat2Scale, {num_experts, scale_k, scale_n}, "mat2Scale");
    check_sfb_layout(mat2Scale);
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty({m, n}, torch::TensorOptions()
                                        .dtype(at::ScalarType::BFloat16)
                                        .device(mat1.device()));
    auto gemm_runner = get_gemm_runner(
        mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->moe_gemm_fp8_nt_groupwise(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(token_offset.data_ptr()), num_experts,
        m, n, k, stream,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k));
    return out;
}

torch::Tensor fused_moe_fp8_nt_groupwise(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor const& token_offset, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);
    check_layout(token_offset, "token_offset");
    check_same_device(mat1, token_offset, "token_offset");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0,
                "mat1 data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat2 data pointer must be 16-byte aligned");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be [M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [E, 2I, K]");
    TORCH_CHECK(token_offset.dim() == 1, "token_offset must be 1D");

    auto const m = mat1.size(0);
    auto const k = mat1.size(1);
    auto const num_experts = mat2.size(0);
    auto const fused_n = mat2.size(1);
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(fused_n > 0 && fused_n % 2 == 0,
                "mat2 dimension 1 must be positive and even, got ", fused_n);
    auto const n = fused_n / 2;
    TORCH_CHECK(mat2.size(2) == k, "mat1 and mat2 K must match");
    TORCH_CHECK(k > 0 && k % 16 == 0,
                "K must be positive and a multiple of 16, got ", k);
    TORCH_CHECK(n % 64 == 0,
                "FusedMoe FP8 I must be a multiple of 64, got ", n);
    TORCH_CHECK(token_offset.size(0) == num_experts + 1,
                "token_offset size must be num_experts + 1");
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    check_zero_padding_sfa_layout(
        mat1Scale, num_experts, m, k,
        scale_granularity_m, scale_granularity_k);
    auto const scale_k = ceil_div(k, scale_granularity_k);
    auto const scale_n = ceil_div(fused_n, scale_granularity_n);
    check_shape(
        mat2Scale, {num_experts, scale_k, scale_n}, "mat2Scale");
    check_sfb_layout(mat2Scale);

    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty({m, n}, torch::TensorOptions()
                                       .dtype(at::ScalarType::BFloat16)
                                       .device(mat1.device()));
    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->moe_gemm_fp8_nt_groupwise(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(token_offset.data_ptr()),
        num_experts, m, fused_n, k, stream,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k), /*is_gated=*/true);
    return out;
}

torch::Tensor group_gemm_fp8_nt_groupwise_contiguous(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor const& m_indices, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k,
    bool use_psum_layout)
{
    check_inputs(mat1, mat2, mat1Scale, mat2Scale);
    check_layout(m_indices, "m_indices");
    check_same_device(mat1, m_indices, "m_indices");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be [M, K]");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be [E, N, K]");
    TORCH_CHECK(m_indices.dim() == 1, "m_indices must be 1D");
    TORCH_CHECK(mat1.size(1) == mat2.size(2), "mat1 and mat2 K must match");
    auto const m = mat1.size(0);
    auto const num_groups = mat2.size(0);
    auto const n = mat2.size(1);
    auto const k = mat1.size(1);
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    if (use_psum_layout) {
        TORCH_CHECK(m_indices.size(0) == num_groups,
                    "use_psum_layout=true requires m_indices size to match num_groups");
    } else {
        TORCH_CHECK(m_indices.size(0) == m,
                    "use_psum_layout=false requires m_indices size to match M");
    }
    check_mnk(n, k);
    check_scale_granularity_mnk(
        scale_granularity_m, scale_granularity_n, scale_granularity_k);
    check_scale_shapes(mat1Scale, mat2Scale, num_groups, m, n, k,
                       scale_granularity_m, scale_granularity_n,
                       scale_granularity_k, false, true);
    check_sfa_layout(
        mat1Scale, 1, m, k, scale_granularity_m, scale_granularity_k,
        false);
    check_sfb_layout(mat2Scale);
    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty({m, n}, torch::TensorOptions()
                                        .dtype(at::ScalarType::BFloat16)
                                        .device(mat1.device()));
    auto gemm_runner = get_gemm_runner(
        mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->group_gemm_fp8_nt_groupwise_contiguous(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(m_indices.data_ptr()), num_groups,
        m, n, k, stream,
        static_cast<float const*>(mat1Scale.data_ptr()),
        static_cast<float const*>(mat2Scale.data_ptr()),
        static_cast<int>(scale_granularity_m),
        static_cast<int>(scale_granularity_n),
        static_cast<int>(scale_granularity_k), use_psum_layout);
    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(custom_ops, m)
{
    m.def("gemm_fp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, int scale_granularity_m, int scale_granularity_n, "
          "int scale_granularity_k) -> Tensor");
    m.def("batch_gemm_fp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, int scale_granularity_m, int scale_granularity_n, "
          "int scale_granularity_k) -> Tensor");
    m.def("batch_gemm_fp8_nt_groupwise_out(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, Tensor(a!) out, int scale_granularity_m, "
          "int scale_granularity_n, int scale_granularity_k) -> Tensor(a!)");
    m.def("moe_gemm_fp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, Tensor token_offset, int scale_granularity_m, "
          "int scale_granularity_n, int scale_granularity_k) -> Tensor");
    m.def("fused_moe_fp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, Tensor token_offset, int scale_granularity_m, "
          "int scale_granularity_n, int scale_granularity_k) -> Tensor");
    m.def("group_gemm_fp8_nt_groupwise_masked(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, Tensor masked_m, int scale_granularity_m, "
          "int scale_granularity_n, int scale_granularity_k) -> Tensor");
    m.def("group_gemm_fp8_nt_groupwise_contiguous(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, Tensor m_indices, int scale_granularity_m, "
          "int scale_granularity_n, int scale_granularity_k, "
          "bool use_psum_layout=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CUDA, m)
{
    m.impl("gemm_fp8_nt_groupwise", &torch_ext::gemm_fp8_nt_groupwise);
    m.impl("batch_gemm_fp8_nt_groupwise", &torch_ext::batch_gemm_fp8_nt_groupwise);
    m.impl("batch_gemm_fp8_nt_groupwise_out", &torch_ext::batch_gemm_fp8_nt_groupwise_out);
    m.impl("moe_gemm_fp8_nt_groupwise", &torch_ext::moe_gemm_fp8_nt_groupwise);
    m.impl("fused_moe_fp8_nt_groupwise", &torch_ext::fused_moe_fp8_nt_groupwise);
    m.impl("group_gemm_fp8_nt_groupwise_masked", &torch_ext::group_gemm_fp8_nt_groupwise_masked);
    m.impl("group_gemm_fp8_nt_groupwise_contiguous",
           &torch_ext::group_gemm_fp8_nt_groupwise_contiguous);
}
