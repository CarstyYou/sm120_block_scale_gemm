/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <tuple>
#include <torch/library.h>
#include <torch/types.h>

#define CHECK_TYPE(x, st)                                                      \
    TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                     \
    CHECK_TH_CUDA(x);                                                          \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                 \
    CHECK_CPU(x);                                                              \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_TYPE(x, st)

using namespace mxfp8_cute_sm120;

namespace torch_ext
{
constexpr auto FP8_BLOCK_SCALING_SF_DTYPE = torch::ScalarType::Float;

template <typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline T get_val(torch::Tensor& t, int idx)
{
    assert(idx < t.numel());
    return reinterpret_cast<T*>(t.data_ptr())[idx];
}

void check_input_dtypes(torch::Tensor const& mat, torch::Tensor const& matScale)
{
    TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                mat.scalar_type() == at::ScalarType::BFloat16,
                "Matrix dtype must be FP8 (the matrix will be dequantized on "
                "the fly).");

}

using Fp8BlockScaleGemmRunnerPtr =
    std::unique_ptr<CuteSm120Mxfp8GemmRunnerInterface>;

Fp8BlockScaleGemmRunnerPtr get_gemm_runner(at::ScalarType a_type,
                                           at::ScalarType b_type)
{
    if (a_type == at::ScalarType::Float8_e4m3fn &&
             b_type == at::ScalarType::Float8_e4m3fn)
    {
        return std::make_unique<CuteSm120Mxfp8GemmRunner<
            cute::float_e4m3_t, cute::bfloat16_t, float, cute::float_ue8m0_t>>();
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input types: ", a_type, " and ",
                    b_type);
    }
}

extern torch::Tensor gemm_mxfp8_nt_groupwise(torch::Tensor const& mat1,
                                            torch::Tensor const& mat2,
                                            torch::Tensor const& mat1Scale,
                                            torch::Tensor const& mat2Scale,
                                            int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1],
                "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0],
                "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x",
                mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = torch::empty({m, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    auto D = out.data_ptr();
    auto A = reinterpret_cast<void const*>(mat1.data_ptr());
    auto B = reinterpret_cast<void const*>(mat2.data_ptr());
    auto A_scales = static_cast<int32_t const*>(mat1Scale.data_ptr());
    auto B_scales = static_cast<int32_t const*>(mat2Scale.data_ptr());

    gemm_runner->gemm_mxfp8_nt_groupwise(D, A, B, m, n, k, A_scales, B_scales, stream, static_cast<int>(granK));

    return out;
}

// All inputs are k-major
torch::Tensor batch_gemm_mxfp8_nt_groupwise_out(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor& out, int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 3, "mat1 must be a batched matrix");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batched matrix");
    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 batch dim must be the same but got", mat1.sizes()[0],
        ", and ", mat2.sizes()[0]);
    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "mat1 and mat2 k dim must be the same but got", mat1.sizes()[2],
        ", and ", mat2.sizes()[2]);

    // mat1 could be strided due to padding

    auto const b = mat1.sizes()[0];
    auto const m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];
    auto const k = mat1.sizes()[2];
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    CHECK_TH_CUDA(out);
    CHECK_TYPE(out, at::ScalarType::BFloat16);
    auto const& out_shape = out.sizes();
    TORCH_CHECK(out_shape[0] == b && out_shape[1] == m && out_shape[2] == n, "out shape must be (", b, ", ", m, ", ", n,
        "), but got (", out_shape[0], ", ", out_shape[1], ", ", out_shape[2], ").");

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());


    void* out_ptr = reinterpret_cast<void*>(out.data_ptr());
    void* mat1_ptr = reinterpret_cast<void*>(mat1.data_ptr());
    void* mat2_ptr = reinterpret_cast<void*>(mat2.data_ptr());
    int32_t* mat1ScalePtr = static_cast<int32_t*>(mat1Scale.data_ptr());
    int32_t* mat2ScalePtr = static_cast<int32_t*>(mat2Scale.data_ptr());

    TORCH_CHECK(out.strides()[2] == 1, "The last stride of out must be 1, not ", out.strides()[2]);
    TORCH_CHECK(mat1.strides()[2] == 1, "The last stride of mat1 must be 1, not ", mat1.strides()[2]);
    TORCH_CHECK(mat2.strides()[2] == 1, "The last stride of mat2 must be 1, not ", mat2.strides()[2]);

    auto const strideD = out.strides()[0]; // m * n
    auto const ldd = out.strides()[1];     // n

    auto const strideA = mat1.strides()[0];
    auto const lda = mat1.strides()[1];

    auto const strideB = mat2.strides()[0];
    auto const ldb = mat2.strides()[1];

    gemm_runner->batch_gemm_mxfp8_nt_groupwise(
        mat1_ptr, lda, strideA,
        mat2_ptr, ldb, strideB,
        out_ptr, ldd, strideD,
        mat1ScalePtr, mat2ScalePtr,
        b, m, n, k, stream, static_cast<int>(granK));

    return out;
}

// All inputs are k-major
torch::Tensor batch_gemm_mxfp8_nt_groupwise(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, std::optional<c10::ScalarType> out_dtype,
    int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    auto const b = mat1.sizes()[0];
    auto const m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];

    at::Tensor out = torch::empty({b, m, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));
    return batch_gemm_mxfp8_nt_groupwise_out(mat1, mat2, mat1Scale, mat2Scale, out, granK);
}

torch::Tensor group_gemm_mxfp8_nt_groupwise_masked(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& masked_m, int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 3, "mat1 must be a batched matrix");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batched matrix");
    TORCH_CHECK(masked_m.sizes()[0] == mat1.sizes()[0], "masked_m and mat1 batch dim must be the same but got", masked_m.sizes()[0],
        ", and ", mat1.sizes()[0]);
    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 batch dim must be the same but got", mat1.sizes()[0],
        ", and ", mat2.sizes()[0]);
    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "mat1 and mat2 k dim must be the same but got", mat1.sizes()[2],
        ", and ", mat2.sizes()[2]);

    auto const num_groups = masked_m.sizes()[0];
    auto const max_m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];
    auto const k = mat1.sizes()[2];

    at::Tensor out = torch::empty({num_groups, max_m, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));

    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());


    void* out_ptr = reinterpret_cast<void*>(out.data_ptr());
    void* mat1_ptr = reinterpret_cast<void*>(mat1.data_ptr());
    void* mat2_ptr = reinterpret_cast<void*>(mat2.data_ptr());
    int32_t* mat1_scales_ptr = static_cast<int32_t*>(mat1Scale.data_ptr());
    int32_t* mat2_scales_ptr = static_cast<int32_t*>(mat2Scale.data_ptr());
    int* masked_m_ptr = reinterpret_cast<int*>(masked_m.data_ptr());

    gemm_runner->group_gemm_mxfp8_nt_groupwise_masked(
        out_ptr, mat1_ptr, mat2_ptr,
        masked_m_ptr,
        num_groups, max_m, n, k, stream,
        mat1_scales_ptr, mat2_scales_ptr, static_cast<int>(granK));


    return out;
}

torch::Tensor moe_gemm_mxfp8_nt_groupwise(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& token_offset, int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);
    CHECK_CONTIGUOUS(mat1);
    CHECK_CONTIGUOUS(mat2);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a batched matrix");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batched matrix");


    TORCH_CHECK(mat1.size(-1) == mat2.size(-1), "mat1 and mat2 k dim must be the same but got", mat1.size(-1),
        ", and ", mat2.size(-1));

    auto const total_rows = mat1.sizes()[0];
    auto const num_experts = mat2.sizes()[0];
    auto const n = mat2.sizes()[1];
    auto const k = mat2.sizes()[2];
    auto const offset_size = token_offset.sizes()[0];

    TORCH_CHECK(offset_size == num_experts + 1, "token_offset size must be same as mat1 batch dim + 1 but got", offset_size,
    ", and ", num_experts);

    at::Tensor out = torch::empty({total_rows, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));

    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());


    void* out_ptr = reinterpret_cast<void*>(out.data_ptr());
    void* mat1_ptr = reinterpret_cast<void*>(mat1.data_ptr());
    void* mat2_ptr = reinterpret_cast<void*>(mat2.data_ptr());
    int32_t* mat1_scales_ptr = static_cast<int32_t*>(mat1Scale.data_ptr());
    int32_t* mat2_scales_ptr = static_cast<int32_t*>(mat2Scale.data_ptr());
    int32_t* token_offset_ptr = reinterpret_cast<int32_t*>(token_offset.data_ptr());

    gemm_runner->moe_gemm_mxfp8_nt_groupwise(
        out_ptr, mat1_ptr, mat2_ptr,
        token_offset_ptr,
        num_experts, total_rows, n, k, stream,
        mat1_scales_ptr, mat2_scales_ptr, static_cast<int>(granK));


    return out;
}

torch::Tensor fused_moe_mxfp8_nt_groupwise(
    torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale,
    torch::Tensor const& token_offset, int64_t granK)
{
    TORCH_CHECK(granK == 32 || granK == 128,
                "FusedMoe MXFP8 granK must be 32 or 128, got ", granK);
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "mat1 dtype must be Float8_e4m3fn");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "mat2 dtype must be Float8_e4m3fn");
    TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Int,
                "mat1Scale dtype must be Int32");
    TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Int,
                "mat2Scale dtype must be Int32");
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Int,
                "token_offset dtype must be Int32");
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && mat1Scale.is_cuda() &&
                    mat2Scale.is_cuda() && token_offset.is_cuda(),
                "FusedMoe inputs must be CUDA tensors");
    TORCH_CHECK(mat1.device() == mat2.device() &&
                    mat1.device() == mat1Scale.device() &&
                    mat1.device() == mat2Scale.device() &&
                    mat1.device() == token_offset.device(),
                "FusedMoe inputs must be on the same device");
    TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
    TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
    TORCH_CHECK(token_offset.is_contiguous(), "token_offset must be contiguous");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1.data_ptr()) % 16 == 0,
                "mat1 data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat2.data_ptr()) % 16 == 0,
                "mat2 data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat1Scale.data_ptr()) % 16 == 0,
                "mat1Scale data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat2Scale.data_ptr()) % 16 == 0,
                "mat2Scale data pointer must be 16-byte aligned");
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
    TORCH_CHECK(n % 16 == 0,
                "FusedMoe MXFP8 I must be a multiple of 16, got ", n);
    TORCH_CHECK(token_offset.size(0) == num_experts + 1,
                "token_offset size must be num_experts + 1");

    auto const scale_pack_k = (k + granK * 4 - 1) / (granK * 4);
    auto const padded_m =
        (m + num_experts * 3) / 4 * 4;
    TORCH_CHECK(mat1Scale.dim() == 2 &&
                    mat1Scale.size(0) == padded_m &&
                    mat1Scale.size(1) == scale_pack_k,
                "mat1Scale must have shape [", padded_m, ", ", scale_pack_k,
                "], got ", mat1Scale.sizes());
    TORCH_CHECK(mat1Scale.stride(0) == 1 &&
                    mat1Scale.stride(1) == padded_m,
                "mat1Scale must have stride [1, padded_m]");
    TORCH_CHECK(mat2Scale.dim() == 3 &&
                    mat2Scale.size(0) == num_experts &&
                    mat2Scale.size(1) == fused_n &&
                    mat2Scale.size(2) == scale_pack_k,
                "mat2Scale must have shape [", num_experts, ", ", fused_n,
                ", ", scale_pack_k, "], got ", mat2Scale.sizes());
    TORCH_CHECK(mat2Scale.stride(0) == fused_n * scale_pack_k &&
                    mat2Scale.stride(1) == 1 &&
                    mat2Scale.stride(2) == fused_n,
                "mat2Scale must have stride [2I*scale_pack_k, 1, 2I]");

    c10::cuda::CUDAGuard device_guard(mat1.device());
    auto out = torch::empty({m, n}, torch::TensorOptions()
                                       .dtype(at::ScalarType::BFloat16)
                                       .device(mat1.device()));
    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    gemm_runner->moe_gemm_mxfp8_nt_groupwise(
        out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int32_t const*>(token_offset.data_ptr()),
        num_experts, m, n, k, stream,
        static_cast<int32_t const*>(mat1Scale.data_ptr()),
        static_cast<int32_t const*>(mat2Scale.data_ptr()),
        static_cast<int>(granK), /*is_gated=*/true);
    return out;
}

torch::Tensor group_gemm_mxfp8_nt_groupwise_contiguous(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& m_indices,
    int64_t granK, bool use_psum_layout)
{
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);
    CHECK_CONTIGUOUS(mat1);
    CHECK_CONTIGUOUS(mat2);
    TORCH_CHECK(m_indices.defined(), "m_indices must be defined "
        "(per-row group assignment for use_psum_layout=false, or "
        "per-group cumsum aligned m for use_psum_layout=true).");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2D");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D");
    TORCH_CHECK(mat1.size(-1) == mat2.size(-1), "mat1 and mat2 k dim must match but got ",
        mat1.size(-1), " vs ", mat2.size(-1));

    auto const m = mat1.sizes()[0];
    auto const num_groups = mat2.sizes()[0];
    auto const n = mat2.sizes()[1];
    auto const k = mat2.sizes()[2];
    TORCH_CHECK(num_groups > 0, "num_groups must be > 0, got ", num_groups);

    // m_indices shape contract:
    //   use_psum_layout=false (contiguous): (m,)          per-row group assignment
    //   use_psum_layout=true  (psum_layout): (num_groups,) cumsum aligned m (no leading 0)
    if (use_psum_layout) {
        TORCH_CHECK(m_indices.sizes()[0] == num_groups,
            "use_psum_layout=true requires m_indices.size(0) == num_groups (", num_groups,
            "); got ", m_indices.sizes()[0]);
    } else {
        TORCH_CHECK(m_indices.sizes()[0] == m,
            "use_psum_layout=false requires m_indices.size(0) == m (", m,
            "); got ", m_indices.sizes()[0]);
    }
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = torch::empty({m, n}, torch::TensorOptions()
                                              .dtype(at::ScalarType::BFloat16)
                                              .device(mat1.device()));

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    gemm_runner->group_gemm_mxfp8_nt_groupwise_contiguous(
        reinterpret_cast<void*>(out.data_ptr()),
        reinterpret_cast<void*>(mat1.data_ptr()),
        reinterpret_cast<void*>(mat2.data_ptr()),
        reinterpret_cast<int32_t*>(m_indices.data_ptr()),
        num_groups, m, n, k, stream,
        static_cast<int32_t*>(mat1Scale.data_ptr()),
        static_cast<int32_t*>(mat2Scale.data_ptr()),
        static_cast<int>(granK), use_psum_layout);

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> fp8_quant_and_transform_for_moe(
    torch::Tensor & input, torch::Tensor & token_offset, int64_t granK)
{
    int nDim = input.dim();
    TORCH_CHECK(nDim == 2, "input must be a 2D tensor, but got ", nDim);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(token_offset);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
                "input must be BFloat16, but got ", input.scalar_type());
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Int,
                "token_offset must be Int32, but got ", token_offset.scalar_type());
    TORCH_CHECK(granK == 32 || granK == 128, "granK must be 32 or 128, but got ", granK);

    int64_t num_experts = token_offset.sizes()[0] - 1;
    int64_t token_num = input.sizes()[0];
    int64_t size_k = input.sizes()[1];

    TORCH_CHECK(size_k % 16 == 0,
                "k dimension must be multiple of 16, but got ", size_k);

    int64_t pack_nk = granK * 4;
    int64_t m_padded = (token_num + num_experts * 3) / 4 * 4;
    int64_t k_align = (size_k + pack_nk - 1) / pack_nk;

    at::Tensor output = torch::empty({token_num, size_k}, torch::TensorOptions()
                                              .dtype(at::ScalarType::Float8_e4m3fn)
                                              .device(input.device()));
    at::Tensor outScale = torch::zeros({k_align, m_padded}, torch::TensorOptions()
                                              .dtype(at::ScalarType::Int)
                                              .device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    mxfp8_cute_sm120::quantize_mxfp8_for_moe(
        output.data_ptr(),
        outScale.data_ptr(),
        input.data_ptr(),
        token_offset.data_ptr(),
        num_experts,
        token_num,
        size_k,
        stream,
        static_cast<int>(granK)
    );

    outScale = outScale.transpose(0, 1);

    return std::make_tuple(output, outScale);
}

} // namespace torch_ext

// Declare the operator
TORCH_LIBRARY_FRAGMENT(custom_ops, m)
{
    m.def("gemm_mxfp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, "
          "Tensor mat2Scale, int granK=128) -> Tensor");
    m.def(
        "batch_gemm_mxfp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, ScalarType? "
        "out_dtype=None, int granK=128) -> Tensor");
    m.def(
        "batch_gemm_mxfp8_nt_groupwise_out(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor(a!) out, int granK=128) -> "
        "Tensor(a!)");
    m.def(
        "group_gemm_mxfp8_nt_groupwise_masked(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor masked_m, int granK=128) -> "
        "Tensor(a!)");
    m.def(
        "moe_gemm_mxfp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor token_offset, int granK=128) -> "
        "Tensor(a!)");
    m.def(
        "fused_moe_mxfp8_nt_groupwise(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor token_offset, int granK=128) -> Tensor");
    m.def(
        "group_gemm_mxfp8_nt_groupwise_contiguous(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor m_indices, int granK=128, bool use_psum_layout=False) -> "
        "Tensor");
    m.def(
        "fp8_quant_and_transform_for_moe(Tensor input, Tensor token_offset, int granK=128) -> (Tensor, Tensor)");
}

// Implement the operator
TORCH_LIBRARY_IMPL(custom_ops, CUDA, m)
{
    m.impl("gemm_mxfp8_nt_groupwise", &torch_ext::gemm_mxfp8_nt_groupwise);
    m.impl("batch_gemm_mxfp8_nt_groupwise", &torch_ext::batch_gemm_mxfp8_nt_groupwise);
    m.impl("batch_gemm_mxfp8_nt_groupwise_out", &torch_ext::batch_gemm_mxfp8_nt_groupwise_out);
    m.impl("group_gemm_mxfp8_nt_groupwise_masked", &torch_ext::group_gemm_mxfp8_nt_groupwise_masked);
    m.impl("moe_gemm_mxfp8_nt_groupwise", &torch_ext::moe_gemm_mxfp8_nt_groupwise);
    m.impl("fused_moe_mxfp8_nt_groupwise", &torch_ext::fused_moe_mxfp8_nt_groupwise);
    m.impl("group_gemm_mxfp8_nt_groupwise_contiguous", &torch_ext::group_gemm_mxfp8_nt_groupwise_contiguous);
    m.impl("fp8_quant_and_transform_for_moe", &torch_ext::fp8_quant_and_transform_for_moe);
}
