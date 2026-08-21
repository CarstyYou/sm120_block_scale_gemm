# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Production correctness + perf test for FP8 block-scale GEMM (4 variants).

Run:
    python test/test_mxfp8.py
"""

import torch
import torch.nn.functional as F

from utils import (
    per_block_cast_to_fp8_e8m0,
    transform_sf_into_required_layout,
    per_token_quant_and_transform,
    grouped_token_quant_and_transform,
    per_token_cast_to_mxfp8_for_moe_gemm,
)
from benchmark import bench_kineto, calc_diff
from common import _init_op


# ---------------------------------------------------------------------------
# Normal GEMM
# ---------------------------------------------------------------------------

def generate_normal(m: int, n: int, k: int):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    ref_d = (a @ b.t()).to(torch.bfloat16)
    fp8_a, sf_a = per_token_quant_and_transform(a)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=None, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), ref_d


def test_gemm():
    print("\n=== gemm_mxfp8_nt_groupwise ===")
    for m in (128, 4096):
        for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                     (7168, 16384), (4096, 7168), (7168, 2048)]:
            x_fp8, y_fp8, ref_d = generate_normal(m=m, n=n, k=k)

            out = torch.ops.custom_ops.gemm_mxfp8_nt_groupwise(
                x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1])
            diff = calc_diff(out, ref_d)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            def fn():
                torch.ops.custom_ops.gemm_mxfp8_nt_groupwise(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1])

            t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')


# ---------------------------------------------------------------------------
# Batched GEMM
# ---------------------------------------------------------------------------

def generate_batch(g: int, m: int, n: int, k: int,
                   accumulate: bool = False,
                   out_dtype: torch.dtype = torch.bfloat16,
                   is_permute102: bool = False):
    a_shape = (g, m, k) if not is_permute102 else (m, g, k)
    a = torch.randn(a_shape, device='cuda', dtype=torch.bfloat16)
    b = torch.randn((g, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((g, m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((g, m, n), device='cuda', dtype=out_dtype)
    c = torch.randn((g, 1, n), device='cuda', dtype=out_dtype)
    if is_permute102:
        ref_d = torch.einsum('mgk,gnk->gmn', a, b) + (d if accumulate else 0)
    else:
        ref_d = torch.einsum('gmk,gnk->gmn', a, b) + (d if accumulate else 0)
    fp8_a, sf_a = per_token_quant_and_transform(a, need_permute102=is_permute102)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=g, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), c, d, ref_d


def test_bmm():
    print("\n=== batch_gemm_mxfp8_nt_groupwise ===")
    for g in (2, 4, 8):
        for m in (128, 4096):
            for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                         (7168, 16384), (4096, 7168), (7168, 2048)]:
                x_fp8, y_fp8, _, d, ref_d = generate_batch(
                    g=g, m=m, n=n, k=k, is_permute102=True)
                out = torch.ops.custom_ops.batch_gemm_mxfp8_nt_groupwise(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1])
                diff = calc_diff(out, ref_d)
                assert diff < 0.001, f'{g=}, {m=}, {k=}, {n=}, {diff:.5f}'

                def fn():
                    torch.ops.custom_ops.batch_gemm_mxfp8_nt_groupwise_out(
                        x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], d)

                t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
                print(f' > Performance (g={g:2}, m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                      f'throughput: {2 * g * m * n * k / t / 1e12:4.0f} TFLOPS, '
                      f'{(g * m * k + g * k * n + g * m * n * 2) / 1e9 / t:4.0f} GB/s')


# ---------------------------------------------------------------------------
# Masked grouped GEMM
# ---------------------------------------------------------------------------

def generate_grouped(num_groups: int, max_m: int, n: int, k: int):
    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups,), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(max_m * 0.5)

    fp8_a, sf_a = grouped_token_quant_and_transform(a, masked_m)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=num_groups, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), masked_m, ref_d


def test_grouped():
    print("\n=== group_gemm_mxfp8_nt_groupwise_masked ===")
    for g in (2, 4, 8):
        for max_m in (128, 4096):
            for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                         (7168, 16384), (4096, 7168), (7168, 2048)]:
                x_fp8, y_fp8, masked_m, ref_d = generate_grouped(
                    num_groups=g, max_m=max_m, n=n, k=k)
                out = torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_masked(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], masked_m)
                for j in range(g):
                    diff = calc_diff(out[j, :masked_m[j].item()],
                                     ref_d[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{g=}, {max_m=}, {k=}, {n=}, {diff:.5f}'

                def fn():
                    torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_masked(
                        x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], masked_m)

                t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
                print(f' > Performance (g={g:2}, m={max_m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                      f'throughput: {2 * g * max_m * 0.5 * n * k / t / 1e12:4.0f} TFLOPS, '
                      f'{(g * max_m * 0.5 * k + g * k * n + g * max_m * 0.5 * n * 2) / 1e9 / t:4.0f} GB/s')


# ---------------------------------------------------------------------------
# MoE GEMM (ZeroPadding convention)
# ---------------------------------------------------------------------------

def generate_moe(num_rows: int, topk: int, num_experts: int,
                 hidden_dim: int, inter_dim: int):
    # Random routing
    expert_ids = torch.randint(0, num_experts, (num_rows, topk),
                               device='cuda', dtype=torch.int)
    token_per_expert = torch.bincount(expert_ids.flatten(),
                                      minlength=num_experts).int()
    token_offset = torch.cumsum(token_per_expert, dim=0).to(torch.int32)
    token_offset = torch.cat([torch.zeros((1,), device='cuda', dtype=torch.int32),
                              token_offset], dim=0)

    total_rows = token_per_expert.sum()
    a = torch.randn((total_rows, hidden_dim), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_experts, inter_dim, hidden_dim),
                    device='cuda', dtype=torch.bfloat16)

    # Reference: per-expert dense matmul
    ref_d = torch.zeros(total_rows, inter_dim, device='cuda', dtype=torch.bfloat16)
    for i in range(num_experts):
        start = token_offset[i]
        end = token_offset[i + 1]
        if start < end:
            ref_d[start:end] = a[start:end] @ b[i].t()

    # Quantize via Python reference helper (transparent SF-layout construction)
    fp8_a, sf_a = per_token_cast_to_mxfp8_for_moe_gemm(a, token_offset)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=num_experts, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), token_offset, ref_d


def generate_fused_moe(
        token_counts, hidden_dim: int, inter_dim: int, gran_k: int = 128):
    num_experts = len(token_counts)
    token_per_expert = torch.tensor(
        token_counts, device="cuda", dtype=torch.int32)
    token_offset = torch.cat((
        torch.zeros((1,), device="cuda", dtype=torch.int32),
        torch.cumsum(token_per_expert, dim=0, dtype=torch.int32)))
    total_rows = sum(token_counts)
    a = torch.randn(
        (total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(
        (num_experts, inter_dim, hidden_dim),
        device="cuda", dtype=torch.bfloat16)
    w_gate = torch.randn(
        (num_experts, inter_dim, hidden_dim),
        device="cuda", dtype=torch.bfloat16)
    w13 = torch.cat((w_up, w_gate), dim=1).contiguous()

    ref_d = torch.zeros(
        (total_rows, inter_dim), device="cuda", dtype=torch.bfloat16)
    for expert_idx in range(num_experts):
        start = token_offset[expert_idx].item()
        end = token_offset[expert_idx + 1].item()
        if start < end:
            up = a[start:end] @ w_up[expert_idx].t()
            gate = a[start:end] @ w_gate[expert_idx].t()
            ref_d[start:end] = F.silu(gate) * up

    padded_k = ((hidden_dim + 127) // 128) * 128
    a_padded = torch.zeros(
        (total_rows, padded_k), device="cuda", dtype=torch.bfloat16)
    a_padded[:, :hidden_dim] = a
    w13_padded = torch.zeros(
        (num_experts, 2 * inter_dim, padded_k),
        device="cuda", dtype=torch.bfloat16)
    w13_padded[:, :, :hidden_dim] = w13
    fp8_a_padded, sf_a = per_token_cast_to_mxfp8_for_moe_gemm(
        a_padded, token_offset, gran_k=gran_k)
    fp8_w13_padded, sf_w13 = per_block_cast_to_fp8_e8m0(
        w13_padded, gran_k=gran_k)
    fp8_a = fp8_a_padded[:, :hidden_dim].contiguous()
    fp8_w13 = fp8_w13_padded[:, :, :hidden_dim].contiguous()
    sf_w13 = transform_sf_into_required_layout(
        sf=sf_w13, mn=2 * inter_dim, k=padded_k,
        recipe=(1, gran_k, gran_k), num_groups=num_experts, is_sfa=False)
    return (fp8_a, sf_a), (fp8_w13, sf_w13), token_offset, ref_d


def test_fused_moe_mxfp8_nt_groupwise():
    print("\n=== fused_moe_mxfp8_nt_groupwise ===")
    token_counts = (0, 1, 63, 128, 129)
    for gran_k in (32, 128):
        for inter_dim, hidden_dim in (
                (80, 128), (128, 144), (192, 512), (80, 1024)):
            x_fp8, w13_fp8, token_offset, ref_d = generate_fused_moe(
                token_counts, hidden_dim=hidden_dim, inter_dim=inter_dim,
                gran_k=gran_k)
            out = torch.ops.custom_ops.fused_moe_mxfp8_nt_groupwise(
                x_fp8[0], w13_fp8[0], x_fp8[1], w13_fp8[1], token_offset,
                gran_k)
            diff = calc_diff(out, ref_d)
            assert diff < 0.002, (
                f"{gran_k=}, {inter_dim=}, {hidden_dim=}, {diff:.5f}")

    for gran_k in (32, 128):
        for zero_expert_counts in ((1, 0, 63, 128, 129),
                                   (1, 63, 128, 129, 0)):
            x_fp8, w13_fp8, token_offset, ref_d = generate_fused_moe(
                zero_expert_counts, hidden_dim=128, inter_dim=128,
                gran_k=gran_k)
            out = torch.ops.custom_ops.fused_moe_mxfp8_nt_groupwise(
                x_fp8[0], w13_fp8[0], x_fp8[1], w13_fp8[1], token_offset,
                gran_k)
            diff = calc_diff(out, ref_d)
            assert diff < 0.002, (
                f"{gran_k=}, {zero_expert_counts=}, {diff:.5f}")


def test_fused_moe_mxfp8_rejects_invalid_i():
    for inter_dim in (78, 84):
        x_fp8, w13_fp8, token_offset, _ = generate_fused_moe(
            (1, 1), hidden_dim=128, inter_dim=inter_dim)
        try:
            torch.ops.custom_ops.fused_moe_mxfp8_nt_groupwise(
                x_fp8[0], w13_fp8[0], x_fp8[1], w13_fp8[1], token_offset)
        except RuntimeError as error:
            assert "multiple of 16" in str(error)
        else:
            raise AssertionError(f"invalid {inter_dim=} was not rejected")


def test_group_gemm():
    print("\n=== moe_gemm_mxfp8_nt_groupwise ===")
    for num_experts in (4, 8, 16):
        for num_rows, topk in [(32, 4), (256, 8), (512, 16), (4096, 32)]:
            for n, k in [(4096, 7168), (7168, 4096)]:
                x_fp8, y_fp8, token_offset, ref_d = generate_moe(
                    num_rows=num_rows, topk=topk, num_experts=num_experts,
                    hidden_dim=k, inter_dim=n)
                out = torch.ops.custom_ops.moe_gemm_mxfp8_nt_groupwise(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], token_offset)
                diff = calc_diff(out, ref_d)
                assert diff < 0.001, (f'{num_experts=}, {num_rows=}, {topk=}, '
                                      f'{n=}, {k=}, {diff:.5f}')

                def fn():
                    torch.ops.custom_ops.moe_gemm_mxfp8_nt_groupwise(
                        x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], token_offset)

                t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
                total_rows = ref_d.shape[0]
                print(f' > Performance (E={num_experts:2}, rows={num_rows:5}, topk={topk:3}, '
                      f'n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                      f'throughput: {2 * total_rows * n * k / t / 1e12:4.0f} TFLOPS, '
                      f'{(total_rows * k + num_experts * n * k + total_rows * n * 2) / 1e9 / t:4.0f} GB/s')


# ---------------------------------------------------------------------------
# Grouped MoE GEMM (Contiguous, use_psum_layout=True — PsumLayout convention)
# ---------------------------------------------------------------------------

def generate_grouped_contiguous_psum_layout(num_groups: int, m_per_group: int, n: int, k: int):
    m = num_groups * m_per_group
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    m_indices = torch.tensor(
        [(i + 1) * m_per_group for i in range(num_groups)],
        dtype=torch.int32, device='cuda',
    )

    ref_d = torch.zeros(m, n, device='cuda', dtype=torch.bfloat16)
    for j in range(num_groups):
        s = j * m_per_group
        e = (j + 1) * m_per_group
        ref_d[s:e] = a[s:e] @ b[j].t()

    fp8_a, sf_a = per_token_quant_and_transform(a)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=num_groups, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), m_indices, ref_d


def test_group_gemm_contiguous_psum_layout():
    print("\n=== group_gemm_mxfp8_nt_groupwise_contiguous (use_psum_layout=True) ===")
    for num_groups in (4, 8):
        for m_per_group in (64, 128, 256):
            for n, k in [(4096, 7168), (7168, 4096)]:
                x_fp8, y_fp8, m_indices, ref_d = generate_grouped_contiguous_psum_layout(
                    num_groups=num_groups, m_per_group=m_per_group, n=n, k=k)
                out = torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_contiguous(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], m_indices, 128, True)
                diff = calc_diff(out, ref_d)
                assert diff < 0.001, (f'{num_groups=}, {m_per_group=}, '
                                      f'{n=}, {k=}, {diff:.5f}')

                def fn():
                    torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_contiguous(
                        x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], m_indices, 128, True)

                t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
                total_m = num_groups * m_per_group
                print(f' > Performance (E={num_groups:2}, m_per_E={m_per_group:4}, '
                      f'n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                      f'throughput: {2 * total_m * n * k / t / 1e12:4.0f} TFLOPS, '
                      f'{(total_m * k + num_groups * n * k + total_m * n * 2) / 1e9 / t:4.0f} GB/s')


# ---------------------------------------------------------------------------
# Grouped MoE GEMM (Contiguous, use_psum_layout=False — DG-canonical MGroupedContiguous)
# ---------------------------------------------------------------------------

_MGROUP_ALIGN_M = 128


def _mgroup_align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def generate_grouped_contiguous(num_groups: int, actual_m_per_group: int,
                                                  n: int, k: int):
    aligned_m_per_group = _mgroup_align(actual_m_per_group, _MGROUP_ALIGN_M)
    m_total = num_groups * aligned_m_per_group

    a = torch.zeros((m_total, k), device='cuda', dtype=torch.bfloat16)
    m_indices = torch.empty(m_total, dtype=torch.int32, device='cuda')
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    for j in range(num_groups):
        base = j * aligned_m_per_group
        actual_end = base + actual_m_per_group
        aligned_end = base + aligned_m_per_group
        a[base:actual_end] = torch.randn(actual_m_per_group, k, device='cuda', dtype=torch.bfloat16)
        m_indices[base:actual_end] = j
        m_indices[actual_end:aligned_end] = -1

    ref_d = torch.zeros(m_total, n, device='cuda', dtype=torch.bfloat16)
    for j in range(num_groups):
        base = j * aligned_m_per_group
        aligned_end = base + aligned_m_per_group
        ref_d[base:aligned_end] = a[base:aligned_end] @ b[j].t()

    fp8_a, sf_a = per_token_quant_and_transform(a)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = transform_sf_into_required_layout(
        sf=sf_b, mn=b.shape[-2], k=b.shape[-1],
        recipe=(1, 128, 128), num_groups=num_groups, is_sfa=False)
    return (fp8_a, sf_a), (fp8_b, sf_b), m_indices, ref_d


def test_group_gemm_contiguous():
    print("\n=== group_gemm_mxfp8_nt_groupwise_contiguous (use_psum_layout=False, MGroupedContiguous) ===")
    cells = [(4, 8), (4, 32), (4, 96), (4, 128), (4, 192), (4, 256),
             (8, 8), (8, 64), (8, 128), (8, 192)]
    for num_groups, actual_m in cells:
        for n, k in [(4096, 7168), (7168, 4096)]:
            x_fp8, y_fp8, m_indices, ref_d = generate_grouped_contiguous(
                num_groups=num_groups, actual_m_per_group=actual_m, n=n, k=k)
            out = torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_contiguous(
                x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], m_indices, 128, False)
            diff = calc_diff(out, ref_d)
            assert diff < 0.001, (f'{num_groups=}, {actual_m=}, {n=}, {k=}, {diff:.5f}')

            def fn():
                torch.ops.custom_ops.group_gemm_mxfp8_nt_groupwise_contiguous(
                    x_fp8[0], y_fp8[0], x_fp8[1], y_fp8[1], m_indices, 128, False)

            t = bench_kineto(fn, 'device_kernel', suppress_kineto_output=True)
            aligned_m = _mgroup_align(actual_m, _MGROUP_ALIGN_M)
            total_m = num_groups * aligned_m
            print(f' > Performance (E={num_groups:2}, actual_m={actual_m:4}, aligned={aligned_m:4}, '
                  f'n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * total_m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(total_m * k + num_groups * n * k + total_m * n * 2) / 1e9 / t:4.0f} GB/s')


if __name__ == '__main__':
    _init_op()
    test_gemm()
    test_bmm()
    test_grouped()
    test_group_gemm()
    test_fused_moe_mxfp8_nt_groupwise()
    test_fused_moe_mxfp8_rejects_invalid_i()
    test_group_gemm_contiguous_psum_layout()
    test_group_gemm_contiguous()
