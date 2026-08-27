"""Production BF16 GEMM correctness entry across GemmType."""

from pathlib import Path

import torch

from benchmark import calc_diff

PROJECT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = PROJECT_DIR.parent


def load_prod_op():
    lib_path = ROOT_DIR / "build" / "thop" / "libth_op.so"
    torch.ops.load_library(str(lib_path))


def _matmul_nt_fp32(a, b):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(a.float(), b.float().transpose(-1, -2))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _assert_close(out, ref_d, label):
    diff = calc_diff(out, ref_d)
    assert diff < 0.001, f"{label}, {diff:.5f}"


def generate_normal(m: int, n: int, k: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    ref_d = _matmul_nt_fp32(a, b).to(torch.bfloat16)
    return a, b, ref_d


def test_gemm_bf16():
    load_prod_op()
    for m in (128, 4096):
        for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                     (7168, 16384), (4096, 7168), (7168, 2048)]:
            x, y, ref_d = generate_normal(m=m, n=n, k=k)
            out = torch.ops.custom_ops.gemm_bf16(x, y)
            _assert_close(out, ref_d, f"{m=}, {k=}, {n=}")
            del x, y, ref_d, out


def generate_batch(g: int, m: int, n: int, k: int):
    a = torch.randn((g, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((g, n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((g, m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = _matmul_nt_fp32(a, b).to(torch.bfloat16)
    return a, b, d, ref_d


def test_batch_gemm_bf16():
    load_prod_op()
    for g in (2, 4, 8):
        for m in (128, 4096):
            for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                         (7168, 16384), (4096, 7168), (7168, 2048)]:
                x, y, d, ref_d = generate_batch(g=g, m=m, n=n, k=k)
                out = torch.ops.custom_ops.batch_gemm_bf16(x, y)
                _assert_close(out, ref_d, f"{g=}, {m=}, {k=}, {n=}")

                returned = torch.ops.custom_ops.batch_gemm_bf16_out(x, y, d)
                assert returned.data_ptr() == d.data_ptr()
                _assert_close(d, ref_d, f"out {g=}, {m=}, {k=}, {n=}")
                del x, y, d, ref_d, out, returned


def generate_grouped(num_groups: int, max_m: int, n: int, k: int):
    a = torch.randn((num_groups, max_m, k),
                    device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k),
                    device="cuda", dtype=torch.bfloat16)
    ref_d = _matmul_nt_fp32(a, b).to(torch.bfloat16)

    masked_m = torch.full(
        (num_groups,), max_m // 2, device="cuda", dtype=torch.int32)
    return a, b, masked_m, ref_d


def test_group_gemm_bf16_masked():
    load_prod_op()
    for g in (2, 4, 8):
        for max_m in (128, 4096):
            for n, k in [(2112, 7168), (24576, 1536), (32768, 512),
                         (7168, 16384), (4096, 7168), (7168, 2048)]:
                x, y, masked_m, ref_d = generate_grouped(
                    num_groups=g, max_m=max_m, n=n, k=k)
                out = torch.ops.custom_ops.group_gemm_bf16_masked(
                    x, y, masked_m)
                for j in range(g):
                    valid_m = masked_m[j].item()
                    _assert_close(
                        out[j, :valid_m], ref_d[j, :valid_m],
                        f"{g=}, {max_m=}, {k=}, {n=}, {j=}")
                del x, y, masked_m, ref_d, out


def generate_moe(num_rows: int, topk: int, num_experts: int,
                 hidden_dim: int, inter_dim: int, expert_ids=None):
    if expert_ids is None:
        expert_ids = torch.randint(
            0, num_experts, (num_rows, topk),
            device="cuda", dtype=torch.int32)
    else:
        assert expert_ids.shape == (num_rows, topk)
        expert_ids = expert_ids.to(device="cuda", dtype=torch.int32)
    token_per_expert = torch.bincount(
        expert_ids.flatten(), minlength=num_experts).to(torch.int32)
    token_offset = torch.cat((
        torch.zeros((1,), device="cuda", dtype=torch.int32),
        torch.cumsum(token_per_expert, dim=0).to(torch.int32)))

    total_rows = num_rows * topk
    a = torch.randn(
        (total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    b = torch.randn(
        (num_experts, inter_dim, hidden_dim),
        device="cuda", dtype=torch.bfloat16)

    ref_d = torch.zeros(
        (total_rows, inter_dim), device="cuda", dtype=torch.bfloat16)
    for i in range(num_experts):
        start = token_offset[i].item()
        end = token_offset[i + 1].item()
        if start < end:
            ref_d[start:end] = _matmul_nt_fp32(
                a[start:end], b[i]).to(torch.bfloat16)
    return a, b, token_offset, ref_d


def generate_fused_moe(token_counts, hidden_dim: int, inter_dim: int):
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
            up = _matmul_nt_fp32(a[start:end], w_up[expert_idx])
            gate = _matmul_nt_fp32(a[start:end], w_gate[expert_idx])
            silu_gate = gate / (1.0 + torch.exp(-gate))
            ref_d[start:end] = (silu_gate * up).to(torch.bfloat16)
    return a, w13, token_offset, ref_d


def test_fused_moe_bf16():
    load_prod_op()
    token_counts = (0, 1, 63, 128, 129)
    for inter_dim, hidden_dim in (
            (64, 128), (128, 144), (192, 512), (64, 1024)):
        x, w13, token_offset, ref_d = generate_fused_moe(
            token_counts, hidden_dim=hidden_dim, inter_dim=inter_dim)
        out = torch.ops.custom_ops.fused_moe_bf16(
            x, w13, token_offset)
        _assert_close(
            out, ref_d, f"{inter_dim=}, {hidden_dim=}")

    for zero_expert_counts in ((1, 0, 63, 128, 129),
                               (1, 63, 128, 129, 0)):
        x, w13, token_offset, ref_d = generate_fused_moe(
            zero_expert_counts, hidden_dim=128, inter_dim=128)
        out = torch.ops.custom_ops.fused_moe_bf16(
            x, w13, token_offset)
        _assert_close(out, ref_d, f"{zero_expert_counts=}")


def test_fused_moe_bf16_rejects_invalid_i():
    load_prod_op()
    x, w13, token_offset, _ = generate_fused_moe(
        (1, 1), hidden_dim=128, inter_dim=84)
    try:
        torch.ops.custom_ops.fused_moe_bf16(x, w13, token_offset)
    except RuntimeError as error:
        assert "multiple of 8" in str(error)
    else:
        raise AssertionError("invalid inter_dim=84 was not rejected")


def test_moe_gemm_bf16():
    load_prod_op()
    for num_experts in (4, 8, 16):
        for num_rows, topk in [(32, 4), (256, 8), (512, 16), (4096, 32)]:
            for n, k in [(4096, 7168), (7168, 4096)]:
                x, y, token_offset, ref_d = generate_moe(
                    num_rows=num_rows, topk=topk, num_experts=num_experts,
                    hidden_dim=k, inter_dim=n)
                out = torch.ops.custom_ops.moe_gemm_bf16(
                    x, y, token_offset)
                _assert_close(
                    out, ref_d,
                    f"{num_experts=}, {num_rows=}, {topk=}, {n=}, {k=}")
                del x, y, token_offset, ref_d, out


def generate_grouped_contiguous_psum_layout(
        num_groups: int, m_per_group: int, n: int, k: int):
    m = num_groups * m_per_group
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.tensor(
        [(i + 1) * m_per_group for i in range(num_groups)],
        dtype=torch.int32, device="cuda")

    ref_d = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
    for j in range(num_groups):
        start = j * m_per_group
        end = (j + 1) * m_per_group
        ref_d[start:end] = _matmul_nt_fp32(
            a[start:end], b[j]).to(torch.bfloat16)
    return a, b, m_indices, ref_d


def test_group_gemm_bf16_contiguous_psum():
    load_prod_op()
    for num_groups in (4, 8):
        for m_per_group in (64, 128, 256):
            for n, k in [(4096, 7168), (7168, 4096)]:
                x, y, m_indices, ref_d = (
                    generate_grouped_contiguous_psum_layout(
                        num_groups=num_groups, m_per_group=m_per_group,
                        n=n, k=k))
                out = torch.ops.custom_ops.group_gemm_bf16_contiguous(
                    x, y, m_indices, True)
                _assert_close(
                    out, ref_d,
                    f"{num_groups=}, {m_per_group=}, {n=}, {k=}")
                del x, y, m_indices, ref_d, out


_MGROUP_ALIGN_M = 128


def _mgroup_align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def generate_grouped_contiguous(
        num_groups: int, actual_m_per_group: int, n: int, k: int):
    aligned_m_per_group = _mgroup_align(actual_m_per_group, _MGROUP_ALIGN_M)
    m_total = num_groups * aligned_m_per_group

    a = torch.zeros((m_total, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m_total, dtype=torch.int32, device="cuda")
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)

    for j in range(num_groups):
        base = j * aligned_m_per_group
        actual_end = base + actual_m_per_group
        aligned_end = base + aligned_m_per_group
        a[base:actual_end] = torch.randn(
            (actual_m_per_group, k), device="cuda", dtype=torch.bfloat16)
        m_indices[base:actual_end] = j
        m_indices[actual_end:aligned_end] = -1

    ref_d = torch.zeros((m_total, n), device="cuda", dtype=torch.bfloat16)
    for j in range(num_groups):
        base = j * aligned_m_per_group
        aligned_end = base + aligned_m_per_group
        ref_d[base:aligned_end] = _matmul_nt_fp32(
            a[base:aligned_end], b[j]).to(torch.bfloat16)
    return a, b, m_indices, ref_d


def test_group_gemm_bf16_contiguous():
    load_prod_op()
    cells = [(4, 8), (4, 32), (4, 96), (4, 128), (4, 192), (4, 256),
             (8, 8), (8, 64), (8, 128), (8, 192)]
    for num_groups, actual_m in cells:
        for n, k in [(4096, 7168), (7168, 4096)]:
            x, y, m_indices, ref_d = generate_grouped_contiguous(
                num_groups=num_groups, actual_m_per_group=actual_m, n=n, k=k)
            out = torch.ops.custom_ops.group_gemm_bf16_contiguous(
                x, y, m_indices, False)
            valid = m_indices >= 0
            _assert_close(
                out[valid], ref_d[valid],
                f"{num_groups=}, {actual_m=}, {n=}, {k=}")
            del x, y, m_indices, ref_d, out


def test_small_m():
    load_prod_op()
    for m in (8, 9, 128, 129):
        x, y, ref_d = generate_normal(m=m, n=128, k=128)
        out = torch.ops.custom_ops.gemm_bf16(x, y)
        _assert_close(out, ref_d, f"Normal {m=}")
        del x, y, ref_d, out

        x, y, d, ref_d = generate_batch(g=2, m=m, n=128, k=128)
        out = torch.ops.custom_ops.batch_gemm_bf16(x, y)
        _assert_close(out, ref_d, f"Batched {m=}")
        del x, y, d, ref_d, out

        x, y, masked_m, ref_d = generate_grouped(
            num_groups=2, max_m=m, n=128, k=128)
        out = torch.ops.custom_ops.group_gemm_bf16_masked(
            x, y, masked_m)
        for group_idx in range(2):
            valid_m = masked_m[group_idx].item()
            _assert_close(
                out[group_idx, :valid_m], ref_d[group_idx, :valid_m],
                f"Masked {m=}")
        del x, y, masked_m, ref_d, out


def test_grouped_small_m():
    load_prod_op()
    for m_per_group in (8, 16):
        x, y, m_indices, ref_d = (
            generate_grouped_contiguous_psum_layout(
                num_groups=2, m_per_group=m_per_group, n=128, k=128))
        out = torch.ops.custom_ops.group_gemm_bf16_contiguous(
            x, y, m_indices, True)
        _assert_close(out, ref_d, f"Psum {m_per_group=}")
        del x, y, m_indices, ref_d, out

    for num_rows in (2, 16):
        expert_ids = torch.zeros(
            (num_rows, 4), device="cuda", dtype=torch.int32)
        x, y, token_offset, ref_d = generate_moe(
            num_rows=num_rows, topk=4, num_experts=4,
            hidden_dim=128, inter_dim=128, expert_ids=expert_ids)
        assert (token_offset[1:] == token_offset[:-1]).any()
        out = torch.ops.custom_ops.moe_gemm_bf16(x, y, token_offset)
        _assert_close(out, ref_d, f"ZeroPadding {num_rows=}")
        del x, y, token_offset, ref_d, out


if __name__ == "__main__":
    test_gemm_bf16()
    test_batch_gemm_bf16()
    test_group_gemm_bf16_masked()
    test_moe_gemm_bf16()
    test_fused_moe_bf16()
    test_fused_moe_bf16_rejects_invalid_i()
    test_group_gemm_bf16_contiguous_psum()
    test_group_gemm_bf16_contiguous()
    test_small_m()
    test_grouped_small_m()
