import torch
from typing import Optional, Tuple, Union
import triton
import triton.language as tl

_num_sms = None


def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert 0 < num_sms <= torch.cuda.get_device_properties(device='cuda').multi_processor_count
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count
    return _num_sms

def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def get_m_alignment_for_contiguous_layout():
    return 128

def get_k_alignment_for_contiguous_layout():
    return 128

def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)

def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True

    b, m, n = x.shape
    aligned_m = get_tma_aligned_size(m, x.element_size())

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


@torch.compile(dynamic=True)
def per_token_cast_to_fp8(
        x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization with FP32 scales."""
    assert x.dim() in (2, 3)
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(0)

    g, m, k = x.shape
    padded_k = align(k, gran_k)
    x_padded = torch.zeros((g, m, padded_k), dtype=x.dtype, device=x.device)
    x_padded[:, :, :k] = x
    x_view = x_padded.view(g, m, padded_k // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=3).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf.unsqueeze(3))).to(torch.float8_e4m3fn)
    x_scaled = x_scaled.view_as(x_padded)[:, :, :k].contiguous()

    if squeezed:
        x_scaled = x_scaled.squeeze(0)
        sf = sf.squeeze(0)

    return x_scaled, sf


@torch.compile(dynamic=True)
def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        assert x.size(1) % gran_k == 0
        m, n = x.shape
        x_view = x.view(m, -1, gran_k)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
            m, n), sf
    else:
        assert x.size(2) % gran_k == 0
        g, m, n = x.shape
        x_view = x.view(g, m, -1, gran_k)
        x_amax = x_view.abs().float().amax(dim=3).view(g, m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(3))).to(torch.float8_e4m3fn).view(
            g, m, n), sf


def unpack_ue8m0_scales(sf_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack int32 containing 4 UE8M0 scales and convert to float.

    Packing format (from Triton kernel):
        packed = (s3 << 24) | (s2 << 16) | (s1 << 8) | s0
        where s0 is for block 0, s1 for block 1, etc.

    Args:
        sf_packed: int32 tensor, shape [..., num_packed_int32].
                   Each int32 packs 4 UE8M0 scales covering 4 consecutive
                   gran_k-element blocks (gran_k in {32, 128}).

    Returns:
        float tensor, shape [..., num_packed_int32 * 4].
    """
    # Extract 4 uint8 values from each int32
    # Triton packs: pack_index=0 -> bits 0-7, pack_index=1 -> bits 8-15, etc.
    s0 = sf_packed & 0xFF                # bits 0-7   (block 0)
    s1 = (sf_packed >> 8) & 0xFF         # bits 8-15  (block 1)
    s2 = (sf_packed >> 16) & 0xFF        # bits 16-23 (block 2)
    s3 = (sf_packed >> 24) & 0xFF        # bits 24-31 (block 3)
    
    # Stack in correct order: [block0, block1, block2, block3]
    scales_uint8 = torch.stack([s0, s1, s2, s3], dim=-1)
    original_shape = sf_packed.shape[:-1]
    scales_uint8 = scales_uint8.view(*original_shape, -1)
    
    # Convert UE8M0 to float: value = 2^(exp - 127)
    scales_float = torch.pow(2.0, scales_uint8.float() - 127.0)
    return scales_float


@torch.compile(dynamic=True)
def per_token_dequant_from_fp8_e8m0(
        x_fp8: torch.Tensor,
        sf_packed: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        gran_k: int = 128,
    ) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to original dtype using packed E8M0 scale factors.

    Args:
        x_fp8: Quantized FP8 tensor, shape [m, n] or [b, m, n]
        sf_packed: Packed scale factors (int32), shape [m, ceil(n/(gran_k*4))]
                   or [b, m, ceil(n/(gran_k*4))].
                   Each int32 contains 4 UE8M0 scales for 4 consecutive
                   gran_k-element blocks. May be padded to align to 4 scales
                   (gran_k*4 elements).
        dtype: Output dtype (default: bfloat16).
        gran_k: K-axis quantization granularity (default 128; OCP MXFP8 uses 32).

    Returns:
        Dequantized tensor with original shape.
    """
    # Unpack scales: [..., ceil(n/(gran_k*4))] -> [..., ceil(n/(gran_k*4))*4]
    sf = unpack_ue8m0_scales(sf_packed)

    if x_fp8.dim() == 2:
        m, n = x_fp8.shape
        num_blocks = (n + gran_k - 1) // gran_k
        x_view = x_fp8.view(m, -1, gran_k).to(torch.float32)
        sf = sf[:, :num_blocks]
        x_dequant = x_view * sf.unsqueeze(2)
        return x_dequant.view(m, n).to(dtype)
    else:
        b, m, n = x_fp8.shape
        num_blocks = (n + gran_k - 1) // gran_k
        x_view = x_fp8.view(b, m, -1, gran_k).to(torch.float32)
        sf = sf[:, :, :num_blocks]
        x_dequant = x_view * sf.unsqueeze(3)
        result = x_dequant.view(b, m, n).to(dtype)

        return result


@torch.compile(dynamic=True)
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 or x.dim() == 3
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(0)

    g, m, n = x.shape
    x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_scaled = x_scaled.view_as(x_padded)[:, :m, :n].contiguous()
    sf = sf.view(x_view.size(0), x_view.size(1), x_view.size(3))

    if squeezed:
        x_scaled = x_scaled.squeeze(0)
        sf = sf.squeeze(0)

    return x_scaled, sf

def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D-block FP8 quantization with UE8M0 scale (DeepGEMM-aligned).

    `gran_k` controls BOTH the M-axis and K-axis block size — the produced
    scale tensor has shape `(ceil(m/gran_k), ceil(n/gran_k))`, i.e. a
    symmetric `(gran_k, gran_k)` 2D block. This matches DeepGEMM's
    `per_block_cast_to_fp8(gran_k=...)` convention.

    Supported values: `gran_k in {32, 128}`. The corresponding transform
    recipe for B is `(gran_k, gran_k)` — i.e. `(128, 128)` (DeepSeek-style)
    or `(32, 32)` (MXFP8 OCP). Asymmetric blocks like `(128, 32)` are NOT
    supported by this helper.
    """
    assert x.dim() == 2 or x.dim() == 3
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(0)

    g, m, n = x.shape
    x_padded = torch.zeros((g, align(m, gran_k), align(n, gran_k)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(g, -1, gran_k, x_padded.size(-1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_scaled = x_scaled.view_as(x_padded)[:, :m, :n].contiguous()
    sf = sf.view(x_view.size(0), x_view.size(1), x_view.size(3))

    if squeezed:
        x_scaled = x_scaled.squeeze(0)
        sf = sf.squeeze(0)

    return x_scaled, sf

def resmooth_to_fp8_e8m0_old(weight: torch.Tensor,
                         sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = weight.cuda()
    sf = sf.cuda()
    if weight.dim() == 2:
        x = weight.float() * sf.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1)[:weight.shape[0], :weight.shape[1]]
    else:
        x = weight.float() * sf.repeat_interleave(128, dim=1).repeat_interleave(
            128, dim=2)[:weight.shape[0], :weight.shape[1], :weight.shape[2]]
    return per_block_cast_to_fp8_e8m0(x)
    
@triton.jit
def _resmooth_kernel(
    w_ptr,
    s_ptr,
    M,
    K,
    stride_wb,
    stride_wm,
    stride_wk,
    stride_sb,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    curr_w_ptr = w_ptr + batch_idx * stride_wb
    curr_s_ptr = s_ptr + batch_idx * stride_sb

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    s_offset = pid_m * stride_sm + pid_k * stride_sk
    old_scale = tl.load(curr_s_ptr + s_offset)

    w_mask = (rm[:, None] < M) & (rk[None, :] < K)
    w_offsets = rm[:, None] * stride_wm + rk[None, :] * stride_wk
    w_fp8 = tl.load(curr_w_ptr + w_offsets, mask=w_mask, other=0.0)
    w_fp32 = w_fp8.to(tl.float32)

    w_val = w_fp32 * old_scale
    block_amax = tl.maximum(tl.max(tl.abs(w_val)), 1e-4)

    new_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(block_amax / 448.0)))
    w_requant = w_val * (1.0 / new_scale)

    w_requant_fp8 = w_requant.to(tl.float8e4nv)
    tl.store(curr_w_ptr + w_offsets, w_requant_fp8, mask=w_mask)
    tl.store(curr_s_ptr + s_offset, new_scale)


def resmooth_to_fp8_e8m0(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: tuple[int, int] = (128, 128),
):
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.float32

    weight = weight.cuda()
    weight_scale = weight_scale.cuda()

    orig_shape = weight.shape
    M, K = orig_shape[-2:]
    w_view = weight.view(-1, M, K)
    s_view = weight_scale.view(-1, weight_scale.shape[-2],
                               weight_scale.shape[-1])

    num_batches = w_view.shape[0]
    BLOCK_M, BLOCK_K = block_size

    grid = (num_batches, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _resmooth_kernel[grid](
        w_view,
        s_view,
        M,
        K,
        w_view.stride(0),
        w_view.stride(1),
        w_view.stride(2),
        s_view.stride(0),
        s_view.stride(1),
        s_view.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    # this is an in-place operation, however, we return for simplicity
    return weight, weight_scale

def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k),
                         device=x.device,
                         dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn,
                                                        aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(
        torch.empty((b, aligned_k // 4, aligned_mn),
                    device=x.device,
                    dtype=torch.int), 1, 2)
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def check_sf_layout(sf: torch.Tensor,
                    mn: int,
                    k: int,
                    gran: Tuple[int, int],
                    num_groups: Optional[int],
                    tma_stride_check: bool = False,
                    type_check: Optional[torch.dtype] = None) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(
        k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


def transform_sf_into_required_layout(
        sf: torch.Tensor,
        mn: int,
        k: int,
        recipe: Union[Tuple[int, int, int], Tuple[int, int]],
        num_groups: Optional[int] = None,
        is_sfa: Optional[bool] = None):
    """Transform a scale-factor tensor into the kernel-required layout
    (INT32-packed, per-row, TMA-aligned, MN-major). DeepGEMM-aligned API.

    recipe accepts two forms (mirroring DeepGEMM):
      - 3-tuple (m_gran_a, m_gran_b, k_gran): A/B share the same k_gran.
        Requires `is_sfa` to pick the correct m_gran.
      - 2-tuple (m_gran, k_gran): per-matrix recipe (preferred form when
        A and B have independent k_gran).

    Supported k_gran: {32, 128}. Supported m_gran: {1, 32, 128}
    (2D-block cases are broadcast to per-row via index_select).
    """
    # Resolve (m_gran, k_gran) from recipe.
    if len(recipe) == 3:
        assert is_sfa is not None, (
            "3-tuple recipe requires is_sfa to select A or B granularity.")
        gran = (recipe[0 if is_sfa else 1], recipe[2])
    elif len(recipe) == 2:
        gran = recipe
    else:
        raise ValueError(f'Invalid recipe length: {len(recipe)}')

    gran_mn, gran_k = gran
    assert gran_k in (32, 128), f'Unsupported gran_k={gran_k}'

    # (INT, 1, gran_k): already packed + per-row, just verify layout.
    if sf.dtype == torch.int and gran_mn == 1:
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=gran,
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    # Pre-transform shape check for the FP32 paths (and the unsupported INT+block).
    check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, *, gran_k): broadcast to per-row if needed, then pack to INT.
    if sf.dtype == torch.float:
        if gran_mn != 1:
            sf = sf.index_select(
                -2,
                torch.arange(mn, device=sf.device) // gran_mn)
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, gran_k),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    # INT with gran_mn > 1: needs a transpose/broadcast kernel to produce
    # per-row layout. Not implemented yet; require caller to pass FP32 in
    # this case so the broadcast happens via index_select above.
    raise AssertionError(
        f'Unsupported INT input with gran_mn>1: dtype={sf.dtype}, gran={gran}. '
        f'Pass FP32 scale for block-quantized B, or implement INT broadcast.')


@triton.jit
def _per_token_quant_and_transform_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    token_num_cur_expert,
    size_k,
    fp8_max,
    fp8_min,
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    GROUP_LAYOUT=None,
):
    batch_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_output_scale_0 = tl.cast(stride_output_scale_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)
    stride_output_scale_1 = tl.cast(stride_output_scale_1, dtype=tl.int64)
    stride_input_2 = tl.cast(stride_input_2, dtype=tl.int64)
    stride_output_2 = tl.cast(stride_output_2, dtype=tl.int64)
    stride_output_scale_2 = tl.cast(stride_output_scale_2, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK + tl.arange(0, BLOCK // 4)
    input_ptr_offs = input_ptr + batch_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + batch_id * stride_output_0 + offs_in_d
    output_scale_offs = (output_scale_ptr + batch_id * stride_output_scale_0 +
                         hidden_dim_block_index * stride_output_scale_1)

    if GROUP_LAYOUT is not None:
        token_num_cur_expert = tl.load(GROUP_LAYOUT + batch_id)

    # BLOCK = gran_k * 4 (4 UE8M0 scales packed per int32 along K axis)
    GRAN_K: tl.constexpr = BLOCK // 4
    for token_index in tl.range(token_id,
                                token_num_cur_expert,
                                block_num_per_expert,
                                num_stages=NUM_STAGE):
        output_s_int32 = 0
        for pack_index in tl.range(4):
            local_mask = offs_in_d + pack_index * GRAN_K
            act = tl.load(
                input_ptr_offs + token_index * stride_input_1 +
                pack_index * GRAN_K,
                mask=local_mask < size_k,
                other=0.0,
            ).to(tl.float32)
            _absmax = tl.maximum(tl.max(tl.abs(act)), 1e-10)
            output_s = _absmax / fp8_max
            if SCALE_UE8M0:
                output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
            output_q = tl.clamp(act / output_s, fp8_min,
                                fp8_max).to(output_ptr.dtype.element_ty)
            output_s_int32 += ((output_s.to(tl.int32, bitcast=True) >> 23) <<
                               (8 * pack_index))
            tl.store(
                output_ptr_offs + token_index * stride_output_1 +
                pack_index * GRAN_K,
                output_q,
                mask=local_mask < size_k,
            )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_2,
            output_s_int32,
        )


def per_token_quant_and_transform(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    need_permute102: bool = False,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    """

    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    b = 1
    original_input_rank = len(input.shape)
    if (original_input_rank == 2):
        assert input.is_contiguous()
        input = input.unsqueeze(0)
        b, m, k = input.shape
    elif (original_input_rank == 3):
        if need_permute102:
            input = input.transpose(0, 1)
        b, m, k = input.shape
    else:
        assert False, f"Unsupported input shape rank: {original_input_rank}"


    # Create output
    output = torch.empty((b, m, k), dtype=torch.float8_e4m3fn, device="cuda")

    # Create output scale
    alignment = 4
    scale_k = ceil_div(k, quant_group_size)
    m_padded = align(m, alignment)
    scale_k_padded = align(scale_k, alignment)
    output_scale = torch.zeros((b, scale_k_padded // 4, m_padded),
                               dtype=torch.int32,
                               device='cuda')

    # Get block/grid/stage/warp
    BLOCK_NUM_PER_EXPERT = 64

    BLOCK = quant_group_size * 4
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(k, BLOCK)
    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        b,
    )
    _per_token_quant_and_transform_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        m,
        k,
        fp8_max,
        fp8_min,
        BLOCK=BLOCK,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    if (original_input_rank == 2):
        output = output.squeeze(0)
        output_scale = output_scale.squeeze(0)
        output_scale = output_scale.transpose(0, 1)[:m, :]
    else:
        output_scale = output_scale.transpose(1, 2)[:, :m, :]

    check_sf_layout(
        output_scale,
        m,
        k,
        (1, quant_group_size),
        num_groups= b if original_input_rank == 3 else None,
        tma_stride_check=True,
    )
    return output, output_scale


def grouped_token_quant_and_transform(
    input: torch.Tensor,
    masked_m: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
):
    """
    input shape [g, max_m, k]
    masked_m shape [g]
    output shape [g, max_m, k], dtype fp8
    output_scale [g, ceil_div(k, 512), max_m], dtype int32
    quant_group_size int
    """
    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    g, max_m, k = input.shape

    assert len(input.shape) == 3, f"Unsupported input shape rank: {len(input.shape)}"

    assert masked_m.shape == (g,), f" masked_m shape should be : {g}"


    # Create output
    output = torch.zeros((g, max_m, k), dtype=torch.float8_e4m3fn, device="cuda")

    # Create output scale
    alignment = 4
    scale_k = ceil_div(k, quant_group_size)
    m_padded = align(max_m, alignment)
    scale_k_padded = align(scale_k, alignment)
    output_scale = torch.zeros((g, scale_k_padded // 4, m_padded),
                               dtype=torch.int32,
                               device='cuda')

    # Get block/grid/stage/warp
    BLOCK_NUM_PER_EXPERT = 64

    BLOCK = quant_group_size * 4
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(k, BLOCK)
    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        g,
    )
    _per_token_quant_and_transform_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        max_m,
        k,
        fp8_max,
        fp8_min,
        BLOCK=BLOCK,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
        GROUP_LAYOUT=masked_m,
    )

    output_scale = output_scale.transpose(1, 2)
    check_sf_layout(
        output_scale,
        max_m,
        k,
        (1, quant_group_size),
        num_groups= g,
        tma_stride_check=True,
    )
    return output, output_scale


def per_token_cast_to_mxfp8_for_moe_gemm(
    x: torch.Tensor,
    token_offset: torch.Tensor,
    gran_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert token_offset.dtype == torch.int32
    assert token_offset[0].item() == 0

    token_num, k = x.shape
    E = token_offset.numel() - 1
    PACK_NSF = 4
    PACK_NK = gran_k * PACK_NSF
    m_padded = (token_num + E * 3) // 4 * 4
    k_align = (k + PACK_NK - 1) // PACK_NK

    fp8_output = torch.empty((token_num, k), dtype=torch.float8_e4m3fn, device=x.device)
    sf_int = torch.zeros((k_align, m_padded), dtype=torch.int32, device=x.device)

    for i in range(E):
        start = token_offset[i].item()
        end = token_offset[i + 1].item()
        if start == end:
            continue
        actual_m = end - start
        expert_fp8, expert_sf_fp32 = per_token_cast_to_fp8_e8m0(x[start:end], gran_k=gran_k)

        n_sf = k // gran_k
        n_sf_padded = ((n_sf + PACK_NSF - 1) // PACK_NSF) * PACK_NSF
        if n_sf_padded != n_sf:
            pad = torch.zeros((actual_m, n_sf_padded - n_sf), dtype=torch.float32, device=x.device)
            expert_sf_fp32 = torch.cat([expert_sf_fp32, pad], dim=1)

        packed = ((expert_sf_fp32.view(torch.int32) >> 23) & 0xFF)
        packed = packed.view(actual_m, n_sf_padded // PACK_NSF, PACK_NSF)
        packed_int = (packed[..., 0]
                      | (packed[..., 1] << 8)
                      | (packed[..., 2] << 16)
                      | (packed[..., 3] << 24)).to(torch.int32)

        fp8_output[start:end] = expert_fp8
        def compute_padded_offset(offset: int, expert_idx: int) -> int:
            return (offset + expert_idx * 3) // 4 * 4
        padded_offset = compute_padded_offset(start, i)
        sf_int[:, padded_offset:padded_offset + actual_m] = packed_int.t()

    return fp8_output, sf_int.transpose(0, 1)
