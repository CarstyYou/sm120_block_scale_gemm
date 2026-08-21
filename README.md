# SM120 MXFP8 / FP8 Groupwise GEMM

## Purpose

Optimized GEMM kernels for NVIDIA Blackwell RTX (SM 120/120a). The package
contains MXFP8 groupwise kernels with UE8M0 scale factors and dense FP8
blockwise kernels with float scale factors.

Implementation uses CuTe directly (layout / TMA / QMMA), exposed as PyTorch
custom operators.

## Architecture

```
.
├── 3rdparty/cutlass/               # CUTLASS dependency (submodule)
├── kernels/
│   ├── include/
│   │   └── cute_sm120_mxfp8_groupwise/
│   │       ├── cute_sm120_mxfp8_runner.h
│   │       ├── cute_sm120_fp8_runner.h
│   │       ├── sm120_blockscaled/   # MXFP8 kernel templates and builders
│   │       ├── sm120_blockscaling/  # FP8 float-scale kernel templates
│   │       └── sm120_common/        # Shared scheduler / TMA / epilogue helpers
│   └── src/
│       └── cute_sm120_mxfp8_groupwise/
│           ├── cute_sm120_mxfp8_runner.cu
│           └── cute_sm120_fp8_runner.cu
├── thop/
│   ├── mxfp8GroupwiseGemm.cpp      # MXFP8 PyTorch op bindings
│   └── fp8GroupwiseGemm.cpp        # FP8 PyTorch op bindings
├── test/                           # Customer-facing unit tests
├── CMakeLists.txt
└── build.py                        # cmake + make wrapper
```

## Supported GEMM variants

| Variant | Problem shape | Op |
|---|---|---|
| Dense MXFP8 GEMM | `[M, N, K]` | `gemm_mxfp8_nt_groupwise` |
| Dense FP8 GEMM | `[M, N, K]` | `gemm_fp8_nt_groupwise` |
| Batched MXFP8 GEMM | `[M, N, K, L]` | `batch_gemm_mxfp8_nt_groupwise` |
| Batched FP8 GEMM | `[M, N, K, L]` | `batch_gemm_fp8_nt_groupwise` |
| Masked grouped MXFP8 GEMM | `[max_m, N, K, num_groups]` + `masked_m` | `group_gemm_mxfp8_nt_groupwise_masked` |
| Masked grouped FP8 GEMM | `[max_m, N, K, num_groups]` + `masked_m` | `group_gemm_fp8_nt_groupwise_masked` |
| MoE MXFP8 GEMM | unpadded A + `token_offset[E+1]` | `moe_gemm_mxfp8_nt_groupwise` |
| MoE FP8 GEMM | unpadded A + `token_offset[E+1]` | `moe_gemm_fp8_nt_groupwise` |
| Contiguous MoE MXFP8 GEMM | DeepGEMM-style A + `m_indices` | `group_gemm_mxfp8_nt_groupwise_contiguous` |
| Contiguous MoE FP8 GEMM | DeepGEMM-style A + `m_indices` | `group_gemm_fp8_nt_groupwise_contiguous` |

## Python ops

Bindings registered under `torch.ops.custom_ops`:

| Op | Signature |
|---|---|
| `gemm_mxfp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, granK=128) -> Tensor` |
| `gemm_fp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, granM, granN, granK) -> Tensor` |
| `batch_gemm_mxfp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, out_dtype=None, granK=128) -> Tensor` |
| `batch_gemm_mxfp8_nt_groupwise_out` | `(mat1, mat2, sf1, sf2, out, granK=128) -> Tensor(a!)` |
| `group_gemm_mxfp8_nt_groupwise_masked` | `(mat1, mat2, sf1, sf2, masked_m, granK=128) -> Tensor` |
| `moe_gemm_mxfp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, token_offset, granK=128) -> Tensor` |
| `group_gemm_mxfp8_nt_groupwise_contiguous` | `(mat1, mat2, sf1, sf2, m_indices, granK=128, use_psum_layout=False) -> Tensor` |
| `fp8_quant_and_transform_for_moe` | `(input, token_offset, granK=128) -> (Tensor, Tensor)` |
| `batch_gemm_fp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, granM, granN, granK) -> Tensor` |
| `batch_gemm_fp8_nt_groupwise_out` | `(mat1, mat2, sf1, sf2, out, granM, granN, granK) -> Tensor(a!)` |
| `group_gemm_fp8_nt_groupwise_masked` | `(mat1, mat2, sf1, sf2, masked_m, granM, granN, granK) -> Tensor` |
| `moe_gemm_fp8_nt_groupwise` | `(mat1, mat2, sf1, sf2, token_offset, granM, granN, granK) -> Tensor` |
| `group_gemm_fp8_nt_groupwise_contiguous` | `(mat1, mat2, sf1, sf2, m_indices, granM, granN, granK, use_psum_layout=False) -> Tensor` |

FP8 accepts only `(granM, granN, granK) = (1,128,128)`. SwapAB routes require
contiguous SFA. Non-SwapAB routes require a 16-byte-aligned logical `[Kb,M]`
view with physical M stride `align_up(M,4)`; Batched/Masked batch stride must
cover that padded extent. `test/utils/layout.py` provides
`get_col_major_tma_aligned_tensor`. Zero-padding MoE instead uses contiguous `[Kb,MpE]`,
where `MpE = floor((M + 3E) / 4) * 4`, and expert `i` begins at
`floor((token_offset[i] + 3i) / 4) * 4`. A remains unpadded and SFB contiguous.
For Psum layout on a non-SwapAB route, every cumulative expert boundary in
`m_indices` must be 4-row aligned; this is a caller precondition.

## Quick start

### Requirements

- GPU: NVIDIA Blackwell RTX (SM 120a)
- CUDA: 12.8 or above
- PyTorch: 2.1 or above
- CUTLASS: cloned via Git submodule

### Build

```bash
# From the source root (cutlass submodule must be initialized)
python build.py
```

`build.py` runs cmake + make under `build/`, producing
`libfp8_block_scale_gemm_kernels_static.a` + `libth_op.so`.

### Run unit tests

```bash
python -m pytest -q test/test_fp8.py
python -m pytest -q test/test_mxfp8.py
```

`test/benchmark.py` includes the `bench_kineto` helper used by the shipped
tests. Nsight Compute / external profiler capture scripts are not part of the
release package.

## License

NVIDIA Proprietary Software

## Acknowledgements

Built on NVIDIA CUTLASS and the PyTorch ecosystem.
