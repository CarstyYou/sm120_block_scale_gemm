# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch


def _register_fake():
    @torch.library.register_fake("custom_ops::gemm_mxfp8_nt_groupwise")
    def _(a, b, a_scale, b_scale, bias):
        m = a.shape[0]
        n = b.shape[0]
        return a.new_empty((m, n))


def _init_op() -> None:
    project_dir = str(Path(__file__).parent.parent.absolute())
    ft_decoder_lib = project_dir + "/build/thop/libth_op.so"

    try:
        torch.classes.load_library(ft_decoder_lib)
        _register_fake()
    except Exception as e:
        msg = (
            "\nFATAL: Decoding operators failed to load. This may be caused by "
            "the incompatibility between PyTorch and TensorRT-LLM. Please rebuild "
            "and install TensorRT-LLM."
        )
        raise ImportError(str(e) + msg)
