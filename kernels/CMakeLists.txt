# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

set(INTERNAL_SRC_CU
    src/cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.cu
    src/cute_sm120_mxfp8_groupwise/cute_sm120_fp8_runner.cu)

add_library(${BLOCK_SCALE_KERNELS_STATIC_TARGET} STATIC
            ${INTERNAL_SRC_CU})

target_include_directories(
  ${BLOCK_SCALE_KERNELS_STATIC_TARGET} BEFORE
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${3RDPARTY_DIR}/cutlass/include
  PRIVATE ${3RDPARTY_DIR}/cutlass/examples)

set_target_properties(
  ${BLOCK_SCALE_KERNELS_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${BLOCK_SCALE_KERNELS_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
# changed in GCC 4.6 This note appears for kernels using TMA and clutters the
# compilation output.
if(NOT WIN32)
  target_compile_options(
    ${BLOCK_SCALE_KERNELS_STATIC_TARGET}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi -Xcicc --uumn
            -Xptxas -uumn >)
endif()
