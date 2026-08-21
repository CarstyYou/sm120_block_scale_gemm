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

cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(PROJECT_NAME sm120_fp8_block_scale_gemm)

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)




if(NOT WIN32 AND NOT DEFINED USE_CXX11_ABI)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)
  # Convert the bool variable to integer.
  if(USE_CXX11_ABI)
    set(USE_CXX11_ABI 1)
  else()
    set(USE_CXX11_ABI 0)
  endif()
endif()




# Check CUDA version (requires CUDA 12.8 or higher for SM120 support)
if(CUDA_VERSION VERSION_LESS "12.8")
  message(FATAL_ERROR "CUDA 12.8 or higher is required for SM120 support")
endif()

# Set CUDA architecture flag for NVIDIA Blackwell RTX (SM120a)
set(CMAKE_CUDA_ARCHITECTURES 120a-real)

# Auto-detect host GPU SM count for L2 swizzle hint (scheduler.cuh kDefaultNumSMs).
# Compile-run a small CUDA runtime detector at configure time. Override via
# -DSM120_DEFAULT_NUM_SMS=N. Falls back to 188 on detect failure.
if(NOT DEFINED SM120_DEFAULT_NUM_SMS)
  set(_DETECTOR_SRC "${CMAKE_BINARY_DIR}/detect_sm.cu")
  set(_DETECTOR_BIN "${CMAKE_BINARY_DIR}/detect_sm")
  file(WRITE ${_DETECTOR_SRC} "#include <cuda_runtime.h>
#include <cstdio>
int main() {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
  printf(\"%d\", prop.multiProcessorCount);
  return 0;
}
")
  execute_process(
    COMMAND nvcc -arch=sm_120a -o ${_DETECTOR_BIN} ${_DETECTOR_SRC}
    RESULT_VARIABLE _COMPILE_RES
    OUTPUT_QUIET ERROR_QUIET
    TIMEOUT 30)
  if(_COMPILE_RES EQUAL 0)
    execute_process(
      COMMAND ${_DETECTOR_BIN}
      OUTPUT_VARIABLE _DETECTED_SM_COUNT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _RUN_RES
      ERROR_QUIET
      TIMEOUT 5)
    if(_RUN_RES EQUAL 0 AND _DETECTED_SM_COUNT MATCHES "^[0-9]+$")
      set(SM120_DEFAULT_NUM_SMS ${_DETECTED_SM_COUNT})
    endif()
  endif()
  if(NOT DEFINED SM120_DEFAULT_NUM_SMS)
    set(SM120_DEFAULT_NUM_SMS 188)
  endif()
endif()
add_compile_definitions(SM120_DEFAULT_NUM_SMS=${SM120_DEFAULT_NUM_SMS})

# Set CUDA flags
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills"
)# -Werror -v
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -O3 -Xptxas='${CUDA_PTXAS_FLAGS}'"
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_120a,code=sm_120a")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_ARCH_FEAT_SM120_ALL")


function(parse_make_options options result)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${result}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()

option(BUILD_PYT "Build in PyTorch TorchScript class mode" ON)
if(BUILD_PYT)
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    message(
      WARNING
        "Ignoring environment variable TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}"
    )
  endif()
  unset(ENV{TORCH_CUDA_ARCH_LIST})
  set(TORCH_CUDA_ARCH_LIST "8.0")

  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  link_directories("${Python3_LIBRARY_DIRS}")
  list(APPEND COMMON_HEADER_DIRS ${Python3_INCLUDE_DIRS})

  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} "-c"
      "from __future__ import print_function; import torch; print(torch.__version__,end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_VERSION)
  if(TORCH_VERSION VERSION_LESS "1.5.0")
    message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR)
  if(NOT _PYTHON_SUCCESS EQUAL 0)
    message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  set(USE_SYSTEM_NVTX ON)
  set(nvtx3_dir ${3RDPARTY_DIR}/NVTX/include)
  set(CMAKE_CUDA_ARCHITECTURES_BACKUP ${CMAKE_CUDA_ARCHITECTURES})
  find_package(Torch REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_BACKUP})
  string(REPLACE "-gencode arch=compute_80,code=sm_80 " "" CMAKE_CUDA_FLAGS_NEW
                 "${CMAKE_CUDA_FLAGS}")
  if("${CMAKE_CUDA_FLAGS_NEW}" STREQUAL "${CMAKE_CUDA_FLAGS}")
    message(
      FATAL_ERROR
        "Torch didn't generate expected placeholder CUDA architecture flags.")
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_NEW}")


  add_compile_definitions(TORCH_CUDA=1)

  if(DEFINED TORCH_CXX_FLAGS)
    add_compile_options(${TORCH_CXX_FLAGS})
    if(DEFINED USE_CXX11_ABI)
      parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
      if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
         AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL
             ${USE_CXX11_ABI})
        message(
          WARNING
            "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
            "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}, and the project "
            "setting will be discarded.")
      endif()
    endif()
  endif()

else()
  if(NOT WIN32)
    if(NOT USE_CXX11_ABI)
      add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
  endif()
endif()

set(3RDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

include_directories(
    SYSTEM
    ${TORCH_INCLUDE_DIRS}
    ${3RDPARTY_DIR}/cutlass/include
    ${3RDPARTY_DIR}/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/
    ${CMAKE_CURRENT_SOURCE_DIR}
	)


set(BLOCK_SCALE_KERNELS_TARGET_NAME fp8_block_scale_gemm_kernels)
set(BLOCK_SCALE_KERNELS_STATIC_TARGET
    ${BLOCK_SCALE_KERNELS_TARGET_NAME}_static)
set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

add_subdirectory(kernels)

add_subdirectory(thop)
