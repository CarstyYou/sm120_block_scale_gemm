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

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

message(STATUS "KERNELS_DIR: ${KERNELS_DIR}")
message(STATUS "BLOCK_SCALE_KERNELS_STATIC_TARGET: ${BLOCK_SCALE_KERNELS_STATIC_TARGET}")

add_library(
  th_op SHARED
  mxfp8GroupwiseGemm.cpp
  fp8GroupwiseGemm.cpp
  bf16Gemm.cpp)
set_property(TARGET th_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(th_op PUBLIC ${TORCH_LIBRARIES}
                                        ${Python3_LIBRARIES} ${BLOCK_SCALE_KERNELS_STATIC_TARGET})

target_include_directories(th_op PUBLIC
    ${KERNELS_DIR}/include/
    ${TORCH_INCLUDE_DIRS}
)

if(NOT WIN32)
  set_target_properties(
    th_op
    PROPERTIES LINK_FLAGS
               "-Wl,-rpath='$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()
