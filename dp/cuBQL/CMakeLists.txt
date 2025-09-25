# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(deepeeRT_cuBQL STATIC
  CuBQLBackend.h
  CuBQLBackend.cu
)
target_link_libraries(deepeeRT_cuBQL PUBLIC
  deepeeRT_common
  cuBQL
#  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
  $<BUILD_INTERFACE:cuBQL_cuda_double3_static>
)

