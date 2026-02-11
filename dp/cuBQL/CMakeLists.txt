# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(deepeeRT_cuBQL OBJECT
  CuBQLBackend.h
  CuBQLBackend.cu 
  Triangles.h
  Triangles.cu
  InstanceGroup.h
  InstanceGroup.cu
)
target_link_libraries(deepeeRT_cuBQL
  PUBLIC
  deepeeRT_common
  cuBQL
  # do not link to cuBQL_cuda_double3_static; we explicitly
  # instantiate builder(s) in CubQLBackend.cu
)

