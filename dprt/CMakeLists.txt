# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(dprt_common INTERFACE
)
target_include_directories(dprt_common INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>/include
)
target_link_libraries(dprt_common INTERFACE
  cuBQL
  CUDA::cuda_driver
  CUDA::cudart_static
)
target_compile_definitions(dprt_common INTERFACE
  -Ddprt_STATIC=1)

add_library(dprt STATIC
  ../include/dprt/dprt.h
  Backend.h
  Backend.cpp
  common.h
  Triangles.h
  Triangles.cpp
  Instances.h
  Group.h
  Context.h
  Context.cpp
  dprt.cpp
)
target_include_directories(dprt PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )

add_subdirectory(cuBQL)
target_link_libraries(dprt PRIVATE
  $<BUILD_INTERFACE:dprt_cuBQL>
  $<BUILD_INTERFACE:dprt_common>
)

set_target_properties(dprt
  PROPERTIES
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
)

add_library(dprt::dprt ALIAS dprt)

