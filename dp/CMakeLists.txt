# Copyright 2025 NVIDIA Corp
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(deepeeRT_common STATIC
  Backend.h
  Backend.cpp
)
target_include_directories(deepeeRT_common PUBLIC
  ${PROJECT_SOURCE_DIR}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(deepeeRT_common PUBLIC
  cuBQL
  CUDA::cuda_driver
  CUDA::cudart_static
)

add_library(deepeeRT STATIC
  common.h
  Triangles.h
  Triangles.cu
  World.h
  World.cu
  Context.h
  Context.cpp
  Group.h
  Group.cu
  deepeeRT.cpp
)

add_subdirectory(cuBQL)
target_link_libraries(deepeeRT PUBLIC
  deepeeRT_common
  deepeeRT_cuBQL
)
