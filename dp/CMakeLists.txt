# Copyright 2025 NVIDIA Corp
# SPDX-License-Identifier: Apache-2.0

add_library(deepeeRT
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
find_package(CUDAToolkit REQUIRED)

target_include_directories(deepeeRT PUBLIC
  ${PROJECT_SOURCE_DIR}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(deepeeRT PUBLIC
  cuBQL
  #CUDA::cuda_driver
  CUDA::cudart_static
)
