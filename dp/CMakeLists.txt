# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(deepeeRT_common INTERFACE#STATIC
)
target_include_directories(deepeeRT_common INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>/include
#  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(deepeeRT_common INTERFACE#PUBLIC
  cuBQL
  CUDA::cuda_driver
  CUDA::cudart_static
)
target_compile_definitions(deepeeRT_common INTERFACE#PUBLIC
  -DdeepeeRT_STATIC=1)

add_library(deepeeRT STATIC
  ../include/deepeeRT/deepeeRT.h
  Backend.h
  Backend.cpp
  common.h
  Triangles.h
  Triangles.cpp
  World.h
  World.cpp
  Group.h
  Group.cpp 
  Context.h
  Context.cpp
  deepeeRT.cpp
)
target_include_directories(deepeeRT PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

add_subdirectory(cuBQL)
target_link_libraries(deepeeRT PRIVATE#PUBLIC
  $<BUILD_INTERFACE:deepeeRT_cuBQL>
  $<BUILD_INTERFACE:deepeeRT_common>
)

set_target_properties(deepeeRT
  PROPERTIES
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
)
