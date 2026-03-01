# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set (DEEPEERT_OMP ON)
if (DEEPEERT_OMP)
  #  return()
  set(CMAKE_CUDA_ARCHITECTURES native)
  project(dp_miniapp LANGUAGES C CXX CUDA)
  find_package(deepeeRT REQUIRED)
  #  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

endif()

add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  import_OBJ.cpp
  Camera.h
  Camera.cu
  miniapp.cu
)
target_link_libraries(dpMiniApp PUBLIC
    deepeeRT::deepeeRT
    #deepeeRT
  CUDA::cuda_driver
  CUDA::cudart_static
  )

add_executable(dpInstanceMiniApp
  DGEF.h
  DGEF.cpp
  import_OBJ.cpp
  Camera.h
  Camera.cu
  instanceMiniApp.cu
)
target_link_libraries(dpInstanceMiniApp PUBLIC
  deepeeRT::deepeeRT
  CUDA::cuda_driver
  CUDA::cudart_static
#  deepeeRT
)

if (DEEPEERT_OMP)
  target_include_directories(dpMiniApp PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../submodules/cuBQL)
  target_include_directories(dpInstanceMiniApp PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../submodules/cuBQL)
endif()
