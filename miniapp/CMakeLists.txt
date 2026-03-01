# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#set (DEEPEERT_OMP ON)
#if (DEEPEERT_OMP)

project(dp_miniapp LANGUAGES C CXX)

if (NOT DEEPEERT_OMP)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (NOT (TARGET deepeeRT)) 
    #  return()
    find_package(deepeeRT REQUIRED)
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

  #  if (DEEPEERT_OMP)
  target_include_directories(dpMiniApp PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../submodules/cuBQL)
  target_include_directories(dpInstanceMiniApp PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../submodules/cuBQL)
  #endif()
endif()

if (DEEPEERT_OMP)
  macro(dp_configure_source src)
    set_source_files_properties(${src} PROPERTIES
      LANGUAGE CXX
    )
    set_source_files_properties(${src} PROPERTIES
      COMPILE_OPTIONS
      "-fopenmp;-gpu=ccnative;-mp=gpu" 
    )
  endmacro()
  dp_configure_source(miniapp.cu)
  dp_configure_source(Camera.cu)
  dp_configure_source(instanceMiniApp.cu)
  target_compile_definitions(dpMiniApp PUBLIC -DDP_OMP=1)
  target_link_options(dpMiniApp PRIVATE "-fopenmp;-mp=gpu")
endif()
endif()
