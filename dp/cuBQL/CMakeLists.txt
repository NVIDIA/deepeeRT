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
)

#if (DEEPEERT_OMP)
#  target_link_libraries(deepeeRT_cuBQL PUBLIC
#    $<BUILD_INTERFACE:cuBQL_cpu_double3_static>
#  )
#else()
#  target_link_libraries(deepeeRT_cuBQL PUBLIC
#    $<BUILD_INTERFACE:cuBQL_cuda_double3_static>
#  )
#endif()

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
  dp_configure_source(CuBQLBackend.cu)
  dp_configure_source(Triangles.cu)
  dp_configure_source(InstanceGroup.cu)
  target_compile_definitions(deepeeRT_cuBQL PUBLIC -DDP_OMP=1)
  target_link_options(deepeeRT_cuBQL PRIVATE "-fopenmp;-mp=gpu")
endif()
