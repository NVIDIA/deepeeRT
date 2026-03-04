
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(dprt_cuBQL OBJECT
  OWLBackend.h
  OWLBackend.cpp
)
target_link_libraries(dprt_cuBQL
  PUBLIC
  dprt_common
  cuBQL
  owl::owl
)

if (DPRT_OMP)
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
  target_compile_definitions(dprt_cuBQL PUBLIC -DDPRT_OMP=1)
  target_link_options(dprt_cuBQL PRIVATE "-fopenmp;-mp=gpu")
endif()

target_link_libraries(dprt PRIVATE
  $<BUILD_INTERFACE:dprt_cuBQL>
)
