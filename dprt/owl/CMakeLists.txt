
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

option(EXP_DOUBLE_DISTANCE "Experimental owl-plus-double-distance test" ON)

if (EXP_DOUBLE_DISTANCE)
  target_compile_definitions(dprt_common INTERFACE -DEXP_DOUBLE_DISTANCE=1)
endif()

embed_ptx(
  OUTPUT_TARGET
  devCode-ptx
  
  PTX_LINK_LIBRARIES
  #  owl_prime-interface
  dprt_common
  owl::owl
  
  SOURCES
  devCode.cu
  )


add_library(dprt_owl OBJECT
  OWLBackend.h
  OWLBackend.cpp
)
target_link_libraries(dprt_owl
  PUBLIC
  dprt_common
  cuBQL
  owl::owl
  devCode-ptx
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
  target_compile_definitions(dprt_owl PUBLIC -DDPRT_OMP=1)
  target_link_options(dprt_owl PRIVATE "-fopenmp;-mp=gpu")
endif()

target_link_libraries(dprt PRIVATE
  $<BUILD_INTERFACE:dprt_owl>
)
