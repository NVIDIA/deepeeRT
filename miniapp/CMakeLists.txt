# Copyright 2025 NVIDIA Corp
# SPDX-License-Identifier: Apache-2.0

add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  Camera.h
  Camera.cu
  miniapp.cu
)
target_link_libraries(dpMiniApp PUBLIC deepeeRT)

