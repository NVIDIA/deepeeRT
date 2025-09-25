# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(dpMiniApp
  Mesh.h
  Mesh.cpp
  Camera.h
  Camera.cu
  miniapp.cu
)
target_link_libraries(dpMiniApp PUBLIC deepeeRT)

