// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Triangles.h"
#include "dp/Context.h"

namespace dp {
  TrianglesDP::TrianglesDP(Context         *context,
                           uint64_t         userData,
                           const vec3d     *vertexArray,
                           int              vertexCount,
                           const vec3i     *indexArray,
                           int              indexCount)
    : userData(userData),
      vertexArray(vertexArray),
      indexArray(indexArray),
      vertexCount(vertexCount),
      indexCount(indexCount),
      context(context)
  {}
  
} // ::dp

