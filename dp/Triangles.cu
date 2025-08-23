// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Triangles.h"
#include "dp/Context.h"

namespace dp {
  TrianglesDP::TrianglesDP(Context         *context,
                           uint64_t         userData,
                           const vec3d     *d_vertexArray,
                           int              vertexCount,
                           const vec3i     *d_indexArray,
                           int              indexCount)
    : userData(userData),
      d_vertexArray(d_vertexArray),
      d_indexArray(d_indexArray),
      vertexCount(vertexCount),
      indexCount(indexCount),
      context(context)
  {}
  
} // ::dpr

