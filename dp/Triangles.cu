// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Triangles.h"
#include "dp/Context.h"

namespace dp {
  TrianglesDP::TrianglesDP(Context         *context,
                           uint64_t         userData,
                           const vec3d     *_vertexArray,
                           int              vertexCount,
                           const vec3i     *_indexArray,
                           int              indexCount)
    : userData(userData),
      vertexCount(vertexCount),
      indexCount(indexCount),
      context(context)
  {
    cudaMalloc((void**)&vertexArray,vertexCount*sizeof(vec3d));
    cudaMemcpy((void*)vertexArray,_vertexArray,
               vertexCount*sizeof(vec3d),cudaMemcpyDefault);
    cudaMalloc((void**)&indexArray,indexCount*sizeof(vec3d));
    cudaMemcpy((void*)indexArray,_indexArray,
               indexCount*sizeof(vec3d),cudaMemcpyDefault);
  }

  TrianglesDP::~TrianglesDP()
  {
    cudaFree((void*)indexArray);
    cudaFree((void*)vertexArray);
  }
  
} // ::dp

