// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Triangles.h"
#include "dp/Context.h"

namespace dp {
  TriangleMesh::TriangleMesh(Context         *context,
                             uint64_t         userData,
                             const vec3d     *_vertexArray,
                             int              vertexCount,
                             const vec3i     *_indexArray,
                             int              indexCount)
    : userData(userData),
      context(context)
  {
    /* iw - note this class will NOT store any pointers to host data,
       it's the job of the derived class(es) to sture data as, if, and
       where required*/
  }
  
  TrianglesGroup::TrianglesGroup(Context *context,
                                 const std::vector<TriangleMesh *> &geoms)
    : Group(context)
  {
    /* iw - note this class will NOT store any pointers to host data,
       it's the job of the derived class(es) to sture data as, if, and
       where required*/
  }
  
} // ::dp

