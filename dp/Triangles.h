// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"

namespace dp {

  struct Context;

  /*! a mesh of triangles, for a dp context, with vertices in doubles */
  struct TrianglesDP {
    TrianglesDP(Context         *context,
                uint64_t         userData,
                const vec3d     *vertexArray,
                int              vertexCount,
                const vec3i     *indexArray,
                int              indexCount);
     
    uint64_t     const userData      = 0;
    const vec3d *const vertexArray   = 0;
    const vec3i *const indexArray    = 0;
    int          const vertexCount   = 0;
    int          const indexCount    = 0;
    Context     *const context;
  };

} // ::dp


