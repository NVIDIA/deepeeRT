// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"

namespace dp {

  struct Context;

  /*! a mesh of triangles, for a dp context, with vertices in doubles */
  struct TrianglesDP {
    struct DD {
      uint64_t userData;
      vec3d   *d_vertexArray;
      vec3i   *d_indexArray;
      int      vertexCount;
      int      indexCount;
    };
      
    TrianglesDP(Context         *context,
                uint64_t         userData,
                const vec3d     *d_vertexArray,
                int              vertexCount,
                const vec3i     *d_indexArray,
                int              indexCount);

    void fillBuilderInput(box3d *d_boxes, cudaStream_t stream);
    DD getDD() const;
     
    uint64_t     const userData      = 0;
    const vec3d *const d_vertexArray = 0;
    const vec3i *const d_indexArray  = 0;
    int          const vertexCount   = 0;
    int          const indexCount    = 0;
    Context     *const context;
  };

} // ::dp


