// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Group.h"

namespace dp {

  struct Context;

  /*! (virtual base class for) a mesh of triangles, for a dp context,
      with vertices in doubles. implementations of this class may
      create copies of the input arays on either host and/or device */
  struct TriangleMesh {
    TriangleMesh(Context         *context,
                 uint64_t         userData,
                 const vec3d     *vertexArray,
                 int              vertexCount,
                 const vec3i     *indexArray,
                 int              indexCount);
    virtual ~TriangleMesh() = default;

    /* iw - note this base class will NOT store any pointers to host
       data, it's the job of the derived class(es) to sture data as,
       if, and where required*/
    uint64_t     const userData      = 0;
    Context     *const context;
  };

  /*! allows for referencing a specific primitive within a specific
      geometry within multiple geometries that a group may be built
      over */
  struct PrimRef {
    int geomID;
    int primID;
  };

  /*! a "group" of one or more triangle meshes, including the
      acceleration structure to trace a ray against those triangles */
  struct TrianglesGroup : public Group {
    TrianglesGroup(Context *context,
                   const std::vector<TriangleMesh *> &geoms);
    
    /* iw - note this base class will NOT store any pointers to host
       data, it's the job of the derived class(es) to sture data as,
       if, and where required*/
  };

} // ::dp


