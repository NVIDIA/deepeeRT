// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Triangles.h"

namespace dp {

  struct Context;

  struct Group {
    struct PrimSpec {
      int primID;
      int geomID;
    };
  };
  
  struct TrianglesDPGroup : public Group {
    TrianglesDPGroup(Context *context,
                     const std::vector<TrianglesDP *> &geoms);
    /*! fill existing arrays of bounding boxes for the BVH builder */
    void fillBuilderInput(box3d *d_boxes);

    PrimSpec        *d_primSpecs = 0;
    TrianglesDP::DD *d_geoms     = 0;
    std::vector<TrianglesDP *> geoms;
  };
    
} // ::dp

