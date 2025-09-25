// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Triangles.h"

namespace dp {

  struct Context;
  struct TrianglesDPImpl;

  /*! allows for referencing a specific primitive within a specific
      geometry within multiple geometries that a group may be built
      over */
  struct PrimRef {
    int geomID;
    int primID;
  };
  
  struct Group {
  };
  
  struct TrianglesDPGroup : public Group {
    TrianglesDPGroup(Context *context,
                     const std::vector<TrianglesDP *> &geoms);

    std::vector<TrianglesDP *> geoms;
    Context *const context;
    std::shared_ptr<TrianglesDPImpl> impl;
  };

} // ::dp

