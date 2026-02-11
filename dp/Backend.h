// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"

namespace dp {
  struct Context;
  struct TrianglesDPGroup;
  struct InstancesDPGroup;
  
  /*! Internal representation of a ray in the input ray
      queue. CAREFUL: this HAS to match the data layout of DPRRay in
      deepee.h !*/
  struct Ray {
    vec3d origin;
    vec3d direction;
    double  tMin;
    double  tMax;
  };

  /*! Internal representation of a hit in the hit queue to be traced
      against. CAREFUL: this HAS to match the data layout of DPRHit in
      deepee.h !*/
  struct Hit {
    /*! index of prim within the geometry it was created in. A value of
      '-1' means 'no hit' */
    int     primID;
    
    /* index of the instance that contained the hit point. Undefined if
       on hit occurred */
    int     instID;
    
    /*! user-supplied geom ID (the one specified during geometry create
      call) for the geometry that contained the hit. Unlike primID and
      instID this is *not* a linear ID, but whatever int64 value the
      user specified there. */
    uint64_t geomUserData;
    double  t;
    double  u, v;
  };


  static Context *createBackend(int gpuID);
} // ::dp
