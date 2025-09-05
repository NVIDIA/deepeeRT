// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"

namespace dp {
  struct Context;
  struct TrianglesDPGroup;
  struct World;
  
  /*! CAREFUL: this HAS to match the data layout of DPRRay in deepee.h !*/
  struct Ray {
    vec3d origin;
    vec3d direction;
    double  tMin;
    double  tMax;
  };

  /*! CAREFUL: this HAS to match the data layout of DPRHit in deepee.h !*/
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
  
  struct TrianglesDPImpl {
    TrianglesDPImpl(TrianglesDPGroup *const fe) : fe(fe) {}
    virtual ~TrianglesDPImpl() = default;
    TrianglesDPGroup *const fe;
  };
    
  struct WorldImpl {
    WorldImpl(World *const fe) : fe(fe) {}
    virtual ~WorldImpl() = default;
    World *const fe;
  };
  
  struct Backend {
    Backend(Context *const context);
    virtual ~Backend() = default;
    
    virtual std::shared_ptr<WorldImpl>
    createWorldDPImpl(dp::World *fe) = 0;
    
    virtual std::shared_ptr<TrianglesDPImpl>
    createTrianglesDPImpl(dp::TrianglesDPGroup *fe) = 0;
    
    Context *const context;
    int const gpuID;
  };
  
}
