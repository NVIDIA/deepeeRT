// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Group.h"

namespace dp {

  struct Context;
  struct TrianglesGroup;

  /*! a group of double precision instances; each instance is defined
      by a affine transforms and TrianglesDPGroup that it refers to */
  struct InstanceGroup : public Group {
    InstanceGroup(Context *context,
                  const std::vector<TrianglesGroup *> &groups,
                  const DPRAffine            *transforms)
      : Group(context)
    {}

    /*! implements dprTrace() */
    virtual void traceRays(DPRRay *d_rays,
                           DPRHit *d_hits,
                           int numRays,
                           uint64_t flags) = 0;

    /* iw - note this base class will NOT store any pointers to host
       data, it's the job of the derived class(es) to store data as,
       if, and where required*/
  };
    
} // ::dp

