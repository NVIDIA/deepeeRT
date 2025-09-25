// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Triangles.h"

namespace dp {

  struct Context;
  struct Group;

  struct InstancesDPImpl;
  
  struct InstancesDPGroup {
    InstancesDPGroup(Context *context,
                     const std::vector<Group *> &groups,
                     const DPRAffine            *d_transforms);
    
    void traceRays(DPRRay *d_rays, DPRHit *d_hits, int numRays);
    
    std::vector<Group *> const groups;
    const DPRAffine     *const d_transforms;
    Context             *const context;
    std::shared_ptr<InstancesDPImpl> impl;
  };
    
} // ::dp

