// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/World.h"
#include "dp/Group.h"
#include "dp/Context.h"

namespace dp {

  InstancesDPGroup::InstancesDPGroup(Context *context,
                                     const std::vector<Group *> &groups,
                                     const DPRAffine            *d_transforms)
    : context(context),
      groups(groups),
      d_transforms(d_transforms)
  {}
  
  void InstancesDPGroup::traceRays(DPRRay *d_rays, DPRHit *d_hits, int numRays)
  { /* TODO */ }

} // ::dp

