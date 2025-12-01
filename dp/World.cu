// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  {
    impl = context->backend->createInstancesDPImpl(this);
  }
  
  void InstancesDPGroup::traceRays(DPRRay *d_rays, DPRHit *d_hits, int numRays)
  {
    CUBQL_CUDA_SYNC_CHECK();
    if (!isDevicePointer(d_rays) ||
        !isDevicePointer(d_hits))
      throw std::runtime_error("the rays[] and hits[] arrays passed to dpTrace (currently?) have to point to device memory.");
    impl->trace((Ray*)d_rays,
                (Hit*)d_hits,
                numRays);
  }

} // ::dp

