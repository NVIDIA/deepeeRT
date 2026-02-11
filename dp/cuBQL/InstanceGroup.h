// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/cuBQL/Triangles.h"
#include "dp/World.h"

namespace dp {
  namespace cubql_cuda {

    /*! a single triangle mesh; can be created over pointes that are
        either on host or device, but which definitively stores
        vertices on the device */
    struct InstanceGroup : public dp::InstanceGroup {
      struct InstancedObjectDD {
        TrianglesGroup::DD group;
        bool hasXfm;
      };
      struct DD {
        const InstancedObjectDD *instancedGroups;
        const affine3d          *worldToObjectXfms;
        bvh3d bvh;
      };

      InstanceGroup(Context *context,
                    const std::vector<dp::TrianglesGroup *> &groups,
                    const affine3d *transforms);
      ~InstanceGroup() override;
      
      DD getDD() const;

      void traceRays(DPRRay *d_rays,
                     DPRHit *d_hits,
                     int numRays,
                     uint64_t flags) override;

      int                numInstances = 0;
      InstancedObjectDD *d_instanceDDs = 0;
      affine3d          *d_worldToObjectXfms = 0;
      affine3d          *d_objectToWorldXfms = 0;
      bvh3d bvh;
    };
    
  }
}
