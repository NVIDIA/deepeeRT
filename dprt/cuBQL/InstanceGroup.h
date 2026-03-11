// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprt/cuBQL/CuBQLBackend.h"
#include "dprt/cuBQL/Triangles.h"
#include "dprt/Instances.h"

namespace dprt {
  namespace cubql_cuda {

    /*! a single triangle mesh; can be created over pointes that are
        either on host or device, but which definitively stores
        vertices on the device */
    struct InstanceGroup : public dprt::InstanceGroup {
      struct InstancedObjectDD {
        TrianglesGroup::DD group;
        bool hasXfm;
      };
      struct DD {
        const InstancedObjectDD *instancedGroups;
        const impl_affine_t     *worldToObjectXfms;
        const bool              *hasActualTransform;
        impl_bvh_t               bvh;
      };

      InstanceGroup(Context *context,
                    const std::vector<dprt::TrianglesGroup *> &groups,
                    // input is ALWAYS double
                    const affine3d *transforms);
      ~InstanceGroup() override;
      
      DD getDD() const;

      void traceRays(DPRTRay *d_rays,
                     DPRTHit *d_hits,
                     int numRays,
                     uint64_t flags) override;

      /*! if this scene contains a single instance, and that has a
          unit transform, then it can be traced with a single-level,
          no instancing kernel */
      bool               hasAnyActualTransform = false;
      int                numInstances = 0;
      InstancedObjectDD *d_instanceDDs = 0;
      impl_affine_t     *d_worldToObjectXfms = 0;
      // TODO - check if we even need those?
      impl_affine_t     *d_objectToWorldXfms = 0;
      /*! one entry per instance; allows traversal to skip unnecessary
          transforms */
      bool              *d_hasActualTransform = 0;
      /*! one single entry; specifies if there is _any_ non-trivial
          transform; allows to use a single-level transform if this is
          not the case */
      bool              *d_hasAnyActualTransform = 0;
      impl_bvh_t bvh;
    };
    
  }
} // ::dprt
