// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprt/Context.h"
#if DPRT_OMP
# include <omp.h>
#endif
#include <cuBQL/bvh.h>
#include <cuBQL/bvh.h>
#include <cuBQL/math/common.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/math/affine.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>
#include "../Triangles.h"
#include "../Instances.h"
#include "owl/owl.h"

namespace dprt {
  namespace owl {

    struct OWLBackend;
    
    using namespace ::cuBQL;
    
    using cuBQL::affine3d;

    struct LaunchParams {
      DPRTRay *rays;
      DPRTHit *hits;
      uint64_t flags;
      OptixTraversableHandle model;
      int numRays;
    };
    
    struct TriangleMesh : public dprt::TriangleMesh {
      struct DD {
        uint64_t userData;
        vec3f   *vertices;
        vec3i   *indices;
      };
      
      TriangleMesh(OWLBackend *be,
                   uint64_t         userData,
                   const vec3d     *vertexArray,
                   int              vertexCount,
                   const vec3i     *indexArray,
                   int              indexCount);
      OWLGeom geom;
    };
    
    struct TrianglesGroup : public dprt::TrianglesGroup {
      TrianglesGroup(OWLBackend *be,
                     const std::vector<dprt::TriangleMesh *> &geoms);
      OWLGroup group;
    };
    
    struct InstanceGroup : public dprt::InstanceGroup {
      InstanceGroup(OWLBackend *be,
                    const std::vector<dprt::TrianglesGroup *> &groups,
                    const DPRTAffine *transforms); 
      /*! implements dprTrace() */
      void traceRays(DPRTRay *d_rays,
                     DPRTHit *d_hits,
                     int numRays,
                     uint64_t flags) override;
      OWLGroup group;
      OWLBackend *const be;
    };
    
    struct OWLBackend : public dprt::Context
    {
      OWLBackend(int gpuID);
      virtual ~OWLBackend();

      dprt::InstanceGroup *
      createInstanceGroup(const std::vector<dprt::TrianglesGroup *> &groups,
                          const DPRTAffine *transforms) override;
    
      dprt::TriangleMesh *
      createTriangleMesh(uint64_t         userData,
                         const vec3d     *vertexArray,
                         int              vertexCount,
                         const vec3i     *indexArray,
                         int              indexCount) override;
      
      dprt::TrianglesGroup *
      createTrianglesGroup(const std::vector<dprt::TriangleMesh *> &geoms) override;

      OWLContext owl;
      OWLGeomType trianglesGT;
      OWLLaunchParams lp;
      OWLRayGen rg;
    };
    
  }
} // ::dprt


  
