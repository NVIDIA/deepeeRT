// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/InstanceGroup.h"
#include "dp/cuBQL/Triangles.h"

namespace dp {
  namespace cubql_cuda {

    __global__
    void g_prepareInstances(int numInstances,
                            InstanceGroup::InstancedObjectDD *instances,
                            bool hasTransforms,
                            affine3d *worldToObjectXfms,
                            affine3d *objectToWorldXfms,
                            box3d *d_instBounds)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numInstances) return;
      affine3d xfm;
      if (!hasTransforms) {
        xfm = affine3d();
        objectToWorldXfms[tid] = xfm;
      } else {
        xfm = objectToWorldXfms[tid];
      }
      worldToObjectXfms[tid] = rcp(xfm);
      instances[tid].hasXfm = (xfm != affine3d());

      box3d objBounds = instances[tid].group.bvh.nodes[0].bounds;
      vec3d b0 = objBounds.lower;
      vec3d b1 = objBounds.upper;
      box3d instBounds;
      instBounds.extend(xfmPoint(xfm,vec3d(b0.x,b0.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b0.x,b0.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b0.x,b1.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b0.x,b1.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b1.x,b0.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b1.x,b0.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b1.x,b1.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,vec3d(b1.x,b1.y,b1.z)));
      d_instBounds[tid] = instBounds;
    }
    
    InstanceGroup::InstanceGroup(Context *context,
                                 const std::vector<dp::TrianglesGroup *> &groups,
                                 const affine3d *transforms)
      : dp::InstanceGroup(context,groups,
                          (const DPRAffine *)transforms),
        numInstances((int)groups.size())
    {
      CUBQL_CUDA_SYNC_CHECK();
      assert(numInstances > 0);
      std::vector<InstancedObjectDD> instanceDDs;
      for (auto _group : groups) {
        InstancedObjectDD instance;
        TrianglesGroup *group = (TrianglesGroup*)_group;
        instance.group = group->getDD();
        instanceDDs.push_back(instance);
      }

      cudaMalloc((void**)&d_instanceDDs,
                 numInstances*sizeof(*d_instanceDDs));
      cudaMemcpy(d_instanceDDs,
                 instanceDDs.data(),
                 numInstances*sizeof(*d_instanceDDs),
                 cudaMemcpyDefault);
      
      cudaMalloc((void**)&d_worldToObjectXfms,
                 numInstances*sizeof(affine3d));
      cudaMalloc((void**)&d_objectToWorldXfms,
                 numInstances*sizeof(affine3d));
      if (transforms)
        cudaMemcpy(d_objectToWorldXfms,
                   transforms,
                   numInstances*sizeof(affine3d),
                   cudaMemcpyDefault);
      box3d *d_instBounds = 0;
      cudaMalloc((void**)&d_instBounds,
                 numInstances*sizeof(box3d));
      g_prepareInstances
        <<<divRoundUp(numInstances,128),128>>>
        (numInstances,
         d_instanceDDs,
         transforms != 0,
         d_worldToObjectXfms,
         d_objectToWorldXfms,
         d_instBounds);
      CUBQL_CUDA_SYNC_CHECK();

      DeviceMemoryResource memResource;
      ::cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 1;
      std::cout << "==================================================================" << std::endl;
      PING;
      std::cout << "TOP" << std::endl;
      ::cuBQL::cuda::sahBuilder(bvh,
                                d_instBounds,
                                numInstances,
                                buildConfig,
                                0,
                                memResource);
      
      CUBQL_CUDA_SYNC_CHECK();
      PING;
      cudaFree(d_instBounds);
    }

    InstanceGroup::~InstanceGroup()
    {
      cudaFree(d_instanceDDs);
      cudaFree(d_objectToWorldXfms);
      cudaFree(d_worldToObjectXfms);
    }
    
    InstanceGroup::DD InstanceGroup::getDD() const
    {
      return { d_instanceDDs, d_worldToObjectXfms, bvh };
    }

    __global__
    void g_traceRays(/*! the device data for the instancegroup itself */
                     InstanceGroup::DD world,
                     DPRRay *rays,
                     DPRHit *hits,
                     int numRays,
                     uint64_t flags)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

#ifdef NDEBUG
      const bool dbg = false;
#else
      bool dbg = false;//(tid == 512*1024+512);
#endif

      DPRHit hit = hits[tid];
      hit.primID = -1;
      hit.instID = -1;
      hit.t = 1e30;
      struct ObjectSpaceTravState {
        int instID = -1;
        InstanceGroup::InstancedObjectDD instance;
        ::cuBQL::ray3d ray;
      } objectSpace;
      ::cuBQL::ray3d worldRay((const vec3d&)rays[tid].origin,
                              (const vec3d&)rays[tid].direction,
                              rays[tid].tMin,
                              rays[tid].tMax);
      
      auto intersectPrim
        = [&hit,&worldRay,&objectSpace,flags,dbg](uint32_t primID)
        -> double
      {
        RayTriangleIntersection isec;
        auto &group = objectSpace.instance.group;
        PrimRef prim = group.primRefs[primID];
        const TriangleDP tri = group.getTriangle(prim);

        auto getNormal = [tri]() { return cross(tri.b-tri.a,tri.c-tri.a); };
        bool culled = false;
        if (flags & DPR_CULL_FRONT)
          culled |= (dot(getNormal(),objectSpace.ray.direction) <= 0.);
        if (flags & DPR_CULL_BACK)
          culled |= (dot(getNormal(),objectSpace.ray.direction) >= 0.);
        if (!culled && isec.compute(objectSpace.ray,tri)) {
          hit.primID = prim.primID;
          hit.instID = objectSpace.instID;
          hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.t = isec.t;
          worldRay.tMax = isec.t;
        }
        return worldRay.tMax;
      };
      auto enterBlas = [world,worldRay,&objectSpace,dbg]
        (cuBQL::ray3d &out_ray,
         cuBQL::bvh3d &out_bvh,
         int instID) 
      {
        objectSpace.instance = world.instancedGroups[instID];
        objectSpace.instID = instID;
        objectSpace.ray = worldRay;
        if (objectSpace.instance.hasXfm) {
          affine3d worldToObjectXfm = world.worldToObjectXfms[instID];
          objectSpace.ray.origin
            = xfmPoint(worldToObjectXfm,worldRay.origin);
          objectSpace.ray.direction
            = xfmVector(worldToObjectXfm,worldRay.direction);
        }
        out_ray = objectSpace.ray;
        if (dbg) dout << "out ray " << out_ray << "\n";
        out_bvh = objectSpace.instance.group.bvh;
        // out_bvh.nodes = objectSpace.instance.group.bvh.nodes;
      };
      auto leaveBlas = []() -> void {
        /* nothing to do */
      };
      
      ::cuBQL::shrinkingRayQuery::twoLevel::forEachPrim
          (enterBlas,leaveBlas,intersectPrim,world.bvh,worldRay,dbg);
      
      hits[tid] = hit;
    }

    
    void InstanceGroup::traceRays(DPRRay *d_rays,
                                  DPRHit *d_hits,
                                  int numRays,
                                  uint64_t flags)
    {
      int bs = 128;
      int nb = divRoundUp(numRays,bs);
      g_traceRays<<<nb,bs>>>(getDD(),
                             d_rays,d_hits,numRays,
                             flags);
      cudaDeviceSynchronize();
    }
      
  }
}

