// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dprt/cuBQL/InstanceGroup.h"
#include "dprt/cuBQL/Triangles.h"

namespace dprt {
  namespace cubql_cuda {

    __dprt_global
    void g_prepareInstances(Kernel kernel,
                            int numInstances,
                            InstanceGroup::InstancedObjectDD *instances,
                            bool hasTransforms,
                            affine3d *worldToObjectXfms,
                            affine3d *objectToWorldXfms,
                            box3d *d_instBounds)
    {
      int tid = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
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
    
    InstanceGroup::
    InstanceGroup(Context *context,
                  const std::vector<dprt::TrianglesGroup *> &groups,
                  const affine3d *transforms)
      : dprt::InstanceGroup(context,groups,
                            (const DPRTAffine *)transforms),
        numInstances((int)groups.size())
    {
      if (groups.size() == 1 && !transforms)
        doesNotActuallyUseInstancing = true;
#if DPRT_OMP
#else
      CUBQL_CUDA_SYNC_CHECK();
#endif
      assert(numInstances > 0);
      std::vector<InstancedObjectDD> instanceDDs;
      for (auto _group : groups) {
        InstancedObjectDD instance;
        TrianglesGroup *group = (TrianglesGroup*)_group;
        instance.group = group->getDD();
        instanceDDs.push_back(instance);
      }

#if DPRT_OMP
      d_instanceDDs = (InstancedObjectDD*)
        omp_target_alloc(numInstances*sizeof(*d_instanceDDs),
                         context->gpuID);
      omp_target_memcpy(d_instanceDDs,
                        instanceDDs.data(),
                        numInstances*sizeof(*d_instanceDDs),
                        0,0,
                        context->gpuID,
                        context->hostID);
      
      d_worldToObjectXfms  = (affine3d*)
        omp_target_alloc(numInstances*sizeof(affine3d),context->gpuID);
      d_objectToWorldXfms  = (affine3d*)
        omp_target_alloc(numInstances*sizeof(affine3d),context->gpuID);

      if (transforms)
        omp_target_memcpy(d_objectToWorldXfms,
                          transforms,
                          numInstances*sizeof(affine3d),
                          0,0,
                          context->gpuID,
                          context->hostID);
      box3d *d_instBounds = 0;
      cudaMalloc((void**)&d_instBounds,
                 numInstances*sizeof(box3d));
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
      for (int i=0;i<numInstances;i++)
        g_prepareInstances
        // <<<divRoundUp(numInstances,128),128>>>
          (Kernel{i},
           numInstances,
         d_instanceDDs,
         transforms != 0,
         d_worldToObjectXfms,
         d_objectToWorldXfms,
         d_instBounds);
#else
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
        (Kernel{},
         numInstances,
         d_instanceDDs,
         transforms != 0,
         d_worldToObjectXfms,
         d_objectToWorldXfms,
         d_instBounds);
      CUBQL_CUDA_SYNC_CHECK();
#endif

      ::cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 1;
#if DPRT_OMP
      std::vector<box3d> h_instBounds(numInstances);
      omp_target_memcpy(h_instBounds.data(),
                        d_instBounds,
                        numInstances*sizeof(*d_instBounds),
                        0,0,context->hostID,context->gpuID);
      bvh3d h_bvh;
      cuBQL::cpu::spatialMedian(h_bvh,
                                h_instBounds.data(),
                                numInstances,
                                buildConfig);
      bvh = h_bvh;
      // --
      bvh.nodes = (bvh3d::Node *)
        omp_target_alloc(bvh.numNodes*sizeof(*bvh.nodes),
                         context->gpuID);
      omp_target_memcpy(bvh.nodes,h_bvh.nodes,
                        bvh.numNodes*sizeof(*bvh.nodes),
                        0,0,
                        context->gpuID,
                        context->hostID);
      // --
      bvh.primIDs = (uint32_t *)
        omp_target_alloc(bvh.numPrims*sizeof(*bvh.primIDs),context->gpuID);
      omp_target_memcpy(bvh.primIDs,h_bvh.primIDs,
                        bvh.numPrims*sizeof(*bvh.primIDs),
                        0,0,
                        context->gpuID,
                        context->hostID);
      cuBQL::cpu::freeBVH(h_bvh);
      omp_target_free(d_instBounds,context->gpuID);
#else
      DeviceMemoryResource memResource;
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
#endif
    }

    InstanceGroup::~InstanceGroup()
    {
#if DPRT_OMP
      omp_target_free(d_instanceDDs,context->gpuID);
      omp_target_free(d_objectToWorldXfms,context->gpuID);
      omp_target_free(d_worldToObjectXfms,context->gpuID);
#else
      cudaFree(d_instanceDDs);
      cudaFree(d_objectToWorldXfms);
      cudaFree(d_worldToObjectXfms);
#endif
    }
    
    InstanceGroup::DD InstanceGroup::getDD() const
    {
      return { d_instanceDDs, d_worldToObjectXfms, bvh };
    }

    __dprt_global
    void g_traceRays_twoLevel(Kernel kernel,
                              /*! the device data for the instancegroup itself */
                              InstanceGroup::DD world,
                              DPRTRay *rays,
                              DPRTHit *hits,
                              size_t numRays,
                              uint64_t flags)
    {
      int tid = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

#ifdef NDEBUG
      const bool dbg = false;
#else
      bool dbg = false;//(tid == 512*1024+512);
#endif

      DPRTHit hit = hits[tid];
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
        if (flags & DPRT_CULL_FRONT)
          culled |= (dot(getNormal(),objectSpace.ray.direction) <= 0.);
        if (flags & DPRT_CULL_BACK)
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

    
    __dprt_global
    void g_traceRays_noInstances(Kernel kernel,
                                 /*! the device data for the instancegroup itself */
                                 InstanceGroup::DD world,
                                 DPRTRay *rays,
                                 DPRTHit *hits,
                                 int numRays,
                                 uint64_t flags)
    {
      int tid = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

#ifdef NDEBUG
      const bool dbg = false;
#else
      bool dbg = false;//(tid == 512*1024+512);
#endif

      DPRTHit hit = hits[tid];
      hit.primID = -1;
      hit.instID = -1;
      hit.t = 1e30;
      
      auto object_instance = world.instancedGroups[0];

      ::cuBQL::ray3d worldRay((const vec3d&)rays[tid].origin,
                              (const vec3d&)rays[tid].direction,
                              rays[tid].tMin,
                              rays[tid].tMax);
      
      auto intersectPrim
        = [&hit,&worldRay,object_instance,flags,dbg](uint32_t primID)
        -> double
      {
        RayTriangleIntersection isec;
        auto &group = object_instance.group;
        PrimRef prim = group.primRefs[primID];
        const TriangleDP tri = group.getTriangle(prim);

        auto getNormal = [tri]() { return cross(tri.b-tri.a,tri.c-tri.a); };
        bool culled = false;
        if (flags & DPRT_CULL_FRONT)
          culled |= (dot(getNormal(),worldRay.direction) <= 0.);
        if (flags & DPRT_CULL_BACK)
          culled |= (dot(getNormal(),worldRay.direction) >= 0.);
        if (!culled && isec.compute(worldRay,tri)) {
          hit.primID = prim.primID;
          hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.instID = 0;//object_instance.instID;
          // hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.t = isec.t;
          worldRay.tMax = isec.t;
        }
        return worldRay.tMax;
      };
      
      ::cuBQL::shrinkingRayQuery::forEachPrim
          (intersectPrim,object_instance.group.bvh,worldRay,dbg);
      
      hits[tid] = hit;
    }

    


    void InstanceGroup::traceRays(DPRTRay *d_rays,
                                  DPRTHit *d_hits,
                                  int numRays,
                                  uint64_t flags)
    {
      if (doesNotActuallyUseInstancing) {
#if DPRT_OMP
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
        for (int i=0;i<numRays;i++)
          g_traceRays_noInstances(Kernel{i},
                                   getDD(),
                                   d_rays,d_hits,numRays,
                                   flags);
#else
        int bs = 128;
        int nb = divRoundUp(numRays,bs);
        g_traceRays_noInstances<<<nb,bs>>>(Kernel(),
                                            getDD(),
                                            d_rays,d_hits,numRays,
                                            flags);
        cudaDeviceSynchronize();
#endif
      } else {
#if DPRT_OMP
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
        for (int i=0;i<numRays;i++)
          g_traceRays_twoLevel(Kernel{i},
                               getDD(),
                               d_rays,d_hits,numRays,
                               flags);
#else
        int bs = 128;
        int nb = divRoundUp(numRays,bs);
        g_traceRays_twoLevel<<<nb,bs>>>(Kernel(),
                                        getDD(),
                                        d_rays,d_hits,numRays,
                                        flags);
        cudaDeviceSynchronize();
#endif
      }
    }      
  }
}

