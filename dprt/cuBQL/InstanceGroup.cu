// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dprt/cuBQL/InstanceGroup.h"
#include "dprt/cuBQL/Triangles.h"

namespace dprt {
  extern "C" {
    int dprt_dbg_rayID = -1;
  }
  
  namespace cubql_cuda {

    __dprt_global
    void g_prepareInstances(Kernel kernel,
                            int numInstances,
                            InstanceGroup::InstancedObjectDD *instances,
                            bool hasTransforms,
                            impl_affine_t *worldToObjectXfms,
                            impl_affine_t *objectToWorldXfms,
                            impl_box_t *d_instBounds,
                            bool *hasAnyActualTransform)
    {
      int tid = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numInstances) return;
      impl_affine_t xfm;
      if (!hasTransforms) {
        xfm = impl_affine_t();
        objectToWorldXfms[tid] = xfm;
      } else {
        xfm = objectToWorldXfms[tid];
      }

      worldToObjectXfms[tid] = rcp(xfm);
      instances[tid].hasXfm = (xfm != impl_affine_t());
      if (instances[tid].hasXfm)
        *hasAnyActualTransform = true;

      impl_box_t objBounds = instances[tid].group.bvh.nodes[0].bounds;
      impl_vec_t b0 = objBounds.lower;
      impl_vec_t b1 = objBounds.upper;
      impl_box_t instBounds;
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b0.x,b0.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b0.x,b0.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b0.x,b1.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b0.x,b1.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b1.x,b0.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b1.x,b0.y,b1.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b1.x,b1.y,b0.z)));
      instBounds.extend(xfmPoint(xfm,impl_vec_t(b1.x,b1.y,b1.z)));
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
      
      d_worldToObjectXfms  = (impl_affine_t*)
        omp_target_alloc(numInstances*sizeof(impl_affine_t),context->gpuID);
      d_objectToWorldXfms  = (impl_affine_t*)
        omp_target_alloc(numInstances*sizeof(impl_affine_t),context->gpuID);
      d_hasAnyActualTransform = (bool *)
        omp_target_alloc(1*sizeof(bool),context->gpuID);
      hasAnyActualTransform = false;
      omp_target_memcpy(d_hasAnyActualTransform,
                        &hasAnyActualTransform,
                        1*sizeof(bool),
                        0,0,
                        context->gpuID,
                        context->hostID);

      if (transforms) {
        if (std::is_same<impl_affine_t,affine3d>()) {
          // internal format is doubles (input format is _always_
          // doubles!); just copy
          omp_target_memcpy(d_objectToWorldXfms,
                            transforms,
                            numInstances*sizeof(impl_affine_t),
                            0,0,
                            context->gpuID,
                            context->hostID);
        } else {
          // internal format is floats, but input format is (_always_!)
          // doubles; convert to internal format first.
          std::vector<impl_scalar_t> tmp(numInstances*12);
          for (int i=0;i<tmp.size();i++)
            tmp[i] = impl_scalar_t(((double *)transforms)[i]);
          omp_target_memcpy(d_objectToWorldXfms,
                            (impl_affine_t*)tmp.data(),
                            numInstances*sizeof(impl_affine_t),
                            0,0,
                            context->gpuID,
                            context->hostID);
        }
      }
      impl_box_t *d_instBounds = 0;
      cudaMalloc((void**)&d_instBounds,
                 numInstances*sizeof(impl_box_t));
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
           d_instBounds,
           d_hasAnyActualTransform
           );

      omp_target_memcpy(&hasAnyActualTransform,
                        d_hasAnyActualTransform,
                        1*sizeof(bool),
                        0,0,
                        context->hostID,
                        context->gpuID);
#else
      cudaMalloc((void**)&d_instanceDDs,
                 numInstances*sizeof(*d_instanceDDs));
      cudaMemcpy(d_instanceDDs,
                 instanceDDs.data(),
                 numInstances*sizeof(*d_instanceDDs),
                 cudaMemcpyDefault);
      
      cudaMalloc((void**)&d_worldToObjectXfms,
                 numInstances*sizeof(impl_affine_t));
      cudaMalloc((void**)&d_objectToWorldXfms,
                 numInstances*sizeof(impl_affine_t));
      cudaMalloc((void**)&d_hasAnyActualTransform,
                 1*sizeof(bool));
      if (transforms) {
        if (std::is_same<impl_affine_t,affine3d>()) {
          cudaMemcpy(d_objectToWorldXfms,
                     transforms,
                     numInstances*sizeof(impl_affine_t),
                     cudaMemcpyDefault);
        } else {
          std::vector<impl_scalar_t> tmp(numInstances*12);
          for (int i=0;i<tmp.size();i++)
            tmp[i] = impl_scalar_t(((double *)transforms)[i]);
          cudaMemcpy(d_objectToWorldXfms,
                     (impl_affine_t*)tmp.data(),
                     numInstances*sizeof(impl_affine_t),
                     cudaMemcpyDefault);
        }
        cudaMemcpy(d_objectToWorldXfms,
                   transforms,
                   numInstances*sizeof(impl_affine_t),
                   cudaMemcpyDefault);
      }
      impl_box_t *d_instBounds = 0;
      cudaMalloc((void**)&d_instBounds,
                 numInstances*sizeof(impl_box_t));
      hasAnyActualTransform = false;
      cudaMemcpy(d_hasAnyActualTransform,
                 &hasAnyActualTransform,
                 1*sizeof(bool),
                 cudaMemcpyDefault);
      g_prepareInstances
        <<<divRoundUp(numInstances,128),128>>>
        (Kernel{},
         numInstances,
         d_instanceDDs,
         transforms != 0,
         d_worldToObjectXfms,
         d_objectToWorldXfms,
         d_instBounds,
         d_hasAnyActualTransform
         );
      CUBQL_CUDA_SYNC_CHECK();
      cudaMemcpy(&hasAnyActualTransform,
                 d_hasAnyActualTransform,
                 1*sizeof(bool),
                 cudaMemcpyDefault);
#endif

      ::cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 1;
#if DPRT_OMP
      std::vector<impl_box_t> h_instBounds(numInstances);
      omp_target_memcpy(h_instBounds.data(),
                        d_instBounds,
                        numInstances*sizeof(*d_instBounds),
                        0,0,context->hostID,context->gpuID);
      impl_bvh_t h_bvh;
      cuBQL::cpu::spatialMedian(h_bvh,
                                h_instBounds.data(),
                                numInstances,
                                buildConfig);
      bvh = h_bvh;
      // --
      bvh.nodes = (typename impl_bvh_t::Node *)
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
      ::cuBQL::cuda::sahBuilder(bvh,
                                d_instBounds,
                                numInstances,
                                buildConfig,
                                0,
                                memResource);
      
      CUBQL_CUDA_SYNC_CHECK();
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
      return { d_instanceDDs,
               d_worldToObjectXfms,
               bvh };
    }

    __dprt_global
    void g_traceRays_twoLevel(Kernel kernel,
                              /*! the device data for the instancegroup itself */
                              InstanceGroup::DD world,
                              DPRTRay *rays,
                              DPRTHit *hits,
                              size_t numRays,
                              uint64_t flags,
                              int dbgRayID)
    {
      int tid = kernel.workIdx();
      if (tid >= numRays) return;

      if (rays[tid].tMax <= 0.) return;
      
#ifdef NDEBUG
      const bool dbg = false;
#else
      // bool dbg = (tid == 512*1024+512);
      bool dbg = (tid == dbgRayID);
#endif

      DPRTHit hit = hits[tid];
      hit.primID = -1;
      hit.instID = -1;
      hit.t = 1e30;
      struct ObjectSpaceTravState {
        int instID = -1;
        InstanceGroup::InstancedObjectDD instance;
        impl_ray_t ray;
      } objectSpace;
      impl_ray_t worldRay(impl_vec_t((const vec3d&)rays[tid].origin),
                          impl_vec_t((const vec3d&)rays[tid].direction),
                          impl_scalar_t(rays[tid].tMin),
                          impl_scalar_t(rays[tid].tMax));
      auto intersectPrim
        = [&hit,&worldRay,&objectSpace,flags,dbg](uint32_t primID)
        -> double
      {
        if (dbg)
          printf("intersectPrim %i\n",primID);
        impl_RayTriangleIntersection isec;
        auto &group = objectSpace.instance.group;
        PrimRef prim = group.primRefs[primID];
        const impl_triangle_t tri = group.getTriangle(prim);

        auto getNormal = [tri]() { return cross(tri.b-tri.a,tri.c-tri.a); };
        bool culled = false;
        if (flags & DPRT_CULL_FRONT)
          culled |= (dot(getNormal(),objectSpace.ray.direction) <= 0.);
        if (flags & DPRT_CULL_BACK)
          culled |= (dot(getNormal(),objectSpace.ray.direction) >= 0.);
        if (dbg)
          printf("culled? %i\n",(int)culled);
        
        if (!culled && isec.compute(objectSpace.ray,tri,dbg)) {
          hit.primID = prim.primID;
          hit.instID = objectSpace.instID;
          hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.t = isec.t;
          worldRay.tMax = isec.t;
          objectSpace.ray.tMax = isec.t;
          if (dbg)
            printf("*** HIT *** at %lf\n",hit.t);
        }
        return worldRay.tMax;
      };
      auto enterBlas = [world,worldRay,&objectSpace,dbg]
        (impl_ray_t &out_ray,
         impl_bvh_t &out_bvh,
         int instID) 
      {
        objectSpace.instance = world.instancedGroups[instID];
        objectSpace.instID = instID;
        objectSpace.ray = worldRay;
        if (objectSpace.instance.hasXfm) {
          impl_affine_t worldToObjectXfm
            = world.worldToObjectXfms[instID];
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

      if (dbg)
        printf("REPORTED HIT DIST at %lf idx %i\n",hit.t,hit.primID);
      
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

      impl_ray_t worldRay(impl_vec_t((const vec3d&)rays[tid].origin),
                          impl_vec_t((const vec3d&)rays[tid].direction),
                          impl_scalar_t(rays[tid].tMin),
                          impl_scalar_t(rays[tid].tMax));
      // impl_ray_t worldRay((const vec3d&)rays[tid].origin,
      //                     (const vec3d&)rays[tid].direction,
      //                     rays[tid].tMin,
      //                     rays[tid].tMax);
      
      auto intersectPrim
        = [&hit,&worldRay,object_instance,flags,dbg](uint32_t primID)
        -> double
      {
        impl_RayTriangleIntersection isec;
        auto &group = object_instance.group;
        PrimRef prim = group.primRefs[primID];
        const impl_triangle_t tri = group.getTriangle(prim);

        auto getNormal = [tri]() { return cross(tri.b-tri.a,tri.c-tri.a); };
        bool culled = false;
        if (flags & DPRT_CULL_FRONT)
          culled |= (dot(getNormal(),worldRay.direction) <= 0.);
        if (flags & DPRT_CULL_BACK)
          culled |= (dot(getNormal(),worldRay.direction) >= 0.);
        if (!culled && isec.compute(worldRay,tri,dbg)) {
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
      if (numInstances == 1 && !hasAnyActualTransform) {
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
                               flags,
                               dbg_rayID);
#else
        int bs = 128;
        int nb = divRoundUp(numRays,bs);
        g_traceRays_twoLevel<<<nb,bs>>>(Kernel(),
                                        getDD(),
                                        d_rays,d_hits,numRays,
                                        flags,
                                        dprt_dbg_rayID);
        cudaDeviceSynchronize();
#endif
      }
    }      
  }
}

