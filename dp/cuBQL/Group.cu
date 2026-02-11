// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"

namespace dp {
  namespace cubql_cuda {
    __global__
    void g_traceFirstGroupOnly(TrianglesDP::DevGroup group,
                               Ray *rays,
                               Hit *hits,
                               int numRays)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

#ifdef NDEBUG
      const bool dbg = false;
#else
      bool dbg = (tid == -1);
#endif
      
      Hit hit = hits[tid];
      hit.primID = -1;
      int instID = 0;
      ::cuBQL::ray3d ray(rays[tid].origin,
                         rays[tid].direction,
                         rays[tid].tMin,
                         rays[tid].tMax);

      if (dbg) {
        cuBQL::dout << "dbg ray " << ray << "\n";
        cuBQL::dout << "bvh.nodes " << (int*)group.bvh.nodes << "\n";
        cuBQL::dout << "bvh.primIDs " << (int*)group.bvh.primIDs << "\n";
        cuBQL::dout << "group.meshes " << (int*)group.meshes << "\n";
        cuBQL::dout << "group.mesh0 " << group.meshes[0].userData << "\n";
        cuBQL::dout << "group.primRefs " << (int*)group.primRefs << "\n";
      }
      auto intersectPrim = [&ray,&hit,group,instID,dbg](uint32_t primID)
        -> double
      {
        RayTriangleIntersection isec;
        PrimRef prim = group.primRefs[primID];
        const TriangleDP tri = group.getTriangle(prim);
        if (isec.compute(ray,tri)) {
          hit.primID = prim.primID;
          hit.instID = instID;
          hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.t = isec.t;
          if (dbg) printf("hit %i %i\n",hit.instID,hit.primID);
          ray.tMax = isec.t;
        }
        return ray.tMax;
      };
      ::cuBQL::shrinkingRayQuery::forEachPrim(intersectPrim,group.bvh,ray);
      hits[tid] = hit;;
    }





    __global__
    void g_traceWorld(/*! the device data for the instancegroup itself */
                      InstanceGroup::DD model,
                      /*! the list of instance transforms */
                      const DPRAffine     *const d_transforms,
                      /*! the list of instantiated groups */
                      const TrianglesDP::DevGroup *d_instantiatedGroups,
                      Ray *rays,
                      Hit *hits,
                      int numRays)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

#ifdef NDEBUG
      const bool dbg = false;
#else
      bool dbg = (tid == -1);
#endif

      Hit hit = hits[tid];
      hit.primID = -1;
      hit.instID = -1;
      struct {
        int instID = -1;
        TrianglesDP::Group group;
        ::cuBQL::ray3d ray;
      } object;
      ::cuBQL::ray3d worldRay(rays[tid].origin,
                              rays[tid].direction,
                              rays[tid].tMin,
                              rays[tid].tMax);
      
      auto intersectPrim = [&ray,&hit,&object,instID,dbg](uint32_t primID)
        -> double
      {
        RayTriangleIntersection isec;
        PrimRef prim = object.group.primRefs[primID];
        const TriangleDP tri = group.getTriangle(prim);
        if (isec.compute(ray,tri)) {
          hit.primID = prim.primID;
          hit.instID = object.instID;
          hit.geomUserData = group.meshes[prim.geomID].userData;
          hit.t = isec.t;
          if (dbg) printf("hit %i %i\n",hit.instID,hit.primID);
          ray.tMax = isec.t;
        }
        return ray.tMax;
      };
      auto enterBlas = [this,model,&current]
        (cuBQL::ray3f &out_ray,
         cuBQL::bvh3f &out_bvh,
         int instID) 
      {
        current.group  = d_instantiatedGroups[instID];
        current.instID = instID;
        object.ray = world.ray;
        if (!isUnitTransform(currentInstance->worldToObjectXfm)) {
          object.ray.origin
            = xfmPoint(currentInstance->worldToObjectXfm,world.origin);
          object.ray.direction
            = xfmVector(currentInstance->worldToObjectXfm,world.direction);
        }
        out_bvh = {0,0,0,0};
        out_bvh.nodes = object.group.bvhNodes;
      };
      auto leaveBlas = [this]() -> void {
        /* nothing to do */
      };
      
      ::cuBQL::shrinkingRayQuery::twoLevel::forEachPrim
          (enterBlas,leaveBlas,intersectPrim,model->bvh,ray);
      
      hits[tid] = hit;
    }

    
    struct InstancesDP : public dp::InstancesDPImpl {
      InstancesDP(CuBQLCUDABackend *be,
                  dp::InstancesDPGroup *fe)
        : InstancesDPImpl(fe), be(be)
      {
        int numInstances = fe->instances.size();
        if (numInstances == 0) return;

        std::vector<TrianglesDP::DevGroup> instancedGroups;
        for (auto feGroup : fe->groups) {
          dp::TrianglesDPGroup *group = feGroup
          instancedGroups.push_back(inst
                                    }
        cudaMalloc((void **)&d_instancedGroups,
                   numInstances*sizeof(*d_triangleGroups));
        cudaMemcpy((void*)d_instancedGroups,instancedGroups.data(),
                   numInstances*sizeof(*d_triangleGroups),
                   cudaMemcpyDefault);
      }
      void trace(Ray *rays,
                 Hit *hits,
                 int numRays) override;

      TrianglesDP::DevGroup *d_instancedGroups = 0;
      CuBQLCUDABackend *const be;
    };

    void InstancesDP::trace(Ray *rays,
                            Hit *hits,
                            int numRays) 
    {
      CUBQL_CUDA_SYNC_CHECK();
      int bs = 128;
      int nb = divRoundUp(numRays,bs);
#if 0
      assert(fe->groups.size() == 1);
      assert(fe->d_transforms == nullptr);
      TrianglesDPGroup *tg = (TrianglesDPGroup *)fe->groups[0];
      assert(tg);
      TrianglesDP *triangles = (TrianglesDP*)tg->impl.get();
      assert(triangles);

      g_traceFirstGroupOnly<<<nb,bs>>>(triangles->getDevGroup(),
                                       rays,hits,
                                       numRays);
#else
      g_traceInstances<<<nb,bs>>>(triangles->getDevGroup(),
                         rays,hits,
                         numRays);
#endif
      CUBQL_CUDA_SYNC_CHECK();
    }
    
  } // :: cubql_cuda
  
  CuBQLCUDABackend::CuBQLCUDABackend(Context *const context)
    : Backend(context)
  {
    SetActiveGPU forDuration(context->gpuID);
    cudaFree(0);
  }

  std::shared_ptr<InstancesDPImpl>
  CuBQLCUDABackend::createInstancesDPImpl(dp::InstancesDPGroup *fe)
  { return std::make_shared<cubql_cuda::InstancesDP>(this,fe); }
    
  std::shared_ptr<TrianglesDPImpl>
  CuBQLCUDABackend::createTrianglesDPImpl(dp::TrianglesDPGroup *fe) 
  { return std::make_shared<cubql_cuda::TrianglesDP>(this,fe); }
  
}

  
