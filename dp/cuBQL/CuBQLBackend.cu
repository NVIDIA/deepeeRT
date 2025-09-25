// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Context.h"
#include "dp/Group.h"
#include "dp/World.h"

#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

namespace dp {
  namespace cubql_cuda {
    using namespace ::cuBQL;
    
    using bvh3d = bvh_t<double,3>;

    using TriangleDP = cuBQL::triangle_t<double>;
    using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<double>;
    
    /*! an array that can upload an array from host to device, and free
      on destruction. If the pointer provided is *already* a device
      pointer this will just use that pointer */
    template<typename T>
    struct AutoUploadArray {
      AutoUploadArray(const T *elements, int count)
      {
        this->count = count;
        if (isDevicePointer(elements)) {
          this->elements = elements;
          this->needsCudaFree = false;
        } else {
          cudaMalloc((void **)&this->elements,count*sizeof(T));
          cudaMemcpy((void*)this->elements,elements,count*sizeof(T),
                     cudaMemcpyDefault);
          this->needsCudaFree = true;
        }
      }
      
      ~AutoUploadArray() { if (needsCudaFree) cudaFree((void*)elements); }
    
      const T *elements      = 0;
      int  count         = 0;
      bool needsCudaFree = false;
    };

    /*! triangle mesh representation on the device */
    struct DevMesh {
      const vec3d   *vertices;
      const vec3i   *indices;
      uint64_t userData;
    };
  
    struct HostMesh {
      HostMesh(uint32_t userData,
               const vec3d *verticesArray, int verticesCount,
               const vec3i *indicesArray, int indicesCount)
        : userData(userData),
          vertices(verticesArray,verticesCount),
          indices(indicesArray,indicesCount)
      {}
    
      DevMesh getDD() const
      { return { vertices.elements, indices.elements, userData }; }
    
      AutoUploadArray<vec3d> vertices;
      AutoUploadArray<vec3i> indices;
      uint64_t         const userData;
    };
  
    struct TrianglesDP : public dp::TrianglesDPImpl {
      TrianglesDP(CuBQLCUDABackend *be,
                  dp::TrianglesDPGroup *fe);
      virtual ~TrianglesDP();
    
      struct DevGroup {
        inline __cubql_both TriangleDP getTriangle(PrimRef prim) const;
        
        bvh3d     bvh;
        DevMesh  *meshes;
        PrimRef  *primRefs;
      };

      DevGroup getDevGroup() const
      { return { bvh,meshes,primRefs }; }
      
      bvh3d     bvh;
      // DevGroup *group  = nullptr;
      DevMesh  *meshes = nullptr;
      PrimRef  *primRefs = nullptr;
      
      /*! these are stored and owned on the host, and also manage their
        vertex arrays' ownerhip; but vertex arrays themselves will be
        device accessible */
      std::vector<std::shared_ptr<HostMesh>> hostMeshes;
    
      CuBQLCUDABackend     *const be;
    };

    inline __cubql_both
    TriangleDP TrianglesDP::DevGroup::getTriangle(PrimRef prim) const
    {
      DevMesh mesh = meshes[prim.geomID];
      vec3i idx = mesh.indices[prim.primID];
      return { mesh.vertices[idx.x],mesh.vertices[idx.y],mesh.vertices[idx.z] };
    }
    
    __global__
    void generateTriangleInputs(int meshID,
                                PrimRef *primRefs,
                                box3d *primBounds,
                                int numTrisThisMesh,
                                DevMesh mesh)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numTrisThisMesh) return;

      vec3i idx = mesh.indices[tid];
      box3d bb;
      bb.extend(mesh.vertices[idx.x]);
      bb.extend(mesh.vertices[idx.y]);
      bb.extend(mesh.vertices[idx.z]);

      primRefs[tid] = { meshID, tid };
      primBounds[tid] = bb;
    }
    
    TrianglesDP::TrianglesDP(CuBQLCUDABackend *be,
                             dp::TrianglesDPGroup *fe)
      : TrianglesDPImpl(fe), be(be)
    {
      SetActiveGPU forDuration(be->gpuID);
      
      int numTrisTotal = 0;
      std::vector<DevMesh> devMeshes;
      for (auto geom : fe->geoms) {
        numTrisTotal += geom->indexCount;
        // this will automatically upload the vertex arrays if so required:
        auto hm = std::make_shared<HostMesh>
          (geom->userData,
           geom->vertexArray,geom->vertexCount,
           geom->indexArray,geom->indexCount);
        hostMeshes.push_back(hm);
        devMeshes.push_back(hm->getDD());
      }
      cudaMalloc((void **)&meshes,devMeshes.size()*sizeof(DevMesh));
      cudaMemcpy((void*)meshes,devMeshes.data(),
                 devMeshes.size()*sizeof(DevMesh),cudaMemcpyDefault);
      
      cudaMalloc((void **)&primRefs,numTrisTotal*sizeof(*primRefs));

      box3d   *primBounds = nullptr;
      cudaMalloc((void **)&primBounds,numTrisTotal*sizeof(*primBounds));

      int offset = 0;
      for (int meshID=0;meshID<(int)hostMeshes.size();meshID++) {
        auto &hm = hostMeshes[meshID];
        int count = hm->indices.count;
        int bs = 128;
        int nb = divRoundUp(count,bs);
        generateTriangleInputs<<<nb,bs>>>(meshID,
                                          primRefs+offset,
                                          primBounds+offset,
                                          count,
                                          hm->getDD());
        offset += count;
      }
      cudaStreamSynchronize(0);

      std::cout << "#dpr: building BVH over " << prettyNumber(numTrisTotal)
                << " triangles" << std::endl;
      DeviceMemoryResource memResource;
      ::cuBQL::cuda::sahBuilder(bvh,
                                primBounds,
                                numTrisTotal,
                                ::cuBQL::BuildConfig(),
                                0,
                                memResource);
      std::cout << "#dpr: ... bvh built." << std::endl;
      
      cudaFree(primBounds);
    }
  
    TrianglesDP::~TrianglesDP()
    {
      cudaFree(meshes);
      cudaFree(primRefs);
      ::cuBQL::cuda::free(bvh);
    }
    
    __global__ void g_trace(TrianglesDP::DevGroup group,
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
      auto intersectPrim = [&ray,&hit,group,instID,dbg](uint32_t primID) -> double {
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


    struct InstancesDP : public dp::InstancesDPImpl {
      InstancesDP(CuBQLCUDABackend *be,
                  dp::InstancesDPGroup *fe)
        : InstancesDPImpl(fe), be(be)
      {}
      void trace(Ray *rays,
                 Hit *hits,
                 int numRays) override;
      CuBQLCUDABackend *const be;
    };

    void InstancesDP::trace(Ray *rays,
                            Hit *hits,
                            int numRays) 
    {
      CUBQL_CUDA_SYNC_CHECK();
      assert(fe->groups.size() == 1);
      assert(fe->d_transforms == nullptr);
      TrianglesDPGroup *tg = (TrianglesDPGroup *)fe->groups[0];
      assert(tg);
      TrianglesDP *triangles = (TrianglesDP*)tg->impl.get();
      assert(triangles);

      int bs = 128;
      int nb = divRoundUp(numRays,bs);
      g_trace<<<nb,bs>>>(triangles->getDevGroup(),
                         rays,hits,
                         numRays);
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

  
