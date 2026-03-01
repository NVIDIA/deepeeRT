// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/Triangles.h"

namespace dp {
  namespace cubql_cuda {

    /*! triangle mesh representation on the device */
    __dp_global
    void generateTriangleInputs(Kernel   kernel,
                                int      meshID,
                                PrimRef *primRefs,
                                box3d   *primBounds,
                                int      numTrisThisMesh,
                                TriangleMesh::DD mesh)
    {
      int tid = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numTrisThisMesh) return;

      vec3i idx = mesh.indices[tid];
      box3d bb;
      bb.extend(mesh.vertices[idx.x]);
      bb.extend(mesh.vertices[idx.y]);
      bb.extend(mesh.vertices[idx.z]);

      primRefs[tid] = { meshID, tid };
      primBounds[tid] = bb;
    }
    
    TriangleMesh::TriangleMesh(Context         *context,
                               uint64_t         userData,
                               const vec3d     *vertexArray,
                               int              vertexCount,
                               const vec3i     *indexArray,
                               int              indexCount)
      : dp::TriangleMesh(context,userData,
                         vertexArray,vertexCount,
                         indexArray,indexCount),
        vertices(context,vertexArray,vertexCount),
        indices(context,indexArray,indexCount)
    {}
    
    TrianglesGroup::TrianglesGroup(Context *context,
                                   const std::vector<dp::TriangleMesh *> &meshes)
      : dp::TrianglesGroup(context,meshes)
    {
#ifndef DP_OMP
      CUBQL_CUDA_SYNC_CHECK();
      SetActiveGPU forDuration(context->gpuID);
#endif
      int numTrisTotal = 0;
      std::vector<TriangleMesh::DD> devMeshes;
      for (auto _geom : meshes) {
        TriangleMesh *geom = (TriangleMesh*)_geom;
        devMeshes.push_back(geom->getDD());
        numTrisTotal += geom->indices.count;
      }
      box3d   *d_primBounds = nullptr;
#ifdef DP_OMP
      d_meshDDs
        = (TriangleMesh::DD*)
        omp_target_alloc(devMeshes.size()*sizeof(TriangleMesh::DD),
                         context->gpuID);
      omp_target_memcpy(d_meshDDs,devMeshes.data(),
                        devMeshes.size()*sizeof(TriangleMesh::DD),
                        0,0,
                        context->gpuID,
                        context->hostID);
      d_primRefs
        = (PrimRef*)omp_target_alloc(numTrisTotal*sizeof(*d_primRefs),
                                     context->gpuID);
      d_primBounds
        = (box3d*)omp_target_alloc(numTrisTotal*sizeof(*d_primBounds),
                                     context->gpuID);
#else
      cudaMalloc((void **)&d_meshDDs,
                 devMeshes.size()*sizeof(TriangleMesh::DD));
      cudaMemcpy((void*)d_meshDDs,devMeshes.data(),
                 devMeshes.size()*sizeof(TriangleMesh::DD),cudaMemcpyDefault);
      
      cudaMalloc((void **)&d_primRefs,numTrisTotal*sizeof(*d_primRefs));

      cudaMalloc((void **)&d_primBounds,numTrisTotal*sizeof(*d_primBounds));
#endif
      
      int offset = 0;
      for (int meshID=0;meshID<(int)meshes.size();meshID++) {
        TriangleMesh *mesh = (TriangleMesh *)meshes[meshID];
        int count = mesh->indices.count;
#if DP_OMP
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
        for (int i=0;i<count;i++)
          generateTriangleInputs(Kernel{i},
                                 meshID,
                                 d_primRefs+offset,
                                 d_primBounds+offset,
                                 count,
                                 mesh->getDD());
        PING;
#else
        int bs = 128;
        int nb = divRoundUp(count,bs);
        generateTriangleInputs<<<nb,bs>>>(Kernel{},
                                          meshID,
                                          d_primRefs+offset,
                                          d_primBounds+offset,
                                          count,
                                          mesh->getDD());
#endif
        offset += count;
      }
#if DP_OMP
      // temporarily copy all back to host because we need to use
      // cubql host builder...
      std::vector<PrimRef> h_primRefs(numTrisTotal);
      omp_target_memcpy(h_primRefs.data(),d_primRefs,
                        h_primRefs.size()*sizeof(h_primRefs[0]),
                        0,0,
                        context->hostID,
                        context->gpuID);
      std::vector<box3d> h_primBounds(numTrisTotal);
      omp_target_memcpy(h_primBounds.data(),d_primBounds,
                        h_primBounds.size()*sizeof(h_primBounds[0]),
                        0,0,
                        context->hostID,
                        context->gpuID);

      bvh3d h_bvh;
      cuBQL::cpu::spatialMedian(h_bvh,
                                h_primBounds.data(),
                                numTrisTotal,
                                ::cuBQL::BuildConfig());
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
      //--      
      omp_target_free(d_primBounds,context->gpuID);
#else
      cudaStreamSynchronize(0);
      
      // std::cout << "#dpr: building BVH over " << prettyNumber(numTrisTotal)
      //           << " triangles" << std::endl;
      CUBQL_CUDA_SYNC_CHECK();
      DeviceMemoryResource memResource;
#if 0
      ::cuBQL::gpuBuilder(bvh,
                          d_primBounds,
                          numTrisTotal,
                          ::cuBQL::BuildConfig(),
                          0,
                          memResource);
#else
      ::cuBQL::cuda::sahBuilder(bvh,
                                d_primBounds,
                                numTrisTotal,
                                ::cuBQL::BuildConfig(),
                                0,
                                memResource);
#endif
      // std::cout << "#dpr: ... bvh built." << std::endl;
      
      cudaFree(d_primBounds);
      CUBQL_CUDA_SYNC_CHECK();
#endif
    }
  
    TrianglesGroup::~TrianglesGroup()
    {
#if DP_OMP
      omp_target_free(bvh.primIDs,context->gpuID);
      omp_target_free(bvh.nodes,context->gpuID);
      omp_target_free(d_meshDDs,context->gpuID);
      omp_target_free(d_primRefs,context->gpuID);
#else
      CUBQL_CUDA_SYNC_CHECK();
      cudaFree(d_meshDDs);
      cudaFree(d_primRefs);
      ::cuBQL::cuda::free(bvh);
      CUBQL_CUDA_SYNC_CHECK();
#endif
    }

  }
}

    
