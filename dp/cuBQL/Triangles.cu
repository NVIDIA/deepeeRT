// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/cuBQL/Triangles.h"

namespace dp {
  namespace cubql_cuda {

    /*! triangle mesh representation on the device */
    __global__
    void generateTriangleInputs(int      meshID,
                                PrimRef *primRefs,
                                box3d   *primBounds,
                                int      numTrisThisMesh,
                                TriangleMesh::DD mesh)
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
    
    TriangleMesh::TriangleMesh(Context         *context,
                               uint64_t         userData,
                               const vec3d     *vertexArray,
                               int              vertexCount,
                               const vec3i     *indexArray,
                               int              indexCount)
      : dp::TriangleMesh(context,userData,
                         vertexArray,vertexCount,
                         indexArray,indexCount),
        vertices(vertexArray,vertexCount),
        indices(indexArray,indexCount)
    {}
    
    TrianglesGroup::TrianglesGroup(Context *context,
                                   const std::vector<dp::TriangleMesh *> &meshes)
      : dp::TrianglesGroup(context,meshes)
    {
      CUBQL_CUDA_SYNC_CHECK();
      SetActiveGPU forDuration(context->gpuID);
      
      int numTrisTotal = 0;
      std::vector<TriangleMesh::DD> devMeshes;
      for (auto _geom : meshes) {
        TriangleMesh *geom = (TriangleMesh*)_geom;
        devMeshes.push_back(geom->getDD());
        numTrisTotal += geom->indices.count;
      }
      cudaMalloc((void **)&d_meshDDs,
                 devMeshes.size()*sizeof(TriangleMesh::DD));
      cudaMemcpy((void*)d_meshDDs,devMeshes.data(),
                 devMeshes.size()*sizeof(TriangleMesh::DD),cudaMemcpyDefault);
      
      cudaMalloc((void **)&d_primRefs,numTrisTotal*sizeof(*d_primRefs));

      box3d   *d_primBounds = nullptr;
      cudaMalloc((void **)&d_primBounds,numTrisTotal*sizeof(*d_primBounds));
      
      int offset = 0;
      for (int meshID=0;meshID<(int)meshes.size();meshID++) {
        TriangleMesh *mesh = (TriangleMesh *)meshes[meshID];
        int count = mesh->indices.count;
        int bs = 128;
        int nb = divRoundUp(count,bs);
        generateTriangleInputs<<<nb,bs>>>(meshID,
                                          d_primRefs+offset,
                                          d_primBounds+offset,
                                          count,
                                          mesh->getDD());
        offset += count;
      }
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
    }
  
    TrianglesGroup::~TrianglesGroup()
    {
      CUBQL_CUDA_SYNC_CHECK();
      cudaFree(d_meshDDs);
      cudaFree(d_primRefs);
      ::cuBQL::cuda::free(bvh);
      CUBQL_CUDA_SYNC_CHECK();
    }

  }
}

    
