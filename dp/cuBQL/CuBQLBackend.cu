// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_CPU_BUILDER_IMPLEMENTATION 1

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/cuBQL/Triangles.h"
#include "dp/cuBQL/InstanceGroup.h"

namespace cuBQL {
  namespace cpu {
    template
    void spatialMedian(BinaryBVH<double,3>   &bvh,
                       const box_t<double,3> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig);
    template
    void freeBVH(BinaryBVH<double,3> &bvh);

  }
  namespace cuda {
    template
    void sahBuilder(BinaryBVH<double,3>   &bvh,
                    const box_t<double,3> *boxes,
                    uint32_t          numPrims,
                    BuildConfig       buildConfig,
                    cudaStream_t       s,
                    GpuMemoryResource &memResource);

    template
    void free(BinaryBVH<double,3> &bvh,
              cudaStream_t      s,
              GpuMemoryResource& memResource);
  }
}


namespace dp {
  namespace cubql_cuda {
    CuBQLCUDABackend::CuBQLCUDABackend(int gpuID)
      : Context(gpuID)
    {
      SetActiveGPU forDuration(gpuID);
      cudaFree(0);
    }


    dp::InstanceGroup *
    CuBQLCUDABackend::
    createInstanceGroup(const std::vector<dp::TrianglesGroup *> &groups,
                        const DPRAffine *transforms)
    {
      return new InstanceGroup(this, groups,(const affine3d*)transforms);
    }
    
    dp::TriangleMesh *
    CuBQLCUDABackend::
    createTriangleMesh(uint64_t         userData,
                       const vec3d     *vertexArray,
                       int              vertexCount,
                       const vec3i     *indexArray,
                       int              indexCount) 
    {
      return new TriangleMesh(this,
                              userData,
                              vertexArray,
                              vertexCount,
                              indexArray,
                              indexCount);
    }
    
    dp::TrianglesGroup *
    CuBQLCUDABackend::
    createTrianglesGroup(const std::vector<dp::TriangleMesh *> &geoms)
    {
      return new TrianglesGroup(this,geoms);
    }
  } // :: cubql_cuda
  
}

  
