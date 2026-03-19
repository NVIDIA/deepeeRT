// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_CPU_BUILDER_IMPLEMENTATION 1

#include "dprt/cuBQL/CuBQLBackend.h"
#include "dprt/cuBQL/Triangles.h"
#include "dprt/cuBQL/InstanceGroup.h"

#ifdef DPRT_OMP
# include "cuBQL/builder/omp.h"
// do NOT instantiate cuda builder
namespace cuBQL {
  using impl_scalar_t = dprt::cubql_cuda::impl_scalar_t;
  namespace omp {
    template
    void spatialMedian(BinaryBVH<impl_scalar_t,3>   &bvh,
                       const box_t<impl_scalar_t,3> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig,
                       Context          *ctx);
    template
    void freeBVH(BinaryBVH<impl_scalar_t,3> &bvh,
                 Context          *ctx);
  }
}
#else
namespace cuBQL {
  using impl_scalar_t = dprt::cubql_cuda::impl_scalar_t;
  namespace cpu {
    template
    void spatialMedian(BinaryBVH<impl_scalar_t,3>   &bvh,
                       const box_t<impl_scalar_t,3> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig);
    template
    void freeBVH(BinaryBVH<impl_scalar_t,3> &bvh);

  }
  namespace cuda {
    template
    void sahBuilder(BinaryBVH<impl_scalar_t,3>   &bvh,
                    const box_t<impl_scalar_t,3> *boxes,
                    uint32_t          numPrims,
                    BuildConfig       buildConfig,
                    cudaStream_t       s,
                    cuBQL::GpuMemoryResource &memResource);

    template
    void free(BinaryBVH<impl_scalar_t,3> &bvh,
              cudaStream_t      s,
              GpuMemoryResource& memResource);
  }
}
#endif

namespace dprt {

  Context *Context::create(int gpuID)
  {
    return new cubql_cuda::CuBQLCUDABackend(gpuID);
  };
  
  
  namespace cubql_cuda {
    
    CuBQLCUDABackend::CuBQLCUDABackend(int gpuID)
      : Context(gpuID)
    {
      SetActiveGPU forDuration(gpuID);
      cudaFree(0);
    }


    dprt::InstanceGroup *
    CuBQLCUDABackend::
    createInstanceGroup(const std::vector<dprt::TrianglesGroup *> &groups,
                        const DPRTAffine *transforms)
    {
      return new InstanceGroup(this, groups, transforms);
    }
    
    dprt::TriangleMesh *
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
    
    dprt::TrianglesGroup *
    CuBQLCUDABackend::
    createTrianglesGroup(const std::vector<dprt::TriangleMesh *> &geoms)
    {
      return new TrianglesGroup(this,geoms);
    }
    
  } // ::cubql_cuda
} // ::drpt

  
