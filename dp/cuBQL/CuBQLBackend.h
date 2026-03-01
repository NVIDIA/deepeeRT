// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Context.h"
#if DP_OMP
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

namespace dp {
  namespace cubql_cuda {
    
    using namespace ::cuBQL;
    
    using bvh3d = bvh_t<double,3>;
    using TriangleDP = cuBQL::triangle_t<double>;
    using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<double>;
    using cuBQL::affine3d;

#if DP_OMP
# define __dp_global /* nothing */
    struct Kernel {
      int threadIdx;
      inline int workIdx() const { return threadIdx; }
    };
#elif defined (__CUDACC__)
# define __dp_global __global__
    struct Kernel {
      inline __device__
      int workIdx() const { return threadIdx.x+blockIdx.x*blockDim.x; }
    };
#endif

    
    /*! an array that can upload an array from host to device, and free
      on destruction. If the pointer provided is *already* a device
      pointer this will just use that pointer */
    template<typename T>
    struct AutoUploadArray {
      // AutoUploadArray() = default;
      AutoUploadArray(Context *context,
                      const T *elements, size_t count);
      AutoUploadArray(const AutoUploadArray &other) = delete;
      ~AutoUploadArray();

      // move operator
      AutoUploadArray &operator=(AutoUploadArray &&other);
      const T *elements      = 0;
      size_t   count         = 0;
      bool     needsCudaFree = false;
      Context *context = 0;
    };
  
    struct CuBQLCUDABackend : public dp::Context
    {
      CuBQLCUDABackend(int gpuID);
      virtual ~CuBQLCUDABackend() = default;

      dp::InstanceGroup *
      createInstanceGroup(const std::vector<dp::TrianglesGroup *> &groups,
                          const DPRAffine *transforms) override;
    
      dp::TriangleMesh *
      createTriangleMesh(uint64_t         userData,
                         const vec3d     *vertexArray,
                         int              vertexCount,
                         const vec3i     *indexArray,
                         int              indexCount) override;
      
      dp::TrianglesGroup *
      createTrianglesGroup(const std::vector<dp::TriangleMesh *> &geoms) override;
    };

#ifdef __CUDACC__
    // ==================================================================
    // INLINE IMPLEMENTATION SECTION
    // ==================================================================
    template<typename T> inline
    AutoUploadArray<T>::AutoUploadArray(Context *context,
                                        const T *elements,
                                        size_t count)
      : context(context)
    {
      this->count = count;
      // if (isDevicePointer(elements)) {
      //   this->elements = elements;
      //   this->needsCudaFree = false;
      // } else {
      // iw - for now, ALWAYS create a copy
#if DP_OMP
      this->elements = (T*)omp_target_alloc(count*sizeof(T),
                                            context->gpuID);
      omp_target_memcpy((void*)this->elements,(void*)elements,
                        count*sizeof(T),
                        0,0,
                        context->gpuID,
                        context->hostID);
#else
      CUBQL_CUDA_SYNC_CHECK();
      cudaMalloc((void **)&this->elements,count*sizeof(T));
      cudaMemcpy((void*)this->elements,(void*)elements,count*sizeof(T),
                 cudaMemcpyDefault);
      CUBQL_CUDA_SYNC_CHECK();
#endif
      this->needsCudaFree = true;
    }

    template<typename T> inline
    AutoUploadArray<T> &
    AutoUploadArray<T>::operator=(AutoUploadArray &&other)
    {
      context = other->context;
      elements = other.elements; other.elements = 0;
      count = other.count; other.count = 0;
      needsCudaFree = other.needsCudaFree; other.needsCudaFree = 0;
      return *this;
    }
    
    template<typename T> inline
    AutoUploadArray<T>::~AutoUploadArray() {
#if DP_OMP
      if (needsCudaFree)
        omp_target_free((void*)elements,context->gpuID);
#else
      if (needsCudaFree) cudaFree((void*)elements);
      CUBQL_CUDA_SYNC_CHECK();
#endif
    }
#endif
    
  }
}


  
