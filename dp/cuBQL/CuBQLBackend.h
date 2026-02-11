// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Context.h"
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
    
    /*! an array that can upload an array from host to device, and free
      on destruction. If the pointer provided is *already* a device
      pointer this will just use that pointer */
    template<typename T>
    struct AutoUploadArray {
      AutoUploadArray() = default;
      AutoUploadArray(const T *elements, size_t count);
      AutoUploadArray(const AutoUploadArray &other) = delete;
      ~AutoUploadArray();

      // move operator
      AutoUploadArray &operator=(AutoUploadArray &&other);
      const T *elements      = 0;
      size_t   count         = 0;
      bool     needsCudaFree = false;
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
    AutoUploadArray<T>::AutoUploadArray(const T *elements,
                                        size_t count)
    {
      this->count = count;
      // if (isDevicePointer(elements)) {
      //   this->elements = elements;
      //   this->needsCudaFree = false;
      // } else {
      // iw - for now, ALWAYS create a copy
      CUBQL_CUDA_SYNC_CHECK();
      cudaMalloc((void **)&this->elements,count*sizeof(T));
      cudaMemcpy((void*)this->elements,elements,count*sizeof(T),
                 cudaMemcpyDefault);
      this->needsCudaFree = true;
      CUBQL_CUDA_SYNC_CHECK();
    }

    template<typename T> inline
    AutoUploadArray<T> &
    AutoUploadArray<T>::operator=(AutoUploadArray &&other)
    {
      elements = other.elements; other.elements = 0;
      count = other.count; other.count = 0;
      needsCudaFree = other.needsCudaFree; other.needsCudaFree = 0;
      return *this;
    }
    
    template<typename T> inline
    AutoUploadArray<T>::~AutoUploadArray() {
      if (needsCudaFree) cudaFree((void*)elements);
      CUBQL_CUDA_SYNC_CHECK();
    }
#endif
    
  }
}


  
