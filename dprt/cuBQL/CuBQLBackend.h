// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprt/Context.h"
#if DPRT_OMP
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

namespace dprt {
  namespace cubql_cuda {
    
    using namespace ::cuBQL;

    using bvh3d = bvh_t<double,3>;
    using TriangleDP = cuBQL::triangle_t<double>;
    using cuBQL::affine3d;

#ifdef DPRT_EXP_SP
    using impl_scalar_t = float;
    using impl_affine_t = cuBQL::affine3f;
#else
    using impl_scalar_t = double;
    using impl_affine_t = cuBQL::affine3d;
#endif
    using impl_vec_t = cuBQL::vec_t<impl_scalar_t,3>;
    using impl_box_t = cuBQL::box_t<impl_scalar_t,3>;
    using impl_bvh_t = bvh_t<impl_scalar_t,3>;
    using impl_triangle_t = cuBQL::triangle_t<impl_scalar_t>;
    using impl_ray_t = ::cuBQL::ray_t<impl_scalar_t>;

    using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<impl_scalar_t>;
    
#if DPRT_OMP
# define __dprt_global /* nothing */
    struct Kernel {
      int threadIdx;
      inline int workIdx() const { return threadIdx; }
    };
#elif defined (__CUDACC__)
# define __dprt_global __global__
    struct Kernel {
      inline __device__
      int workIdx() const { return threadIdx.x+blockIdx.x*blockDim.x; }
    };
#endif

    
    /*! an array that can upload an array from host to device, and free
      on destruction. If the pointer provided is *already* a device
      pointer this will just use that pointer */
    template<typename T, typename INPUT_T>
    struct AutoUploadArray {
      // AutoUploadArray() = default;
      AutoUploadArray(Context *context,
                      const INPUT_T *elements, size_t count);
      AutoUploadArray(const AutoUploadArray &other) = delete;
      ~AutoUploadArray();

      // move operator
      AutoUploadArray &operator=(AutoUploadArray &&other);
      const T *elements      = 0;
      size_t   count         = 0;
      bool     needsCudaFree = false;
      Context *context = 0;
    };
  
    struct CuBQLCUDABackend : public dprt::Context
    {
      CuBQLCUDABackend(int gpuID);
      virtual ~CuBQLCUDABackend() = default;

      dprt::InstanceGroup *
      createInstanceGroup(const std::vector<dprt::TrianglesGroup *> &groups,
                          const DPRTAffine *transforms) override;
    
      dprt::TriangleMesh *
      createTriangleMesh(uint64_t         userData,
                         const vec3d     *vertexArray,
                         int              vertexCount,
                         const vec3i     *indexArray,
                         int              indexCount) override;
      
      dprt::TrianglesGroup *
      createTrianglesGroup(const std::vector<dprt::TriangleMesh *> &geoms) override;
    };

    // ==================================================================
    // INLINE IMPLEMENTATION SECTION
    // ==================================================================

    template<typename T, typename INPUT_T>
    inline
    AutoUploadArray<T,INPUT_T> &
    AutoUploadArray<T,INPUT_T>::operator=(AutoUploadArray &&other)
    {
      context = other->context;
      elements = other.elements; other.elements = 0;
      count = other.count; other.count = 0;
      needsCudaFree = other.needsCudaFree; other.needsCudaFree = 0;
      return *this;
    }


#ifdef DPRT_OMP
    template<typename T, typename INPUT_T> inline
    AutoUploadArray<T,INPUT_T>::AutoUploadArray(Context *context,
                                        const INPUT_T *elements,
                                        size_t count)
      : context(context)
    {
      this->count = count;
      this->elements = (T*)omp_target_alloc(count*sizeof(T),
                                            context->gpuID);
      if (std::is_same<T,INPUT_T>) {
        omp_target_memcpy((void*)this->elements,(void*)elements,
                          count*sizeof(T),
                          0,0,
                          context->gpuID,
                          context->hostID);
      } else {
        std::vector<T> tmp(count);
        for (int i=0;i<count;i++)
          tmp[i] = T(elements[i]);
        omp_target_memcpy((void*)this->elements,(void*)tmp.data(),
                          count*sizeof(T),
                          0,0,
                          context->gpuID,
                          context->hostID);
      }
      this->needsCudaFree = true;
    }

    template<typename T, typename INPUT_T> inline
    AutoUploadArray<T,INPUT_T>::~AutoUploadArray() {
      if (needsCudaFree)
        omp_target_free((void*)elements,context->gpuID);
    }
#endif

#ifdef __CUDACC__
    template<typename T, typename INPUT_T> inline
    AutoUploadArray<T,INPUT_T>::AutoUploadArray(Context *context,
                                                const INPUT_T *elements,
                                                size_t count)
      : context(context)
    {
      this->count = count;
      CUBQL_CUDA_SYNC_CHECK();
      cudaMalloc((void **)&this->elements,count*sizeof(T));
      if (std::is_same<T,INPUT_T>()) {
        cudaMemcpy((void*)this->elements,(void*)elements,count*sizeof(T),
                   cudaMemcpyDefault);
      } else {
        std::vector<T> tmp(count);
        for (int i=0;i<count;i++)
          tmp[i] = T(elements[i]);
        cudaMemcpy((void*)this->elements,(void*)tmp.data(),count*sizeof(T),
                   cudaMemcpyDefault);
      }
      CUBQL_CUDA_SYNC_CHECK();
      this->needsCudaFree = true;
    }

    template<typename T, typename INPUT_T> inline
    AutoUploadArray<T,INPUT_T>::~AutoUploadArray() {
      if (needsCudaFree)
        cudaFree((void*)elements);
      CUBQL_CUDA_SYNC_CHECK();
    }
#endif
    
  }
} // ::dprt


  
