// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepee/deepee.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/linear.h"
#include <cuda_runtime.h>

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace dp {
  using namespace ::cuBQL;

  /*! helper class that sets the active cuda device to the given gpuID
      for the lifetime of this class, and restores it to whatever it
      was after that variable dies */
  struct SetActiveGPU {
    SetActiveGPU(int gpuID) { cudaGetDevice(&savedActive); cudaSetDevice(gpuID); }
    ~SetActiveGPU() { cudaSetDevice(savedActive); }
    int savedActive = -1;
  };

  inline __cubql_both float abst(float f)   { return (f < 0.f) ? -f : f; }
  inline __cubql_both double abst(double f) { return (f < 0. ) ? -f : f; }

  template<typename T>
  bool isDevicePointer(T *ptr)
  {
    cudaPointerAttributes attributes = {};
    // do NOT check for error: in CUDA<10, passing a host pointer will
    // actually create a cuda error, so let's just call and then
    // ignore/clear the error
    cudaPointerGetAttributes(&attributes,(const void *)ptr);
    cudaGetLastError();
      
    return attributes.devicePointer != 0;
  }

}
