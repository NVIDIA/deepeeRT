// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Context.h"

namespace dp {
  
  Context::Context(int gpuID)
    : gpuID(gpuID)
  {
    cudaSetDevice(gpuID);
    cudaFree(0);
  }
  
} // ::dpr

