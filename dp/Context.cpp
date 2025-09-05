// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Context.h"
#include "dp/cuBQL/CuBQLBackend.h"

namespace dp {
  
  Context::Context(int gpuID)
    : gpuID(gpuID)
  {
    backend = std::make_shared<CuBQLCUDABackend>(this);
  }
  
} // ::dpr

