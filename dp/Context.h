// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Backend.h"

namespace dp {
  
  struct Context {
    Context(int gpuID);

    std::shared_ptr<Backend> backend;
    int const gpuID;
  };
  
} // ::dp

