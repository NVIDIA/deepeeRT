// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

/*! \file api/primer.cpp Implements the primer/primer.h API functions */

#include "dp/common.h"

namespace dp {
  
  struct Context {
    Context(int gpuID);

    int const gpuID;
  };
  
} // ::dp

