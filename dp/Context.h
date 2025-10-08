// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"
#include "dp/Backend.h"

namespace dp {
  
  struct Context {
    Context(int gpuID);

    std::shared_ptr<Backend> backend;
    /*! the cuda gpu ID that this device is going to run on */
    int const gpuID;
  };
  
} // ::dp

