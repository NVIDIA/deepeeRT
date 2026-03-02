// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dprt/Context.h"
#include "dprt/cuBQL/CuBQLBackend.h"

namespace dprt {
  
  Context::Context(int gpuID)
    : gpuID(gpuID)
  {}

} // ::dprt

