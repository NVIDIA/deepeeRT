// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Backend.h"
#include "dp/Context.h"

namespace dp {

  Backend::Backend(Context *const context)
    : context(context),
      gpuID(context->gpuID)
  {}

}
