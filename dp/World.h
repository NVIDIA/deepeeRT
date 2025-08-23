// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Triangles.h"

namespace dp {

  struct Context;
  struct Group;
  
  struct World {
    World(Context *context,
          const std::vector<Group *> &groups,
          const DPRAffine            *d_transforms);

    std::vector<Group *> const groups;
    const DPRAffine     *const d_transforms;
    Context             *const context;
  };
    
} // ::dp

