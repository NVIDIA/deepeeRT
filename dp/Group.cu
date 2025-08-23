// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Group.h"
#include "dp/Context.h"

namespace dp {

  TrianglesDPGroup::TrianglesDPGroup(Context *context,
                                     const std::vector<TrianglesDP *> &geoms)
    : context(context),
      geoms(geoms)
  {}
    
} // ::dp

