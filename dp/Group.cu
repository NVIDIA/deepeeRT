// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dp/Group.h"
#include "dp/Context.h"

namespace dp {

  TrianglesDPGroup::TrianglesDPGroup(Context *context,
                                     const std::vector<TrianglesDP *> &geoms)
    : context(context),
      geoms(geoms)
  {
    impl = context->backend->createTrianglesDPImpl(this);
  }

} // ::dp

