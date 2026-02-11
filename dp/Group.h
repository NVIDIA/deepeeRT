// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/common.h"

namespace dp {

  struct Context;

  /*! abstract base class of a group of one or more intersectable
      things that share an acceleration structures. will be subclassed
      into InstancesGroup and TrianglesGroup, and then implemented in
      each backend based on how this backend works */
  struct Group {
    Group(Context *const context);
    virtual ~Group() = default;
    
    /*! context that this group was created in */
    Context *const context;
  };

} // ::dp

