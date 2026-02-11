// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepeeRT/deepeeRT.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/linear.h"
#include "cuBQL/math/affine.h"
#include <cuda_runtime.h>

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace miniapp {
  using namespace cuBQL;
  namespace dgef {
    struct Mesh {
      std::vector<vec3d> vertices;
      std::vector<vec3i> indices;
    };
    struct Object {
      std::vector<Mesh *> meshes;
    };
    struct Instance {
      affine3d xfm;
      Object *object;
    };
    struct Scene {
      static Scene *load(const std::string &fileName);
      std::vector<Instance *> instances;
      box3d bounds() const;
    };
  }
}
