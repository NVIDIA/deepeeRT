// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepee/deepee.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/linear.h"
#include <cuda_runtime.h>

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace miniapp {
  using namespace cuBQL;

  struct Mesh {
    box3d bounds();
    vec3d center();
    void load_binmesh(const std::string &fileName);
    void translate(vec3d delta);
    void upload(vec3d *&d_vertices,
                vec3i *&d_indices);
    std::vector<vec3d> vertices;
    std::vector<vec3i> indices;
  };
  
  Mesh generateTesselatedQuad(int res,
                              vec3d dx,
                              vec3d dy,
                              vec3d dz,
                              double scale);
}

