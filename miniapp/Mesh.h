// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    void load(const std::string &fileName);
    void translate(vec3d delta);
    void upload(vec3d *&d_vertices,
                vec3i *&d_indices);
    std::vector<vec3d> vertices;
    std::vector<vec3i> indices;
  private:
    void load_binmesh(const std::string &fileName);
    void load_obj(const std::string &fileName);
  };
  
  /*! helper function that creates a mesh with a terrain-like shape,
      consisting of a height field of res*res squares (so 2*res*res
      triangles); spanning [-res..+res] in x and y dimension, and
      randomly chosen heights in a [-.5..+.5]*scale range; all
      trnasformed through a [dx,dy,dz] matrix to allow for additoinal
      scaling or coordinate frame rotation */
  Mesh generateTesselatedQuad(int res,
                              vec3d dx,
                              vec3d dy,
                              vec3d dz,
                              double scale);

  /*! returns the file extension of the given file name */
  std::string extensionOf(const std::string &fileName);
  
}

