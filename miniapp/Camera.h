// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"

namespace miniapp {

  /*! this HAS to be the same data layout as DPRRay in deepee.h */
  struct Ray {
    vec3d origin;
    vec3d direction;
    double  tMin;
    double  tMax;
  };

  inline __cubql_both dbgout operator<<(dbgout o, Ray ray)
  {
    o << "Ray{"<<ray.origin<<"+["<<ray.tMin<<","<<ray.tMax<<"]*"<<ray.direction<<"}";
    return o;
  }

  
  struct Camera {
    /*! generate ray for a given pixel/image plane coordinate. based on
      how the camera was created this could be either a orthogonal or
      a perspective camera (see Camera.cpp) */
    inline __device__ Ray generateRay(vec2d pixel, bool dbg=false) const;

    struct {
      vec3d v,du,dv;
    } origin, direction;
  };

  Camera generateCamera(vec2i imageRes,
                        const box3d &bounds,
                        const vec3d &position,
                        const vec3d &up);

  /*! generate ray for a given pixel/image plane coordinate. based on
      how the camera was created this could be either a orthogonal or
      a perspective camera (see Camera.cpp) */
  inline __device__
  Ray Camera::generateRay(vec2d pixel, bool dbg) const
  {
    Ray ray;
    ray.origin
      = origin.v+pixel.x*origin.du+pixel.y*origin.dv;
    ray.direction
      = normalize(direction.v+pixel.x*direction.du+pixel.y*direction.dv);
    ray.tMin = 0.;
    ray.tMax = INFINITY;
    return ray;
  }

}
