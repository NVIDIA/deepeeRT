// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"

namespace miniapp {

  /*! this HAS to be the same data layout as DPRRay in deepee.h */
  struct Ray {
    vec3d origin;
    vec3d direction;
    double  tmin;
    double  tmax;
  };
  
  struct Camera {
    inline __device__ Ray generateRay(vec2d pixel) const;
    struct {
      vec3d v,du,dv;
    } origin, direction;
  };

  Camera generateCamera(vec2i imageRes,
                        const box3d &bounds,
                        const vec3d &position,
                        const vec3d &up);


  inline __device__ Ray Camera::generateRay(vec2d pixel) const
  {
    Ray ray;
    ray.origin = origin.v+pixel.x*origin.du+pixel.y*origin.dv;
    ray.direction = normalize(direction.v+pixel.x*direction.du+pixel.y*direction.dv);
    ray.tmin = 0.;
    ray.tmax = INFINITY;
    return ray;
  }

}
