// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace miniapp {
  
  Camera generateCamera(vec2i imageRes,
                        const box3d &bounds,
                        const vec3d &from,
                        const vec3d &up)
  {
    Camera camera;
    vec3d target = bounds.center();
    vec3d direction = normalize(target-from); 
    camera.direction.v = direction;
    camera.direction.du = 0.;
    camera.direction.dv = 0.;

    vec3d du = normalize(cross(direction,up));
    vec3d dv = normalize(cross(du,direction));

    double aspect = imageRes.x/double(imageRes.y);
    double scale = length(bounds.size());
    dv *= scale;
    du *= scale*aspect;
    camera.origin.v = from-.5*du-.5*dv;
    camera.origin.du = du * (1./imageRes.x);
    camera.origin.dv = dv * (1./imageRes.y);
    return camera;
  }

}
