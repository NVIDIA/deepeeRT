// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace miniapp {
  
  Camera generateCamera(vec2i imageRes,
                        const box3d &bounds,
                        const vec3d &from_dir,
                        const vec3d &up)
  {
    Camera camera;
    vec3d target = bounds.center();
    vec3d from = target + from_dir;
    vec3d direction = target-from;
    
    vec3d du = normalize(cross(direction,up));
    vec3d dv = normalize(cross(du,direction));
    
    double aspect = imageRes.x/double(imageRes.y);
    double scale = length(bounds.size());

    dv *= scale;
    du *= scale*aspect;
    
#if 0
    camera.direction.v = normalize(direction);
    camera.direction.du = 0.;
    camera.direction.dv = 0.;

    camera.origin.v = from-.5*du-.5*dv;
    camera.origin.du = du * (1./imageRes.x);
    camera.origin.dv = dv * (1./imageRes.y);
#else
    camera.direction.v = direction-.5*du-.5*dv;
    camera.direction.du = du * (1./imageRes.x);
    camera.direction.dv = dv * (1./imageRes.y);

    camera.origin.v = from;
    camera.origin.du = 0.;
    camera.origin.dv = 0.;
#endif
    return camera;
  }

}
