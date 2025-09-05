// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dp/Backend.h"
#include "dp/Group.h"
#include <cuBQL/bvh.h>

namespace dp {
  
  struct CuBQLCUDABackend : public dp::Backend
  {
    CuBQLCUDABackend(Context *const context);
    virtual ~CuBQLCUDABackend() = default;
    
    virtual std::shared_ptr<InstancesDPImpl>
    createInstancesDPImpl(dp::InstancesDPGroup *fe) override;
    
    virtual std::shared_ptr<TrianglesDPImpl>
    createTrianglesDPImpl(dp::TrianglesDPGroup *fe) override;
  };

}


  
