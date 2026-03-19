// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprt/common.h"
#include "dprt/Backend.h"
#include <memory>

namespace dprt {

  struct InstanceGroup;
  struct TriangleMesh;
  struct TrianglesGroup;
  
  struct Context {
    static Context *create(int gpuID);
    
    Context(int gpuID);

    /*! creates a 'world' as a grouping of triangle mesh groups, with
        associated object-to-world space instance
        transforms. Implements the dprTrace() API function */
    virtual dprt::InstanceGroup *
    createInstanceGroup(const std::vector<dprt::TrianglesGroup *> &groups,
                        const DPRTAffine *transforms) = 0;

    /*! creates an object that represents a single triangle
        mesh. implements `dprCreateTrianglesDP()` */
    virtual dprt::TriangleMesh *
    createTriangleMesh(uint64_t         userData,
                       const vec3d     *vertexArray,
                       int              vertexCount,
                       const vec3i     *indexArray,
                       int              indexCount) = 0;
    
    /*! creates an object that represents a group of multiple triangle
        meshes that can then get instantiated. implements
        `dprCreateTrianglesGroup()` */
    virtual dprt::TrianglesGroup *
    createTrianglesGroup(const std::vector<dprt::TriangleMesh *> &geoms) = 0;
    
    
    /*! the cuda gpu ID that this device is going to run on */
    int const gpuID;
    /*! openmp host id. in theory only need this for the openmp
        backend, but don't want to have this as a conditional compile
        in the parent class that might get derived in different
        backends */
    int hostID = -1;
  };
  
} // ::dp

