// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file deepeeRT.cpp Implements the primer/primer.h API functions */

#include "dp/Context.h"
#include "dp/Triangles.h"
#include "dp/Group.h"
#include "dp/World.h"
#include "dp/cuBQL/CuBQLBackend.h"

namespace dp {

  Context *Context::create(int gpuID)
  {
    return new cubql_cuda::CuBQLCUDABackend(gpuID);
  };
  
} // ::dp

DPR_API
DPRContext dprContextCreate(DPRContextType contextType,
                            int gpuToUse)
{
  return (DPRContext)dp::Context::create(gpuToUse);
}

DPR_API
DPRTriangles dprCreateTrianglesDP(DPRContext _context,
                                  /*! a 64-bit user-provided data that
                                    gets attached to this mesh; this is
                                    what gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t userData,
                                  /*! device array of vertices */
                                  DPRvec3 *vertexArray,
                                  size_t   vertexCount,
                                  /*! device array of int3 vertex indices */
                                  DPRint3 *indexArray,
                                  size_t   indexCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  return (DPRTriangles)context->
    createTriangleMesh(userData,
                       (const dp::vec3d*)vertexArray,
                       vertexCount,
                       (const dp::vec3i*)indexArray,
                       indexCount);
}

DPR_API
DPRGroup dprCreateTrianglesGroup(DPRContext   _context,
                                 DPRTriangles *triangleGeomsArray,
                                 size_t        triangleGeomsCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  std::vector<dp::TriangleMesh*> geoms;
  for (int i=0;i<(int)triangleGeomsCount;i++) {
    dp::TriangleMesh *geom = (dp::TriangleMesh *)triangleGeomsArray[i];
    assert(geom);
    assert(geom->context == context);
    geoms.push_back(geom);
  }
  return (DPRGroup)context->createTrianglesGroup(geoms);
}

DPR_API
DPRWorld dprCreateWorldDP(DPRContext _context,
                          DPRGroup   *instanceGroups,
                          DPRAffine  *instanceTransforms,
                          size_t      instanceCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  
  std::vector<dp::TrianglesGroup *> groups;
  for (int i=0;i<(int)instanceCount;i++) {
    dp::TrianglesGroup *group = (dp::TrianglesGroup *)instanceGroups[i];
    assert(group);
    groups.push_back(group);
  }
  return (DPRWorld)context->
    createInstanceGroup(groups,instanceTransforms);
}

DPR_API
void dprTrace(/*! the world we want the rays to be traced against */
              DPRWorld _world,
              /*! *device* array of rays that need tracing */
              DPRRay *d_rays,
              /*! *device* array of where to store the hits */
              DPRHit *d_hits,
              /*! number of rays that need tracing. d_rays and
                d_hits *must* have (at least) that many entires */
              int numRays,
              uint64_t flags)
{
  dp::InstanceGroup *world = (dp::InstanceGroup *)_world;
  assert(world);
  assert(d_hits);
  assert(d_rays);
  assert(numRays > 0);
  world->traceRays(d_rays,d_hits,numRays,flags);
}

DPR_API void dprFreeWorld(DPRWorld world)
{
  assert(world);
  delete (dp::InstanceGroup *)world;
}
    
DPR_API void dprFreeTriangles(DPRTriangles triangles)
{
  assert(triangles);
  delete (dp::TriangleMesh *)triangles;
}
    
DPR_API void dprFreeGroup(DPRGroup group)
{
  assert(group);
  delete (dp::TrianglesGroup *)group;
}
    
DPR_API void dprFreeContext(DPRContext context)
{
  assert(context);
  delete (dp::Context *)context;
}
    



