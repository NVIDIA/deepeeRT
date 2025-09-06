// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

/*! \file deepeeRT.cpp Implements the primer/primer.h API functions */

#include "dp/Context.h"
#include "dp/Triangles.h"
#include "dp/Group.h"
#include "dp/World.h"

namespace dp {
} // ::dp

DPR_API
DPRContext dprContextCreate(DPRContextType contextType,
                            int gpuToUse)
{
  return (DPRContext)new dp::Context(gpuToUse);
}

DPR_API
DPRTriangles dprCreateTrianglesDP(DPRContext _context,
                                  /*! a 64-bit user-provided data that
                                    gets attahed to this mesh; this is
                                    waht gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t userData,
                                  /*! device array of vertices */
                                  DPRvec3 *d_vertexArray,
                                  size_t   vertexCount,
                                  /*! device array of int3 vertex indices */
                                  DPRint3 *d_indexArray,
                                  size_t   indexCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  return (DPRTriangles)new dp::TrianglesDP(context,
                                           userData,
                                           (const dp::vec3d*)d_vertexArray,
                                           vertexCount,
                                           (const dp::vec3i*)d_indexArray,
                                           indexCount);
}

DPR_API
DPRGroup dprCreateTrianglesGroup(DPRContext   _context,
                                 DPRTriangles *triangleGeomsArray,
                                 size_t        triangleGeomsCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  std::vector<dp::TrianglesDP*> geoms;
  for (int i=0;i<(int)triangleGeomsCount;i++) {
    dp::TrianglesDP *geom = (dp::TrianglesDP *)triangleGeomsArray[i];
    assert(geom);
    assert(geom->context == context);
    geoms.push_back(geom);
  }
  return (DPRGroup)new dp::TrianglesDPGroup(context,geoms);
}


DPR_API
DPRWorld dprCreateWorldDP(DPRContext _context,
                          DPRGroup   *instanceGroups,
                          DPRAffine  *d_instanceTransforms,
                          size_t      instanceCount)
{
  dp::Context *context = (dp::Context *)_context;
  assert(context);
  
  assert(instanceCount == 1 && "instancing not yet implemented");
  assert(d_instanceTransform == nullptr && "instancing not yet implemented");
  
  std::vector<dp::Group *> groups;
  for (int i=0;i<(int)instanceCount;i++) {
    dp::Group *group = (dp::Group *)instanceGroups[i];
    assert(group);
    groups.push_back(group);
  }
  return (DPRWorld)new dp::InstancesDPGroup(context,groups,d_instanceTransforms);
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
              int numRays)
{
  dp::InstancesDPGroup *world = (dp::InstancesDPGroup *)_world;
  assert(world);
  assert(d_hits);
  assert(d_rays);
  assert(numRays > 0);
  world->traceRays(d_rays,d_hits,numRays);
}



