// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file dprt.cpp Implements the dprt/dprt.h API functions */

#include "dprt/Context.h"
#include "dprt/Group.h"
#include "dprt/Triangles.h"
#include "dprt/Instances.h"
#include "dprt/cuBQL/CuBQLBackend.h"

namespace dprt {

  Context *Context::create(int gpuID)
  {
    return new cubql_cuda::CuBQLCUDABackend(gpuID);
  };
  
} // ::dprt

DPRT_API
DPRTContext dprtContextCreate(DPRTContextType contextType,
                              int gpuToUse)
{
  return (DPRTContext)dprt::Context::create(gpuToUse);
}

DPRT_API
DPRTTriangles dprtCreateTriangles(DPRTContext _context,
                                  /*! a 64-bit user-provided data that
                                    gets attached to this mesh; this is
                                    what gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t userData,
                                  /*! device array of vertices */
                                  DPRTvec3 *vertexArray,
                                  size_t   vertexCount,
                                  /*! device array of int3 vertex indices */
                                  DPRTint3 *indexArray,
                                  size_t   indexCount)
{
  dprt::Context *context = (dprt::Context *)_context;
  assert(context);
  return (DPRTTriangles)context->
    createTriangleMesh(userData,
                       (const dprt::vec3d*)vertexArray,
                       vertexCount,
                       (const dprt::vec3i*)indexArray,
                       indexCount);
}

DPRT_API
DPRTGroup dprtCreateTrianglesGroup(DPRTContext   _context,
                                   DPRTTriangles *triangleGeomsArray,
                                   size_t        triangleGeomsCount)
{
  dprt::Context *context = (dprt::Context *)_context;
  assert(context);
  std::vector<dprt::TriangleMesh*> geoms;
  for (int i=0;i<(int)triangleGeomsCount;i++) {
    dprt::TriangleMesh *geom = (dprt::TriangleMesh *)triangleGeomsArray[i];
    assert(geom);
    assert(geom->context == context);
    geoms.push_back(geom);
  }
  return (DPRTGroup)context->createTrianglesGroup(geoms);
}

DPRT_API
DPRTModel dprtCreateModel(DPRTContext _context,
                            DPRTGroup   *instanceGroups,
                            DPRTAffine  *instanceTransforms,
                            size_t      instanceCount)
{
  dprt::Context *context = (dprt::Context *)_context;
  assert(context);
  
  std::vector<dprt::TrianglesGroup *> groups;
  for (int i=0;i<(int)instanceCount;i++) {
    dprt::TrianglesGroup *group = (dprt::TrianglesGroup *)instanceGroups[i];
    assert(group);
    groups.push_back(group);
  }
  return (DPRTModel)context->
    createInstanceGroup(groups,instanceTransforms);
}

DPRT_API
void dprtTrace(/*! the model we want the rays to be traced against */
               DPRTModel _model,
               /*! *device* array of rays that need tracing */
               DPRTRay *d_rays,
               /*! *device* array of where to store the hits */
               DPRTHit *d_hits,
               /*! number of rays that need tracing. d_rays and
                 d_hits *must* have (at least) that many entires */
               int numRays,
               uint64_t flags)
{
  dprt::InstanceGroup *model = (dprt::InstanceGroup *)_model;
  assert(model);
  assert(d_hits);
  assert(d_rays);
  assert(numRays > 0);
  model->traceRays(d_rays,d_hits,numRays,flags);
}

DPRT_API void dprtFreeModel(DPRTModel model)
{
  assert(model);
  delete (dprt::InstanceGroup *)model;
}
    
DPRT_API void dprtFreeTriangles(DPRTTriangles triangles)
{
  assert(triangles);
  delete (dprt::TriangleMesh *)triangles;
}
    
DPRT_API void dprtFreeGroup(DPRTGroup group)
{
  assert(group);
  delete (dprt::TrianglesGroup *)group;
}
    
DPRT_API void dprtFreeContext(DPRTContext context)
{
  assert(context);
  delete (dprt::Context *)context;
}
    



