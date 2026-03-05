// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "OWLBackend.h"

extern "C" char devCode_ptx[];

namespace dprt {
  namespace owl {

    OWLBackend::OWLBackend(int gpuID)
      : Context(gpuID)
    {
      owl = owlContextCreate(&gpuID,1);

      OWLModule module = owlModuleCreate(owl,devCode_ptx);
      owlBuildPrograms(owl);
      
      OWLVarDecl rgVars[] = {
        { nullptr },
      };
      rg = owlRayGenCreate(owl,module,"raygen",
                           0,
                           rgVars,-1);

      OWLVarDecl lpVars[] = {
        { "model", OWL_GROUP, OWL_OFFSETOF(LaunchParams,model) },
        { "rays", OWL_ULONG, OWL_OFFSETOF(LaunchParams,rays) },
        { "hits", OWL_ULONG, OWL_OFFSETOF(LaunchParams,hits) },
        { "flags", OWL_ULONG, OWL_OFFSETOF(LaunchParams,flags) },
        { "numRays", OWL_INT, OWL_OFFSETOF(LaunchParams,numRays) },
        { nullptr },
      };
      lp = owlParamsCreate(owl,sizeof(LaunchParams),
                           lpVars,-1);
      
      OWLVarDecl gtVars[] = {
        { "vertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh::DD,vertices) },
        { "indices",  OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh::DD,indices) },
        { "userData", OWL_ULONG,  OWL_OFFSETOF(TriangleMesh::DD,userData) },
        { nullptr },
      };
      trianglesGT = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                             sizeof(TriangleMesh::DD),
                             gtVars,-1);
      owlGeomTypeSetClosestHit(trianglesGT,0,module,"TriMesh");
      
      owlBuildPrograms(owl);
      owlBuildPipeline(owl);
    }

    OWLBackend::~OWLBackend()
    {
      owlContextDestroy(owl);
    }

    TriangleMesh::TriangleMesh(OWLBackend *be,
                               uint64_t         userData,
                               const vec3d     *vertexArray,
                               int              vertexCount,
                               const vec3i     *indexArray,
                               int              indexCount)
      : dprt::TriangleMesh(be,userData,
                           vertexArray,
                           vertexCount,
                           indexArray,
                           indexCount)
    {
      std::vector<vec3f> vtx;
      for (int i=0;i<vertexCount;i++)
        vtx.push_back(vec3f(vertexArray[i]));
      OWLBuffer vertexBuffer
        = owlDeviceBufferCreate(be->owl,OWL_FLOAT3,
                                vertexCount,vtx.data());
      OWLBuffer indexBuffer
        = owlDeviceBufferCreate(be->owl,OWL_INT3,
                                indexCount,indexArray);
      geom = owlGeomCreate(be->owl,be->trianglesGT);
      owlTrianglesSetVertices(geom,vertexBuffer,
                              vertexCount,sizeof(vec3f),0);
      owlTrianglesSetIndices(geom,indexBuffer,
                              indexCount,sizeof(vec3i),0);
      owlGeomSet1ul(geom,"userData",userData);
      owlGeomSetBuffer(geom,"vertices",vertexBuffer);
      owlGeomSetBuffer(geom,"indices",indexBuffer);
    }


    TrianglesGroup::TrianglesGroup(OWLBackend *be,
                                   const std::vector<dprt::TriangleMesh *> &_geoms)
      : dprt::TrianglesGroup(be,_geoms)
    {
      std::vector<OWLGeom> geoms;
      for (auto g : _geoms)
        geoms.push_back(((TriangleMesh*)g)->geom);
      group = owlTrianglesGeomGroupCreate(be->owl,
                                          geoms.size(),geoms.data());
      owlGroupBuildAccel(group);
    }
    
    InstanceGroup::InstanceGroup(OWLBackend *be,
                                 const std::vector<dprt::TrianglesGroup *> &groups,
                                 const DPRTAffine *transforms)
      : dprt::InstanceGroup(be,groups,transforms), be(be)
    {
      std::vector<OWLGroup> _groups;
      for (auto g : groups) 
        _groups.push_back(((TrianglesGroup*)g)->group);
      
      std::vector<affine3f> xfms;
      if (transforms)
        for (int i=0;i<groups.size();i++) {
          auto in = transforms[i];
          affine3f out;
          out.p = vec3f((vec3d&)in.p);
          out.l.vx = vec3f((vec3d&)in.l.vx);
          out.l.vy = vec3f((vec3d&)in.l.vy);
          out.l.vz = vec3f((vec3d&)in.l.vz);
          xfms.push_back(out);
        }
      group = owlInstanceGroupCreate(be->owl,
                                     _groups.size(),
                                     _groups.data(),
                                     nullptr,
                                     (float*)(transforms?xfms.data():0));
      owlGroupBuildAccel(group);

      owlBuildPipeline(be->owl);
      owlBuildSBT(be->owl);
      owlBuildPipeline(be->owl);
      owlBuildSBT(be->owl);
    }
    
    dprt::TriangleMesh *
    OWLBackend::createTriangleMesh(uint64_t         userData,
                                   const vec3d     *vertexArray,
                                   int              vertexCount,
                                   const vec3i     *indexArray,
                                   int              indexCount)
    {
      return new TriangleMesh(this,userData,
                              vertexArray,vertexCount,
                              indexArray,indexCount);
    }

    dprt::TrianglesGroup *
    OWLBackend
    ::createTrianglesGroup(const std::vector<dprt::TriangleMesh *> &geoms) 
    {
      return new TrianglesGroup(this,geoms);
    }

    dprt::InstanceGroup *
    OWLBackend
    ::createInstanceGroup(const std::vector<dprt::TrianglesGroup *> &groups,
                          const DPRTAffine *transforms)
    {
      return new InstanceGroup(this,groups,transforms);
    }

    /*! implements dprTrace() */
    void InstanceGroup::traceRays(DPRTRay *d_rays,
                                  DPRTHit *d_hits,
                                  int numRays,
                                  uint64_t flags)
    {
      auto lp = be->lp;
      owlParamsSet1i(lp,"numRays",numRays);
      owlParamsSet1ul(lp,"flags",flags);
      owlParamsSet1ul(lp,"rays",(uint64_t)d_rays);
      owlParamsSet1ul(lp,"hits",(uint64_t)d_hits);
      owlParamsSetGroup(lp,"model",group);
      // owlParamsSetGroup(lp,"model",owlGroupGetTraversable(group,0));
      owlLaunch2D(be->rg,1024,divRoundUp(numRays,1024),lp);
    }
    
  }
  
  Context *Context::create(int gpuID)
  {
    return new owl::OWLBackend(gpuID);
  };
  
}


