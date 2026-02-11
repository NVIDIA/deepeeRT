#pragma once

#include "dp/cuBQL/CuBQLBackend.h"
#include "dp/Triangles.h"

namespace dp {
  namespace cubql_cuda {

    /*! a single triangle mesh; can be created over pointes that are
        either on host or device, but which definitively stores
        vertices on the device */
    struct TriangleMesh : public dp::TriangleMesh {
      struct DD {
        inline __cubql_both TriangleDP getTriangle(uint32_t primID) const;
        
        const vec3d   *vertices;
        const vec3i   *indices;
        uint64_t userData;
      };

      TriangleMesh(Context         *context,
                   uint64_t         userData,
                   const vec3d     *vertexArray,
                   int              vertexCount,
                   const vec3i     *indexArray,
                   int              indexCount);
      
      DD getDD() const
      { return { vertices.elements, indices.elements, userData }; }
    
      AutoUploadArray<vec3d> vertices;
      AutoUploadArray<vec3i> indices;
    };

    
    /*! a group/acceleration structure over one or more triangle meshes */
    struct TrianglesGroup : public dp::TrianglesGroup {
      /*! device data for a cubql group over one or more triangle
          meshes */
      struct DD {
        /*! return the triangle specified by the given primref */
        inline __cubql_both DD() = default;
        inline __cubql_both TriangleDP getTriangle(PrimRef prim) const;
        
        TriangleMesh::DD *meshes;
        PrimRef          *primRefs;
        bvh3d             bvh;
      };
      
      TrianglesGroup(Context *context,
                     const std::vector<dp::TriangleMesh *> &geoms);
      ~TrianglesGroup() override;


      DD getDD() const
      {
        DD dd;
        dd.meshes = d_meshDDs;
        dd.primRefs = d_primRefs;
        dd.bvh = bvh;
        return dd;
      }
      
      bvh3d             bvh;
      TriangleMesh::DD *d_meshDDs;
      PrimRef          *d_primRefs;
    };

    inline __cubql_both
    TriangleDP TriangleMesh::DD::getTriangle(uint32_t primID) const
    {
      vec3i idx = indices[primID];
      TriangleDP tri;
      tri.a = vertices[idx.x];
      tri.b = vertices[idx.y];
      tri.c = vertices[idx.z];
      return tri;
    }

    inline __cubql_both
    TriangleDP TrianglesGroup::DD::getTriangle(PrimRef prim) const
    {
      const TriangleMesh::DD &mesh = meshes[prim.geomID];
      TriangleDP tri = mesh.getTriangle(prim.primID);
      return tri;
    }
    
  }
}
