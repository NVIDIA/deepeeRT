// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprt/cuBQL/CuBQLBackend.h"
#include "dprt/Triangles.h"

namespace dprt {
  namespace cubql_cuda {

    /*! a single triangle mesh; can be created over pointes that are
        either on host or device, but which definitively stores
        vertices on the device */
    struct TriangleMesh : public dprt::TriangleMesh {
      struct DD {
        inline __cubql_both impl_triangle_t getTriangle(uint32_t primID) const;
        
        const impl_vec_t *vertices;
        const vec3i      *indices;
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
    
      AutoUploadArray<impl_vec_t,vec3d> vertices;
      AutoUploadArray<vec3i,vec3i> indices;
    };

    
    /*! a group/acceleration structure over one or more triangle meshes */
    struct TrianglesGroup : public dprt::TrianglesGroup {
      /*! device data for a cubql group over one or more triangle
          meshes */
      struct DD {
        /*! return the triangle specified by the given primref */
        inline __cubql_both DD() = default;
        inline __cubql_both impl_triangle_t getTriangle(PrimRef prim) const;
        
        TriangleMesh::DD *meshes;
        PrimRef          *primRefs;
        impl_bvh_t        bvh;
      };
      
      TrianglesGroup(Context *context,
                     const std::vector<dprt::TriangleMesh *> &geoms);
      ~TrianglesGroup() override;


      DD getDD() const
      {
        DD dd;
        dd.meshes = d_meshDDs;
        dd.primRefs = d_primRefs;
        dd.bvh = bvh;
        return dd;
      }
      
      impl_bvh_t        bvh;
      TriangleMesh::DD *d_meshDDs;
      PrimRef          *d_primRefs;
    };

    inline __cubql_both
    impl_triangle_t TriangleMesh::DD::getTriangle(uint32_t primID) const
    {
      vec3i idx = indices[primID];
      impl_triangle_t tri;
      tri.a = vertices[idx.x];
      tri.b = vertices[idx.y];
      tri.c = vertices[idx.z];
      return tri;
    }

    inline __cubql_both
    impl_triangle_t TrianglesGroup::DD::getTriangle(PrimRef prim) const
    {
      const TriangleMesh::DD &mesh = meshes[prim.geomID];
      impl_triangle_t tri = mesh.getTriangle(prim.primID);
      return tri;
    }
    
  }
} // ::dprt
