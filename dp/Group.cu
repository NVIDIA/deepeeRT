// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "dp/Group.h"
#include "dp/Context.h"

namespace dp {

  TrianglesDPGroup::TrianglesDPGroup(Context *context,
                                     const std::vector<TrianglesDP *> &geoms)
    : context(context),
      geoms(geoms)
  {
    impl = context->backend->createTrianglesDPImpl(this);
  }

    // template<typename T>
    // __global__ void fillBoxes(owl::common::box_t<vec_t<T,3>> *boxes,
    //                           DevMesh mesh)
    // {
    //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
    //   if (tid >= mesh.numIndices) return;
    //   auto indices = mesh.indices[tid];
    //   owl::common::box3f bb = owl::common::box3f()
    //     .including(mesh.vertices[indices.x])
    //     .including(mesh.vertices[indices.y])
    //     .including(mesh.vertices[indices.z]);
    //   boxes[tid] = { vec_t<T,3>(bb.lower),
    //                  vec_t<T,3>(bb.upper) };
    // }
  
    // template<typename T>
    // owl::common::box_t<vec_t<T,3>> *makeBoxes(DevMesh::SP mesh)
    // {
    //   box_t<vec_t<T,3>> *d_boxes;
    //   cudaMalloc((void **)&d_boxes,mesh->numIndices*sizeof(*d_boxes));
    //   int bs = 128;
    //   int nb = divRoundUp(mesh->numIndices,bs);
    //   fillBoxes<T><<<nb,bs>>>(d_boxes,*mesh);
    //   return d_boxes;
    // }
  
  
  
} // ::dp

