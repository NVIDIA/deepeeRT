// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#include <fstream>
#include <random>

namespace miniapp {

  box3d Mesh::bounds()
  {
    box3d bb;
    for (auto v : vertices)
      bb.extend(v);
    return bb;
  }

  Mesh generateTesselatedQuad(int res,
                              vec3d dx,
                              vec3d dy,
                              vec3d dz,
                              double scale)
  {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng(-.5/res,+.5/res);
    Mesh mesh;
    for (int iy=0;iy<=res;iy++)
      for (int ix=0;ix<=res;ix++) {
        double x = -1.+2.*ix/res;
        double y = -1.+2.*iy/res;
        double z = rng(gen);
        mesh.vertices.push_back(x*dx+y*dy+z*dz);
      }
    
    for (int iy=0;iy<res;iy++)
      for (int ix=0;ix<res;ix++) {
        int i00=(ix+0)+(iy+0)*(res+1);
        int i01=(ix+1)+(iy+0)*(res+1);
        int i10=(ix+0)+(iy+1)*(res+1);
        int i11=(ix+1)+(iy+1)*(res+1);
        mesh.indices.push_back({i00,i01,i11});
        mesh.indices.push_back({i00,i11,i10});
      }
    return mesh;
  }

  vec3d Mesh::center()
  { return bounds().center(); }
  
  void Mesh::load_binmesh(const std::string &fileName)
  {
    vertices.clear();
    indices.clear();
    
    std::ifstream in(fileName.c_str(),std::ios::binary);

    size_t numVertices;
    in.read((char*)&numVertices,sizeof(numVertices));
    std::vector<vec3f> floatVertices;
    floatVertices.resize(numVertices);
    in.read((char*)floatVertices.data(),numVertices*sizeof(floatVertices[0]));
    for (auto v : floatVertices)
      vertices.push_back(vec3d(v));
    
    size_t numIndices;
    in.read((char*)&numIndices,sizeof(numIndices));
    indices.resize(numIndices);
    in.read((char*)indices.data(),numIndices*sizeof(indices[0]));
  }
  
  void Mesh::translate(vec3d delta)
  {
    for (auto &v : vertices)
      v = v + delta;
  }
  
  void Mesh::upload(vec3d *&d_vertices,
                    vec3i *&d_indices)
  {
    cudaMalloc((void **)&d_vertices,vertices.size()*sizeof(*d_vertices));
    cudaMemcpy(d_vertices,vertices.data(),vertices.size()*sizeof(*d_vertices),
               cudaMemcpyDefault);
    cudaMalloc((void **)&d_indices,indices.size()*sizeof(*d_indices));
    cudaMemcpy(d_indices,indices.data(),indices.size()*sizeof(*d_indices),
               cudaMemcpyDefault);
  }
}
