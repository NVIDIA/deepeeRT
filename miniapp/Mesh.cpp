// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#include <fstream>
#include <random>
#include <cuBQL/math/affine.h>

namespace miniapp {
  using cuBQL::affine3d;
  
  box3d Mesh::bounds()
  {
    box3d bb;
    for (auto v : vertices)
      bb.extend(v);
    return bb;
  }

  Mesh generateTessellatedQuad(int res,
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
        mesh.vertices.push_back(scale*x*dx+scale*y*dy+scale*z*dz);
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

  void Mesh::load(const std::string &fileName)
  {
    const std::string ext = extensionOf(fileName);
    if (ext == ".obj")
      load_obj(fileName);
    else if (ext == ".binmesh")
      load_binmesh(fileName);
    else if (ext == ".dgef")
      load_dgef(fileName);
    else
      throw std::runtime_error("un-recognized or un-supported file extension '"+ext+"'");
  }
  
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
  

  void Mesh::load_dgef(const std::string &fileName)
  {
    vertices.clear();
    indices.clear();
    
    std::ifstream in(fileName.c_str(),std::ios::binary);

    size_t header;
    in.read((char*)&header,sizeof(header));

    size_t numMeshes;
    in.read((char*)&numMeshes,sizeof(numMeshes));
    std::vector<Mesh> meshes(numMeshes);
    for (int meshID=0;meshID<numMeshes;meshID++) {
      Mesh &mesh = meshes[meshID];
      
      size_t count;
      in.read((char*)&count,sizeof(count));
      mesh.vertices.resize(count);

      in.read((char*)mesh.vertices.data(),
              count*sizeof(vec3d));
      
      in.read((char*)&count,sizeof(count));
      for (size_t i=0;i<count;i++) {
        vec3ul idx;
        in.read((char*)&idx,sizeof(idx));
        mesh.indices.push_back({(int)idx.x,(int)idx.y,(int)idx.z});
      }
    }
    
    size_t numInstances;
    in.read((char*)&numInstances,sizeof(numInstances));
    for (int instID=0;instID<numInstances;instID++) {
      affine3d xfm;
      in.read((char*)&xfm,sizeof(xfm)); 
      size_t meshID;
      in.read((char*)&meshID,sizeof(meshID)); 
    }

    this->vertices = meshes[0].vertices;
    this->indices = meshes[0].indices;
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
