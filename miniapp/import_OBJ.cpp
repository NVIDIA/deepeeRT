// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../3rdParty/tinyOBJ/tiny_obj_loader.h"

namespace miniapp {

#ifdef _WIN32
constexpr char path_sep = '\\';
#else
constexpr char path_sep = '/';
#endif
  
  std::string pathOf(const std::string &filepath)
  {
    size_t pos = filepath.find_last_of(path_sep);
    if (pos == std::string::npos)
      return "";
    return filepath.substr(0, pos + 1);
  }

  std::string fileOf(const std::string &filepath)
  {
    size_t pos = filepath.find_last_of(path_sep);
    if (pos == std::string::npos)
      return "";
    return filepath.substr(pos + 1, filepath.size());
  }

  std::string extensionOf(const std::string &filepath)
  {
    size_t pos = filepath.rfind('.');
    if (pos == filepath.npos)
      return "";
    return filepath.substr(pos);
  }


  struct OBJData
  {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
  };

  void Mesh::load_obj(const std::string &objFile)
  {
    const std::string modelDir = pathOf(objFile);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    std::cout << "reading OBJ file '" << objFile << " from directory '" << modelDir << "'" << std::endl;
    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
                         &err,
                         objFile.c_str(),
                         modelDir.c_str(),
                         /* triangulate */true);
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);
    }

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];
      
      for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
        if (shape.mesh.num_face_vertices[faceID] != 3)
          throw std::runtime_error("not properly tessellated");
        tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
        tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
        tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];

        indices.push_back(int(vertices.size())+vec3i(0,1,2));
        vertices.push_back(vec3d(vertex_array[idx0.vertex_index]));
        vertices.push_back(vec3d(vertex_array[idx1.vertex_index]));
        vertices.push_back(vec3d(vertex_array[idx2.vertex_index]));
      }
    }
  }

} // namespace tsd
