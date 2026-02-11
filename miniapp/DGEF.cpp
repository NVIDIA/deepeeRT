// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "DGEF.h"
#include <fstream>

namespace miniapp {
  namespace dgef {
    
    Scene *Scene::load(const std::string &fileName)
    {
      std::ifstream in(fileName.c_str(),std::ios::binary);

      std::vector<Mesh *> meshes;
      Scene *scene = new Scene;
      
      size_t header;
      in.read((char*)&header,sizeof(header));
      
      bool isProperDGEF = (header == 0xdefdefdefULL);
      size_t numMeshes;
      in.read((char*)&numMeshes,sizeof(numMeshes));
      for (int meshID=0;meshID<numMeshes;meshID++) {
        Mesh *_mesh = new Mesh;
        meshes.push_back(_mesh);
        Mesh &mesh = *_mesh;
      
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

      if (!isProperDGEF) {
        Object *object = new Object;
        object->meshes = meshes;
        Instance *inst = new Instance;
        inst->xfm = affine3d();
        inst->object = object;
        scene->instances.push_back(inst);
        return scene;
      }
      
      size_t numObjects;
      in.read((char*)&numObjects,sizeof(numObjects));
      int meshBegin = 0;
      std::vector<Object *> objects;
      for (int i=0;i<numObjects;i++) {
        int count;
        in.read((char*)&count,sizeof(count));
        Object *object = new Object;
        for (int i=0;i<count;i++)
          object->meshes.push_back(meshes[meshBegin++]);
        objects.push_back(object);
      }
      
      size_t numInstances;
      in.read((char*)&numInstances,sizeof(numInstances));
      for (int instID=0;instID<numInstances;instID++) {
        Instance *inst = new Instance;
        in.read((char*)&inst->xfm,sizeof(inst->xfm));
        int objectID;
        in.read((char*)&objectID,sizeof(objectID));
        inst->object = objects[objectID];
        scene->instances.push_back(inst);
      }
      return scene;
    }

    box3d Scene::bounds() const
    {
      std::cout << "#dpm: computing scene bounds" << std::endl;
      box3d bounds;
      for (auto inst : instances)
        for (auto m : inst->object->meshes)
          for (auto v : m->vertices)
            bounds.extend(xfmPoint(inst->xfm,v));
      std::cout << "#dpm: bounds = " << bounds << std::endl;
      return bounds;
    }

  }
}

