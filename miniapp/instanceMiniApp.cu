// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"
#include "DGEF.h"
#include <fstream>

namespace miniapp {

  /*! helper function that creates a semi-random color from an ID */
  inline __cubql_both vec3f randomColor(int i)
  {
    const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL;
    const uint64_t FNV_prime = 0x10001a7;
    uint32_t v = (uint32_t)FNV_offset_basis;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;

    int r = v >> 24;
    v = FNV_prime * v ^ i;
    int b = v >> 16;
    v = FNV_prime * v ^ i;
    int g = v >> 8;
    return vec3f((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
  }

  void getFrame(std::string up,
                vec3d &dx,
                vec3d &dy,
                vec3d &dz)
  {
    if (up == "z") {
      dx = {1.,0.,0.};
      dy = {0.,1.,0.};
      dz = {0.,0.,1.};
      return;
    }
    if (up == "y") {
      dx = {1.,0.,0.};
      dz = {0.,1.,0.};
      dy = {0.,0.,1.};
      return;
    }
    throw std::runtime_error("unhandled 'up'-specifier of '"+up+"'");
  }
  
  DPRWorld createWorld(DPRContext context,
                       dgef::Scene *scene)
  {
    std::map<dgef::Object *, DPRGroup> objects;
    CUBQL_CUDA_SYNC_CHECK();

    for (auto inst : scene->instances)
      objects[inst->object] = 0;

    std::cout << "#dpm: creating " << objects.size() << " objects" << std::endl;
    int meshID = 0;
    for (auto &pairs : objects) {
      auto obj = pairs.first;
      std::vector<DPRTriangles> geoms;
      for (auto pm : obj->meshes) {
        std::cout << "#dpm: creating dpr triangle mesh w/ "
                  << prettyNumber(pm->indices.size()) << " triangles"
                  << std::endl;
        DPRTriangles geom
          = dprCreateTrianglesDP(context,
                                 meshID++,
                                 (DPRvec3*)pm->vertices.data(),
                                 pm->vertices.size(),
                                 (DPRint3*)pm->indices.data(),
                                 pm->indices.size());
        CUBQL_CUDA_SYNC_CHECK();
        geoms.push_back(geom);
      }
      CUBQL_CUDA_SYNC_CHECK();
      
      DPRGroup group = dprCreateTrianglesGroup(context,
                                               geoms.data(),
                                               geoms.size());
      objects[obj] = group;
    }
    CUBQL_CUDA_SYNC_CHECK();
    
    std::cout << "#dpm: creating dpr world" << std::endl;
    std::vector<affine3d> xfms;
    std::vector<DPRGroup> groups;
    for (auto inst : scene->instances) {
      xfms.push_back(inst->xfm);
      groups.push_back(objects[inst->object]);
    }
    DPRWorld world = dprCreateWorldDP(context,
                                      groups.data(),
                                      (DPRAffine*)xfms.data(),
                                      groups.size());
    CUBQL_CUDA_SYNC_CHECK();
    return world;
  }


  __global__
  void g_shadeRays(vec4f *d_pixels,
                   DPRRay *d_rays,
                   DPRHit *d_hits,
                   vec2i fbSize)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    //Ray ray = (const Ray &)d_rays[ix+iy*fbSize.x];
    DPRHit hit = d_hits[ix+iy*fbSize.x];
    vec3f color = randomColor(hit.primID + 0x290374*hit.geomUserData);
    vec4f pixel = {color.x,color.y,color.z,1.f};
    int tid = ix+iy*fbSize.x;
    d_pixels[tid] = pixel;
  }
  
  __global__
  void g_generateRays(DPRRay *d_rays,
                      vec2i fbSize,
                      const Camera camera)
  {
    static_assert(sizeof(DPRRay) == sizeof(Ray));
    
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    double u = ix+.5;
    double v = iy+.5;

    bool dbg = false;//ix == 512 && iy == 512;
    vec2d pixel = {u,v};
    Ray ray = camera.generateRay(pixel,dbg);

    int rayID = ix+iy*fbSize.x;
    if (dbg)
      printf("ray %f %f %f : %f %f %f\n",
             (float)ray.origin.x,
             (float)ray.origin.y,
             (float)ray.origin.z,
             (float)ray.direction.x,
             (float)ray.direction.y,
             (float)ray.direction.z);
    ((Ray *)d_rays)[rayID] = ray;
  }
  
  void main(int ac, char **av)
  {
    std::string inFileName;
    std::string outFileName = "deepeeTest.ppm";
    vec2i fbSize = { 1024,1024 };
    uint64_t flags = 0;
    for (int i=1;i<ac;i++) {
      std::string arg = av[i];
      if (arg[0] != '-') {
        inFileName = arg;
      } else if (arg == "-bc" || arg == "--backface-culling") {
        flags |= DPR_CULL_BACK;
      } else if (arg == "-fc" || arg == "--frontface-culling") {
        flags |= DPR_CULL_FRONT;
      } else if (arg == "-or" || arg == "--output-res") {
        fbSize.x = std::stoi(av[++i]);
        fbSize.y = std::stoi(av[++i]);
      } else
        throw std::runtime_error("un-recognized cmdline arg '"+arg+"'");
    }
    if (inFileName.empty())
      throw std::runtime_error("no input file name specified");

    dgef::Scene *scene = dgef::Scene::load(inFileName);

    box3d bounds = scene->bounds();
    Camera camera = generateCamera(fbSize,
                                   /* bounds to focus on */
                                   bounds,
                                   /* point we're looking from*/
                                   length(bounds.size())*vec3d(2,1,4),
                                   /* up for orientation */
                                   vec3d(0,1,0));

    vec2i bs(16,16);
    vec2i nb = divRoundUp(fbSize,bs);
    
    std::cout << "#dpm: creating dpr context" << std::endl;
    DPRContext dpr = dprContextCreate(DPR_CONTEXT_GPU,0);
    std::cout << "#dpm: creating world" << std::endl;
    DPRWorld world = createWorld(dpr,scene);

    CUBQL_CUDA_SYNC_CHECK();
    DPRRay *d_rays = 0;
    cudaMalloc((void **)&d_rays,fbSize.x*fbSize.y*sizeof(DPRRay));
    CUBQL_CUDA_SYNC_CHECK();
    g_generateRays<<<nb,bs>>>(d_rays,fbSize,camera);
    CUBQL_CUDA_SYNC_CHECK();
      
    DPRHit *d_hits = 0;
    cudaMalloc((void **)&d_hits,fbSize.x*fbSize.y*sizeof(DPRHit));

    CUBQL_CUDA_SYNC_CHECK();
    std::cout << "#dpm: calling trace" << std::endl;
    dprTrace(world,d_rays,d_hits,fbSize.x*fbSize.y,flags);

    std::cout << "#dpm: shading rays" << std::endl;
    vec4f *m_pixels = 0;
    cudaMallocManaged((void **)&m_pixels,fbSize.x*fbSize.y*sizeof(vec4f));
    g_shadeRays<<<nb,bs>>>(m_pixels,d_rays,d_hits,fbSize);
    cudaStreamSynchronize(0);


    std::cout << "#dpm: writing test image to " << outFileName << std::endl;
    std::ofstream out(outFileName.c_str());

    char buf[100];
    sprintf(buf,"P3\n#deepee test image\n%i %i 255\n",fbSize.x,fbSize.y);
    out << "P3\n";
    out << "#deepeeRT test image\n";
    out << fbSize.x << " " << fbSize.y << " 255" << std::endl;
    for (int iy=0;iy<fbSize.y;iy++) {
      for (int ix=0;ix<fbSize.x;ix++) {
        vec4f pixel = m_pixels[ix+(fbSize.y-1-iy)*fbSize.x];
        auto write = [&](float f) {
          f = f*256.f;
          f = std::min(f,255.f);
          f = std::max(f,0.f);
          out << int(f) << " ";
        };
        write(pixel.x);
        write(pixel.y);
        write(pixel.z);
        out << " ";
      }
      out << "\n";
    }
  }
}

int main(int ac, char **av)
{
  miniapp::main(ac,av);
  return 0;
}
