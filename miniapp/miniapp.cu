// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#include "Camera.h"
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
                       const std::vector<Mesh *> &meshes)
  {
    std::vector<DPRTriangles> geoms;
    int meshID = 0;
    for (auto pm : meshes) {
      vec3d *d_vertices = 0;
      vec3i *d_indices = 0;
      pm->upload(d_vertices,d_indices);
      std::cout << "#dpm: creating dpr triangle mesh w/ "
                << prettyNumber(pm->indices.size()) << " triangles"
                << std::endl;
      DPRTriangles geom = dprCreateTrianglesDP(context,
                                               meshID++,
                                               (DPRvec3*)d_vertices,
                                               pm->vertices.size(),
                                               (DPRint3*)d_indices,
                                               pm->indices.size());
      geoms.push_back(geom);
    }
    std::cout << "#dpm: creating dpr triangles group w/ "
              << geoms.size() << " meshes" << std::endl;
    DPRGroup group = dprCreateTrianglesGroup(context,
                                             geoms.data(),
                                             geoms.size());
    std::cout << "#dpm: creating dpr world" << std::endl;
    DPRWorld world = dprCreateWorldDP(context,
                                      &group,
                                      nullptr,
                                      1);
    return world;
  }


  __global__
  void g_shadeRays(vec4f *d_pixels,
                   DPRRay *d_rays,
                   DPRHit *d_hits,
                   const vec2i &fbSize)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    //Ray ray = (const Ray &)d_rays[ix+iy*fbSize.x];
    DPRHit hit = d_hits[ix+iy*fbSize.x];
    vec3f color = randomColor(hit.primID + 0x290374*hit.geomUserData);
    vec4f pixel = {color.x,color.y,color.z,1.f};
    d_pixels[ix+iy*fbSize.x] = pixel;
  }
  
  __global__
  void g_generateRays(DPRRay *d_rays,
                      const vec2i &fbSize,
                      const Camera camera)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    double u = ix+.5;
    double v = iy+.5;

    vec2d pixel = {u,v};
    Ray ray = camera.generateRay(pixel,false);
    (Ray &)d_rays[ix+iy*fbSize.x] = ray;
  }
  
  void main(int ac, char **av)
  {
    double scale = 1;
    std::string up = "y";
    std::string inFileName;
    std::string outFileName = "deepeeTest.ppm";
    int terrainRes = 2*1024;
    vec2i fbSize = { 1024,1024 };
    for (int i=1;i<ac;i++) {
      std::string arg = av[i];
      if (arg[0] != '-') {
        inFileName = arg;
      } else if (arg == "-up") {
        up = av[++i];
      } else if (arg == "-or" || arg == "--output-res") {
        fbSize.x = std::stoi(av[++i]);
        fbSize.y = std::stoi(av[++i]);
      } else if (arg == "-tr" || arg == "--terrain-res") {
        terrainRes = std::stoi(av[++i]);
      } else if (arg == "-s" || arg == "--scale") {
        scale = std::stof(av[++i]);
      } else
        throw std::runtime_error("un-recognized cmdline arg '"+arg+"'");
    }
    if (inFileName.empty())
      throw std::runtime_error("no input file name specified");

    Mesh object;
    object.load_binmesh(inFileName);
    scale = scale * length(object.bounds().size());
    vec3d dx,dy,dz;
    getFrame(up,dx,dy,dz);
    
    object.translate(scale*(dx+dy)-object.center());

    std::cout << "#dpm: creating tessellated quad for base terrain" << std::endl;
    Mesh terrain = generateTessellatedQuad(terrainRes,dx,dy,dz,2.f*scale);
    terrain.translate(-.5f*object.bounds().size().z*dz);
    Camera camera = generateCamera(fbSize,
                                   /* bounds to focus on */
                                   object.bounds(),
                                   /* point we're looking from*/
                                   -1.*scale*(dx+dy)+.5*scale*dz,
                                   /* up for orientation */
                                   dz);

    vec2i bs(32,32);
    vec2i nb = divRoundUp(fbSize,bs);
    
    std::cout << "#dpm: creating dpr context" << std::endl;
    DPRContext dpr = dprContextCreate(DPR_CONTEXT_GPU,0);
    std::cout << "#dpm: creating world" << std::endl;
    // DPRWorld world = createWorld(dpr,{&object});
    DPRWorld world = createWorld(dpr,{&object,&terrain});

    DPRRay *d_rays = 0;
    cudaMalloc((void **)&d_rays,fbSize.x*fbSize.y*sizeof(DPRRay));
    g_generateRays<<<bs,nb>>>(d_rays,fbSize,camera);
    
    DPRHit *d_hits = 0;
    cudaMalloc((void **)&d_hits,fbSize.x*fbSize.y*sizeof(DPRHit));

    std::cout << "#dpm: calling trace" << std::endl;
    dprTrace(world,d_rays,d_hits,fbSize.x*fbSize.y);
    
    std::cout << "#dpm: shading rays" << std::endl;
    vec4f *m_pixels = 0;
    cudaMallocManaged((void **)&m_pixels,fbSize.x*fbSize.y*sizeof(vec4f));
    g_shadeRays<<<bs,nb>>>(m_pixels,d_rays,d_hits,fbSize);
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
