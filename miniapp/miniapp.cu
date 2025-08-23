// Copyright 2025-2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"
#include "Camera.h"

namespace miniapp {

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
      dy = {1.,0.,0.};
      dz = {0.,1.,0.};
      dx = {0.,0.,1.};
      return;
    }
    throw std::runtime_error("unhandled 'up'-specifier of '"+up+"'");
  }
  
  DPRWorld createWorld(DPRContext context,
                       const std::vector<Mesh *> &meshes)
  {
    std::vector<DPRTriangles> geoms;
    for (auto pm : meshes) {
      vec3d *d_vertices = 0;
      vec3i *d_indices = 0;
      pm->upload(d_vertices,d_indices);
      DPRTriangles geom = dprCreateTrianglesDP(context,
                                               geoms.size(),
                                               (DPRvec3*)d_vertices,
                                               pm->vertices.size(),
                                               (DPRint3*)d_indices,
                                               pm->indices.size());
      geoms.push_back(geom);
    }
    DPRGroup group = dprCreateTrianglesGroup(context,
                                             geoms.data(),
                                             geoms.size());
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

    Ray ray = (const Ray &)d_rays[ix+iy*fbSize.x];
    vec3f color = vec3f(abs(ray.direction));
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
    
    Ray ray = camera.generateRay({u,v});
    (Ray &)d_rays[ix+iy*fbSize.x] = ray;
  }
  
  void main(int ac, char **av)
  {
    double scale = 1e3f;
    std::string up = "y";
    std::string inFileName;
    int terrainRes = 10*1024;
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
    vec3d dx,dy,dz;
    getFrame(up,dx,dy,dz);
    
    object.translate(scale*(dx+dy)-object.center());
    
    Mesh terrain = generateTesselatedQuad(terrainRes,dx,dy,dz,2.f*scale);
    Camera camera = generateCamera(fbSize,
                                   /* bounds to focus on */
                                   object.bounds(),
                                   /* point we're looking from*/
                                   -2.*scale*(dx+dy)+.1*scale*dz,
                                   /* up for orienation */
                                   dz);

    vec2i bs(32,32);
    vec2i nb = divRoundUp(fbSize,bs);
    
    DPRContext dpr = dprContextCreate(DPR_CONTEXT_GPU,0);
    DPRWorld world = createWorld(dpr,{&object,&terrain});

    DPRRay *d_rays = 0;
    cudaMalloc((void **)&d_rays,fbSize.x*fbSize.y*sizeof(DPRRay));
    g_generateRays<<<bs,nb>>>(d_rays,fbSize,camera);
    
    DPRHit *d_hits = 0;
    cudaMalloc((void **)&d_hits,fbSize.x*fbSize.y*sizeof(DPRHit));

    dprTrace(world,d_rays,d_hits,fbSize.x*fbSize.y);
    
    vec4f *m_pixels = 0;
    cudaMallocManaged((void **)&m_pixels,fbSize.x*fbSize.y*sizeof(vec4f));
    g_shadeRays<<<bs,nb>>>(m_pixels,d_rays,d_hits,fbSize);

  }
}

int main(int ac, char **av)
{
  miniapp::main(ac,av);
  return 0;
}
