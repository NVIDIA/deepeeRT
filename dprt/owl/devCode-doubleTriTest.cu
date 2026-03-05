
#include "owl/owl_device.h"
#include "OWLBackend.h"

using namespace drpt::owl;

struct DoubleTriTestPRD {
  cuBQL::ray_t<double> dpRay;
  DPRTHit hit;
};

OPTIX_BOUNDS_PROGRAM(TriMeshUG)(const void *geomData,
                                box3f &bounds,
                                int primID)
{
  TriangleMesh &mesh = *(TriangleMesh *)geomData;
  vec3i idx = mesh.indices[hit.primID];
  bounds = box3f()
    .extend(mesh.vertices[idx.x])
    .extend(mesh.vertices[idx.y])
    .extend(mesh.vertices[idx.z]);
}

OPTIX_INTERSECT_PROGRAM(TriMeshUG)()
{
  auto &prd = owl::getPRD<>(DoubleTriTestPRD);
  TriangleMesh &mesh = owl::getProgramData<TriangleMesh>();
  vec3i idx = mesh.indices[hit.primID];

  cuBQL::triangle_t<double> tri;
  tri.a = vec3d(mesh.vertices[idx.x]);
  tri.b = vec3d(mesh.vertices[idx.y]);
  tri.c = vec3d(mesh.vertices[idx.z]);
  
  cuBQL::RayTriangleIntersection_t<double> isec;
  if (!isec.compute(prd.dpRay,tri)) return;

  auto &hit = prd.hit;
  hit.primID = optixGetPrimitiveIndex();
  hit.instID = optixGetInstanceIndex();
  hit.geomUserData = geom.userData;
  hit.u = isec.u;
  hit.v = isec.v;
  hit.t = isec.t;
  dpRay.tMax = hit.t;
  optixReportIntersection(nextafter(float(hit.t),CUDART_INF),0);
}

OPTIX_RAYGEN_PROGRAM(raygen)()
{
  int rayID = owl::getLaunchIndex().x+1024*owl::getLaunchIndex().y;
}


