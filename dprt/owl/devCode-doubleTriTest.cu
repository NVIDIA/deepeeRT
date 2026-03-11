
#include "owl/owl_device.h"
#include "OWLBackend.h"

using dprt::owl::TriangleMesh;

struct DoubleTriTestPRD {
  cuBQL::ray_t<double> dpRay;
  DPRTHit hit;
};

OPTIX_BOUNDS_PROGRAM(TriMesh)(const void *geomData,
                              owl::common::box3f &bounds,
                              int primID)
{
  dprt::owl::TriangleMesh::DD &mesh
    = *(dprt::owl::TriangleMesh::DD *)geomData;
  vec3i idx = mesh.indices[primID];
  (box3f&)bounds = box3f()
    .extend(mesh.vertices[idx.x])
    .extend(mesh.vertices[idx.y])
    .extend(mesh.vertices[idx.z]);
  // printf("bounds %i (%f %f %f) (%f %f %f)\n",
  //        primID,
  //        bounds.lower.x,
  //        bounds.lower.y,
  //        bounds.lower.z,
  //        bounds.upper.x,
  //        bounds.upper.y,
  //        bounds.upper.z);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriMesh)()
{
  auto &prd = owl::getPRD<DoubleTriTestPRD>();
}

OPTIX_INTERSECT_PROGRAM(TriMesh)()
{
  auto &prd = owl::getPRD<DoubleTriTestPRD>();
  const dprt::owl::TriangleMesh::DD &mesh
    = owl::getProgramData<dprt::owl::TriangleMesh::DD>();
  int primID = optixGetPrimitiveIndex();
  vec3i idx = mesh.indices[primID];

  cuBQL::triangle_t<double> tri;
  tri.a = vec3d(mesh.vertices[idx.x]);
  tri.b = vec3d(mesh.vertices[idx.y]);
  tri.c = vec3d(mesh.vertices[idx.z]);
  
  cuBQL::RayTriangleIntersection_t<double> isec;
  if (!isec.compute(prd.dpRay,tri)) return;

  auto &hit = prd.hit;
  hit.primID = optixGetPrimitiveIndex();
  hit.instID = optixGetInstanceIndex();
  hit.geomUserData = mesh.userData;
  hit.u = isec.u;
  hit.v = isec.v;
  hit.t = isec.t;
  prd.dpRay.tMax = hit.t;

  optixReportIntersection(__double2float_ru(hit.t),hit.primID);
}

OPTIX_RAYGEN_PROGRAM(raygen)()
{
  int rayID = owl::getLaunchIndex().x+1024*owl::getLaunchIndex().y;
  auto &lp = optixLaunchParams;
  if (rayID >= lp.numRays)
    return;

  DoubleTriTestPRD prd;
  (vec3d&)prd.dpRay.origin = (vec3d&)lp.rays[rayID].origin;
  (vec3d&)prd.dpRay.direction = (vec3d&)lp.rays[rayID].direction;
  prd.dpRay.tMin = lp.rays[rayID].tMin;
  prd.dpRay.tMax = lp.rays[rayID].tMax;
  
  owl::Ray ray(owl::vec3f((owl::vec3d&)prd.dpRay.origin),
               owl::vec3f((owl::vec3d&)prd.dpRay.direction),
               prd.dpRay.tMin,
               prd.dpRay.tMax);

  // printf("trace %f %f\n",prd.dpRay.tMin,prd.dpRay.tMax);
  prd.hit.primID = -1;
  owl::traceRay(lp.model,ray,prd);
  lp.hits[rayID] = prd.hit;
}



