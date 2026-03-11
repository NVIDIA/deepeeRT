
#include "owl/owl_device.h"
#include "OWLBackend.h"
#include "cuBQL/math/Ray.h"

using namespace dprt::owl;

inline __device__ vec3f to_cubql(float3 v) { return (const vec3f&)v; }

struct PRD {
  DPRTHit *hit;
  double t;
};

OPTIX_CLOSEST_HIT_PROGRAM(TriMesh)()
{
  auto &geom = owl::getProgramData<dprt::owl::TriangleMesh::DD>();
  DPRTHit hit;
  hit.primID = optixGetPrimitiveIndex();
  hit.instID = optixGetInstanceIndex();
  hit.geomUserData = geom.userData;
  hit.u = optixGetTriangleBarycentrics().x;
  hit.v = optixGetTriangleBarycentrics().y;

  PRD &prd = owl::getPRD<PRD>();
  hit.t = prd.t;
  
  *prd.hit = hit;
}

OPTIX_ANY_HIT_PROGRAM(TriMesh)()
{
  vec3f f_org = to_cubql(optixGetObjectRayOrigin());
  vec3f f_dir = to_cubql(optixGetObjectRayDirection());
  vec3d org = vec3d(f_org);
  vec3d dir = vec3d(f_dir);

  int primID = optixGetPrimitiveIndex();
  auto &geom = owl::getProgramData<dprt::owl::TriangleMesh::DD>();
  vec3i idx = geom.indices[primID];
  vec3d a = vec3d(geom.vertices[idx.x]);
  vec3d b = vec3d(geom.vertices[idx.y]);
  vec3d c = vec3d(geom.vertices[idx.z]);
  vec3d n = normalize(cross(b-a,c-a));
  
  double t = dot(a-org,n) / dot(dir,n);
  
  auto &prd = owl::getPRD<PRD>();
  prd.t = t;
}




OPTIX_RAYGEN_PROGRAM(raygen)()
{
  int rayID = owl::getLaunchIndex().x+1024*owl::getLaunchIndex().y;
  auto &lp = optixLaunchParams;
  if (rayID >= lp.numRays)
    return;
  
  owl::Ray ray(owl::vec3f((owl::vec3d&)lp.rays[rayID].origin),
               owl::vec3f((owl::vec3d&)lp.rays[rayID].direction),
               lp.rays[rayID].tMin,
               lp.rays[rayID].tMax);

  PRD prd;
  prd.hit = &lp.hits[rayID];
  prd.hit->primID = -1;
  owl::traceRay(lp.model,ray,prd);
}


