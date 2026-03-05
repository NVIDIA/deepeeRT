
#include "owl/owl_device.h"
#include "OWLBackend.h"

using namespace dprt::owl;

extern "C" __constant__ LaunchParams optixLaunchParams;

/*! DEFAULT ch prog */
OPTIX_CLOSEST_HIT_PROGRAM(TriMesh)()
{
  auto &geom = owl::getProgramData<dprt::owl::TriangleMesh::DD>();
  DPRTHit hit;
  hit.primID = optixGetPrimitiveIndex();
  hit.instID = optixGetInstanceIndex();
  hit.geomUserData = geom.userData;
  hit.t = optixGetRayTmax();
  hit.u = optixGetTriangleBarycentrics().x;
  hit.v = optixGetTriangleBarycentrics().y;
  
  auto prd = owl::getPRD<DPRTHit *>();
  *prd = hit;
}

OPTIX_RAYGEN_PROGRAM(raygen)()
{
  int rayID = owl::getLaunchIndex().x+1024*owl::getLaunchIndex().y;
  auto &lp = optixLaunchParams;
  if (rayID >= lp.numRays)
    return;
  
  if (rayID == 512*1024+512) {
    printf("rayid %i\n",rayID);
  }
  owl::Ray ray(owl::vec3f((owl::vec3d&)lp.rays[rayID].origin),
               owl::vec3f((owl::vec3d&)lp.rays[rayID].direction),
               lp.rays[rayID].tMin,
               lp.rays[rayID].tMax);
  DPRTHit *hit = &lp.hits[rayID];
  hit->primID = -1;
  owl::traceRay(lp.model,ray,hit);
}


