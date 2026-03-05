
#include "owl/owl_device.h"
#include "OWLBackend.h"

struct DoubleDistancePRD {
  PDRTRay  dpRay;
  DPRTHit *hit;
};

OPTIX_CLOSEST_HIT_PROGRAM(CH)()
{
  auto &geom = owl::getProgramData<dprt::owl::TriangleMesh::DD>();
  DPRTHit hit;
  hit.primID = optixGetPrimitiveIndex();
  hit.instID = optixGetInstanceIndex();
  hit.geomUserData = geom.userData;
  hit.u = optixGetTriangleBarycentrics().x;
  hit.v = optixGetTriangleBarycentrics().y;

  vec3f f_org = optixGetObjectRayOrigin();
  vec3f f_dir = optixGetObjectRayDirection();
  vec3d org = vec3d(f_org);
  vec3d dir = vec3d(f_dir);
  
  vec3i idx = geom.indices[hit.primID];
  vec3d a = vec3d(geom.vertices[idx.x]);
  vec3d b = vec3d(geom.vertices[idx.y]);
  vec3d c = vec3d(geom.vertices[idx.z]);
  vec3d n = normalize(cross(b-a,c-a));
  
  double t = dot(a-org,N) / dot(dir,N);
  hit.t = t;
  
  auto prd = owl::getPRD<DPRTHit *>();
  *prd = hit;
}



OPTIX_RAYGEN_PROGRAM(raygen)()
{
  int rayID = owl::getLaunchIndex().x+1024*owl::getLaunchIndex().y;
}


