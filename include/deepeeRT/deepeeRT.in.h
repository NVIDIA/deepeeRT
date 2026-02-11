// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/* The "deepee" API: Overview

   On a high level, this API allows for specifying single-level
   instanced scenes, with instances built over groups of geometries
   (currently all geometries are triangle meshes), with a "wavefront"
   API for tracing rays against such scenes. "Wavefront" in this
   context means that once a scene has been built be the user he can
   trace an array (wave) of rays against such a scene, and get an array
   of hit descriptors back. There are currently no anyhit of
   closest-hit programs in this API, nor intersection programs,
   bounding box programs.
  
   Naming convention: Throughout this API the first letter 'p' stands
   for 'prime' (because this follows a optix prime like pattern), the
   letters 'DP' and (eventually) 'SP' then indicate if the respective
   type is intended to be used in a double (fp64)-precision or a single
   (fp32)-precision ray tracing context. A "DP Context" in this context
   refers to a context in which rays are supposed to be traced and
   intersected with doble-precision accuracy, yielding double-precision
   hits, not necessarily that all of the geometry is specified in
   double precision, too. Thus, some functions like dprCreateTriangles
   may further split into dprCreateTrianglesSP (single-precision
   vertices for the triangles, but a double-precision trace context)
   and dprTrianglesDP (double precision triangles, double-precision
   tracing.

   Anything starting with "d_" is expected to be a device-side array
   (ie, accessible by device side code), and unless otherwise noted is
   expected to remain alive/persistent for the duration of any call or
   set of calls the user wants to perform with this data (eg, vertex
   arrays are expected to last ont only the lifetime of creating the
   geometry, but also all the calls for tracing rays that might hit
   such a geometry 

   SP Rays and contexts are currently not supported. 
*/

#pragma once

#include <sys/types.h>
#include <stdint.h>

#define DEEPEERT_VERSION_MAJOR @DEEPEERT_VERSION_MAJOR@
#define DEEPEERT_VERSION_MINOR @DEEPEERT_VERSION_MINOR@
#define DEEPEERT_VERSION_PATCH @DEEPEERT_VERSION_PATCH@


#ifdef _WIN32
// for now, we only support a static build for this library
# if 1 || defined(deepeeRT_STATIC)
#  define DPR_INTERFACE /* nothing */
# elif defined(deepeeRT_EXPORTS)
#  define DPR_INTERFACE __declspec(dllexport)
# else
#  define DPR_INTERFACE __declspec(dllimport)
# endif
#elif defined(__clang__) || defined(__GNUC__)
#  define DPR_INTERFACE __attribute__((visibility("default")))
#else
#  define DPR_INTERFACE
#endif

#ifdef __cplusplus
#  define DPR_API extern "C" DPR_INTERFACE
#  define DPR_IF_CPP(a) a
#else
#  define DPR_API /* bla */
#  define DPR_IF_CPP(a) /* ignore */
#endif

/*! a single "triangle mesh" geometry object */
typedef struct _DPRTriangles     *DPRTriangles;
/*! an group of one or more *geometries* */
typedef struct _DPRGroup         *DPRGroup;
/*! an entire *world*, built over (one or more) *instances* of geometry group */
typedef struct _DPRWorld         *DPRWorld;
typedef struct _DPRContext       *DPRContext;

typedef enum { DPR_CONTEXT_GPU } DPRContextType;

#define DPR_FLAGS_NONE (uint64_t(0))
/*! if enabled, we will skip all intersections with triangles whose
    normal faces TOWARDS the origin (ie, cull iff dot(ray.dir,N)<0) */
#define DPR_CULL_FRONT (uint64_t(1ull<<0))
/*! if enabled, we will skip all intersections with triangles whose
    normal faces AWAY the origin (ie, cull iff dot(ray.dir,N)>0) */
#define DPR_CULL_BACK  (uint64_t(1ull<<1))

struct DPRint3 { int32_t x,y,z; };
struct DPRvec3 { double  x,y,z; };
struct DPRvec4 { double  x,y,z,w; };

/*! affine transform matrix, for instances (with embree-style naming,
  'p' for the translation/offset part, 'l' for the linear
  transform/rotational part, and vx,vy,vz for the vectors of that
  3x3 linear transform */
struct DPRAffine {
  struct {
    DPRvec3 vx;
    DPRvec3 vy;
    DPRvec3 vz;
  } l;
  DPRvec3 p;
};
  
struct DPRRay {
  DPRvec3 origin;
  DPRvec3 direction;
  double  tMin;
  double  tMax;
};

struct DPRHit {
  /*! index of prim within the geometry it was created in. A value of
    '-1' means 'no hit' */
  int     primID;
  /* index of the instance that contained the hit point. Undefined if
     on hit occurred */
  int     instID;
  /*! user-supplied geom ID (the one specified during geometry create
    call) for the geometry that contained the hit. Unlike primID and
    instID this is *not* a linear ID, but whatever int64 value the
    user specified there. */
  uint64_t geomUserData;
  union {
    double  tHit; // << old (pre v0.2) spelling
    double t;
  };
  double  u, v;
};

DPR_API
DPRContext dprContextCreate(DPRContextType contextType,
                            int gpuToUse);

/*! a triangle mesh whose vertices are in double precision, to be used
  within a double-precision tracing context. This function will
  currently make a *copy* of those arrays to make sure the user
  doesn't accidentally use host-side and/or temporary data. A future
  version should arguably allow to specify whether this copy should or
  shouldn't be done. */
DPR_API
DPRTriangles dprCreateTrianglesDP(DPRContext context,
                                  /*! a 64-bit user-provided data that
                                    gets attached to this mesh; this is
                                    what gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t userData,
                                  /*! device array of vertices */
                                  DPRvec3 *vertexArray,
                                  size_t   vertexCount,
                                  /*! device array of int3 vertex indices */
                                  DPRint3 *indexArray,
                                  size_t   indexCount);

/*! create an object representing a group of one or more triangle
    meshes that can then get instantiated (dpr never directly
    instantiates individual triangle meshes, but always groups of
    meshes. If you need to instantiate a single mesh you need to first
    create a TrianglesGroup with that single mesh, then instantiate
    this). */
DPR_API
DPRGroup dprCreateTrianglesGroup(DPRContext,
                                 DPRTriangles *triangleGeomsArray,
                                 size_t        triangleGeomsCount);

/*! creates a world over one or more triangle mesh groups; each
    instance is defined by a handle to the group it wants to
    instantaite, plus an associated transform that represents the
    object-to-world transform supposed to be applied to this
    geometry. */
DPR_API
DPRWorld dprCreateWorldDP(DPRContext,
                          DPRGroup  *instanceGroups,
                          DPRAffine *instanceTransforms,
                          size_t     instanceCount);

/*! traces a set of rays against a previously computed world. */
DPR_API
void dprTrace(/*! the world we want the rays to be traced against */
              DPRWorld world,
              /*! *device* array of rays that need tracing */
              DPRRay *d_rays,
              /*! *device* array of where to store the hits */
              DPRHit *d_hits,
              /*! number of rays that need tracing. d_rays and
                d_hits *must* have (at least) that many entires */
              int numRays,
              uint64_t flags = 0ull);

/*! frees a previously created world. This should also free all the
    memory that this world object has created for internal
    acceleration structures, but will NOT free the groups that it was
    created over. It is user's job to free those appropriately */
DPR_API void dprFreeWorld(DPRWorld world);

/*! frees a previously created triangle mesh. This should also free
    all the memory that this group has created for internal storage of
    the triangles it was create over. Once freed world objects created
    over this triangle mesh (or over objects created over this
    triangle mesh) are no longer valid and may no longer get traced
    against. */
DPR_API void dprFreeTriangles(DPRTriangles triangles);

/*! frees a previously created triangle mesh group. This should also
    free all the memory that this group has created for internal
    acceleration structures, but will NOT free the triangle meshes
    that it was created over. It is user's job to free those
    appropriately. Once freed world objects created over this group
    are no longer valid and may no longer get traced against. */
DPR_API void dprFreeGroup(DPRGroup group);

/*! frees the root context. This is currently NOT guaranteed to free
  all the objects created within this context. */
DPR_API void dprFreeContext(DPRContext context);


