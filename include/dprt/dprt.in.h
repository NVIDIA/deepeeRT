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

   DPRT currently works exclusively on double-precision data; both for
   rays/hits as well as for input data (instance transforms and vertex
   arrays). Future versions may also support float input geometry, but
   this one doesn't---if this is a major blocker for your application
   please contact us.
   
   Anything starting with "d_" is expected to be a device-side array
   (ie, accessible by device side code), and unless otherwise noted is
   expected to remain alive/persistent for the duration of any call or
   set of calls the user wants to perform with this data. The current
   version of this API does not allow for 'sharing' device data for
   triangle mesh vertices, indices, etc, and will always copy that
   data. A future version will likely allow for expressing that data
   is already on device (and can be shared); but this version does
   not. If this is a major blocker (and want to serve as an early
   tester while this feature is being developed), please let us know.

*/

#pragma once

#include <sys/types.h>
#include <stdint.h>

#define DPRT_VERSION_MAJOR @DPRT_VERSION_MAJOR@
#define DPRT_VERSION_MINOR @DPRT_VERSION_MINOR@
#define DPRT_VERSION_PATCH @DPRT_VERSION_PATCH@

#ifdef _WIN32
# ifndef dprt_STATIC
// for now, we only support a static build for this library
#  define dprt_STATIC
# endif
# if defined(dprt_STATIC)
#  define DPRT_INTERFACE /* nothing */
# elif defined(dprt_EXPORTS)
#  define DPRT_INTERFACE __declspec(dllexport)
# else
#  define DPRT_INTERFACE __declspec(dllimport)
# endif
#elif defined(__clang__) || defined(__GNUC__)
#  define DPRT_INTERFACE __attribute__((visibility("default")))
#else
#  define DPRT_INTERFACE
#endif

#ifdef __cplusplus
#  define DPRT_API extern "C" DPRT_INTERFACE
#  define DPRT_IF_CPP(a) a
#else
#  define DPRT_API /* nothing for now */
#  define DPRT_IF_CPP(a) /* ignore */
#endif

/*! a single "triangle mesh" geometry object */
typedef struct _DPRTTriangles     *DPRTTriangles;
/*! an group of one or more *geometries* */
typedef struct _DPRTGroup         *DPRTGroup;
/*! an entire *model*, built over (one or more) *instances* of geometry group */
typedef struct _DPRTModel         *DPRTModel;
typedef struct _DPRTContext       *DPRTContext;

typedef enum { DPRT_CONTEXT_GPU } DPRTContextType;

/*! default value for any 'flags'-type function argument */
#define DPRT_FLAGS_NONE (uint64_t(0))
#define DPRT_FLAGS_DEFAULT DPRT_FLAGS_NONE

/*! if enabled, we will skip all intersections with triangles whose
  normal faces TOWARDS the origin (ie, cull iff dot(ray.dir,N)<0).
  Only applies to dprtTrace() */
#define DPRT_CULL_FRONT (uint64_t(1ull<<0))

/*! if enabled, we will skip all intersections with triangles whose
  normal faces AWAY the origin (ie, cull iff dot(ray.dir,N)>0).
  Only applies to dprtTrace()  */
#define DPRT_CULL_BACK  (uint64_t(1ull<<1))

/*! only applies to flags that refer to a pointer to a data; specifies
    that the app guarantees this pointer to point to 'persistent' data
    that the app will neither change nor free (ie, dprt _can_ then
    freely share this data with the app and does not have to create a
    copy). Note this only specifies that dprt _can_ share this pointer
    if it wants to, dprt may still make a copy if it so wants */
#define DPRT_PERSISTENT_DATA 1ull

/*! a three-tuple of 32-bit integer values, used to represent
  vertex indices of a triangle mesh */
struct DPRTint3 { int32_t x,y,z; };
/*! a three-tuple of double-precision floats; used to represent
  any sort of double-precision vector or point */
struct DPRTvec3 { double  x,y,z; };
// struct DPRTvec4 { double  x,y,z,w; };

/*! affine transform matrix, for instances (with embree-style naming,
  'p' for the translation/offset part, 'l' for the linear
  transform/rotational part, and vx,vy,vz for the vectors of that
  3x3 linear transform */
struct DPRTAffine {
  struct {
    DPRTvec3 vx;
    DPRTvec3 vy;
    DPRTvec3 vz;
  } l;
  DPRTvec3 p;
};
  
struct DPRTRay {
  DPRTvec3 origin;
  DPRTvec3 direction;
  double   tMin;
  double   tMax;
};

struct DPRTHit {
  /*! index of prim within the geometry it was created in. A value of
    '-1' means 'no hit' */
  int32_t primID;
  /* index of the instance that contained the hit point. Undefined if
     on hit occurred */
  int32_t instID;
  /*! user-supplied geom ID (the one specified during geometry create
    call) for the geometry that contained the hit. Unlike primID and
    instID this is *not* a linear ID, but whatever int64 value the
    user specified when creating the geometry. */
  uint64_t geomUserData;
  /*! distance to hit. */
  double   t;
  double   u, v;
};

DPRT_API
DPRTContext dprtContextCreate(DPRTContextType contextType,
                              int gpuToUse);

/*! a triangle mesh whose vertices are in double precision, to be used
  within a double-precision tracing context. This function will
  currently make a *copy* of those arrays to make sure the user
  doesn't accidentally use host-side and/or temporary data.

  The `flags` argument allows the app to specify that the input vertex and
  index arrays are 'persistent', and that dprt is thus free to share these
  pointers without creating a copy. If that flag is specified dprt will
  assume it applies to both vertex AND index arrays.
*/
DPRT_API
DPRTTriangles dprtCreateTriangles(DPRTContext context,
                                  /*! a 64-bit user-provided data that
                                    gets attached to this mesh; this is
                                    what gets reported in
                                    Hit::geomUserData if this mesh
                                    yielded the intersection.  */
                                  uint64_t  userData,
                                  /*! device array of vertices */
                                  DPRTvec3 *vertexArray,
                                  size_t    vertexCount,
                                  /*! device array of int3 vertex indices */
                                  DPRTint3 *indexArray,
                                  size_t    indexCount,
                                  uint64_t  flags
                                  DPRT_IF_CPP(= DPRT_FLAGS_DEFAULT));

/*! create an object representing a group of one or more triangle
  meshes that can then get instantiated (dpr never directly
  instantiates individual triangle meshes, but always groups of
  meshes. If you need to instantiate a single mesh you need to first
  create a TrianglesGroup with that single mesh, then instantiate
  this). */
DPRT_API
DPRTGroup dprtCreateTrianglesGroup(DPRTContext,
                                   DPRTTriangles *triangleGeomsArray,
                                   size_t        triangleGeomsCount);

/*! creates a model over one or more triangle mesh groups; each
  instance is defined by a handle to the group it wants to
  instantaite, plus an associated transform that represents the
  object-to-model transform supposed to be applied to this
  geometry. A 'null' instancetransform array will internally be
  treated the same as a array will all unit transforms.

  The `flags` argument allows the app to specify that the input vertex and
  index arrays are 'persistent', and that dprt is thus free to share these
  pointers without creating a copy. If that flag is specified dprt will
  assume it applies to both vertex AND index arrays.
*/
DPRT_API
DPRTModel dprtCreateModel(DPRTContext,
                          /*! (host side) pointer to a array of instance
                            group handles. These must be valid handles
                            created through `dprtCreateTrianglesGroup()` 
                          */
                          DPRTGroup  *instanceGroups,
                          /*! host _or_ device side pointer to an
                            array of instance transforms, in a
                            double3x4 `DPRTAffine` data layout (one
                            each per instanceGroup). A null value is
                            allowed, and indicates that all instance
                            transforms are unit transforms. If
                            non-null this must point to
                            `instanceCount` valid affine
                            transformations.
                          */
                          DPRTAffine *instanceTransforms,
                          size_t      instanceCount,
                          /*! allows to specify that the `instanceTransforms`
                            array is a persistent data array that does
                            not have to be copied */
                          uint64_t  flags
                          DPRT_IF_CPP(= DPRT_FLAGS_DEFAULT));

/*! traces a set of rays against a previously computed model. */
DPRT_API
void dprtTrace(/*! the model we want the rays to be traced against */
               DPRTModel model,
               /*! *device* array of rays that need tracing */
               DPRTRay *d_rays,
               /*! *device* array of where to store the hits */
               DPRTHit *d_hits,
               /*! number of rays that need tracing. d_rays and
                 d_hits *must* have (at least) that many entires */
               int numRays,
               uint64_t flags = 0ull);

/*! frees a previously created model. This should also free all the
  memory that this model object has created for internal
  acceleration structures, but will NOT free the groups that it was
  created over. It is user's job to free those appropriately */
DPRT_API void dprtFreeModel(DPRTModel model);

/*! frees a previously created triangle mesh. This should also free
  all the memory that this group has created for internal storage of
  the triangles it was create over. Once freed model objects created
  over this triangle mesh (or over objects created over this
  triangle mesh) are no longer valid and may no longer get traced
  against. */
DPRT_API void dprtFreeTriangles(DPRTTriangles triangles);

/*! frees a previously created triangle mesh group. This should also
  free all the memory that this group has created for internal
  acceleration structures, but will NOT free the triangle meshes
  that it was created over. It is user's job to free those
  appropriately. Once freed model objects created over this group
  are no longer valid and may no longer get traced against. */
DPRT_API void dprtFreeGroup(DPRTGroup group);

/*! frees the root context. This is currently NOT guaranteed to free
  all the objects created within this context. */
DPRT_API void dprtFreeContext(DPRTContext context);


