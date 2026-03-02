# DeepeeRT - A "Basics-only" Ray Tracing Library for Double-Precision Ray Tracing

Build Status:
[![Windows](https://github.com/NVIDIA/deepeeRT/actions/workflows/Windows.yml/badge.svg)](https://github.com/NVIDIA/deepeeRT/actions/workflows/Windows.yml) 
[![Ubuntu](https://github.com/NVIDIA/deepeeRT/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/NVIDIA/deepeeRT/actions/workflows/Ubuntu.yml)

## Introduction

The goal of this library/project is to allow scientists and other
practitioners to perform ray tracing computations in double precision,
i.e., where both rays, hits, and geometry can be specified in FP64
precision, and traversal and intersection happen in FP64 precision as
well.

Wherever possible we encourage users to use single precision based
(and more complete!) tools such as OptiX; but for those that for some
reason cannot rely on use single precision calculations this library
offers a simple, "OptiX Prime" like API for specifying (double
precision) triangle mesh geometry and single instance layer worlds,
and to trace arrays of rays to get arrays of hits. 

Unlike more complete libraries such as OptiX this library will
explicitly *not* contain a full implementation of the RTX Pipeline
(ie, no AnyHit, ClosestHit, Intersection, etc programs, no ray types,
ray payloads, shader binding tables, etc), nor features such as ray
reordering, denoising, etc. Instead, this library intentionally
focusses on only tracing rays, reading rays from a device-array of
rays, and filling in a corresponding device array of hits.

## Changes in 1.x

When initially conceived the naming scheme used in this library
generally built on the abbreviation of "DPRT" for "Double Precision
Ray Tracing", and on the way that DP is generally pronouced as "dee
pee". However, it used this general idea in many different and
inconstent ways---API types and functions started with
`dpr<Function>()` and `DPR<Type>` (some actually used `DP<Type>`...),
constants with `DPR_`; namespace was `dprt::`; main subdir name was
`dp/`; repo name, library name, and install dir name was `deepeeRT`;
some C and CMake constants used `DEEPEERT` and some `DEEPEE`; etc.

For 1.0, this has undergone a significant cleanup: The repo name
cannot easily be changed, but pretty much everything else has changed
to 'dprt': install directory is now `dprt/`, as is the main source
dir; install target name is now also `dprt`, so apps should now use
`find_package(dprt)` and link to `dprt::dprt`. Installed header file
is `dprt/dprt.h`.. Finally, API names are now:
for types `DPRT<Type>`, for functions `dprt<Function>()`,
and for all constants and enums `DPRT_<constant>`.

For 1.0 we also renamed the `world` to be a `model` (to become more
consistent with other graphics APIs); we have also deprecated the
`_DP` vs `_SP` suffix naming scheme that was used for showing which
functions used double vs single precision arguments (`dprt` only uses
doubles, anyway).

## Copyright and License

// Copyright 2025 NVIDIA Corp
// SPDX-License-Identifier: Apache-2.0

## Building and Usage

Preferred way of using this library is as a git submodule through
cmaked, included in your CMake script via

    add_subdirectory(<path to>/deepeeRT EXCLUDE_FROM_ALL)
	
then linked to your library as

    target_link_libraries(yourLib PUBLIC deepeeRT)

This project uses "modern cmake", so this should pull in all
dependencies, includes, defines, etc.


## Release History

- no releases yet.
