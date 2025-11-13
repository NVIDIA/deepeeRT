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
