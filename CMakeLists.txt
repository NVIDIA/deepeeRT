# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

set(DPRT_VERSION_MAJOR 1)
set(DPRT_VERSION_MINOR 0)
set(DPRT_VERSION_PATCH 0)
set(DPRT_VERSION ${DPRT_VERSION_MAJOR}.${DPRT_VERSION_MINOR}.${DPRT_VERSION_PATCH})
  
set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

option(DPRT_OMP "Enable experimental OpenMP build" OFF)
mark_as_advanced(DPRT_OMP)

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0048 NEW)
if (DPRT_OMP)
  project(DPRT VERSION ${DPRT_VERSION} LANGUAGES C CXX)
  set(CUBQL_DISABLE_CUDA ON)
else()
  set(CMAKE_CUDA_ARCHITECTURES native)
  project(DPRT VERSION ${DPRT_VERSION} LANGUAGES C CXX CUDA)
endif()

if(NOT SET_UP_CONFIGURATIONS_DONE)
  set(SET_UP_CONFIGURATIONS_DONE 1)

  # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
  # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
  if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
  else()
    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
  endif()
endif()

if ((${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  # this is a local build, not a subproject
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

option(DPRT_USE_LOCAL_CUBQL "Use *local* cuBQL" OFF)
mark_as_advanced(DPRT_USE_LOCAL_CUBQL)
if (DPRT_USE_LOCAL_CUBQL)
  add_subdirectory(../cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
else()
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
endif()

add_subdirectory(dprt)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  # don't build miniapp if this is a subproject
else()
  # disabled: requires to be able to do find_package(dprt), which
  # isn't supported in CI yet.
  #
  add_subdirectory(miniapp)
endif()


# ==================================================================
# install target(s)
# ==================================================================

include(GNUInstallDirs)

set(DPRT_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/dprt-${DPRT_VERSION}#${PROJECT_VERSION}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/dprt/dprtConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/dprtConfig.cmake"
  INSTALL_DESTINATION
  ${DPRT_CMAKE_INSTALL_DESTINATION}
)

message("DPRT_VERSION ${DPRT_VERSION}")
write_basic_package_version_file(
    "dprtConfigVersion.cmake"
    VERSION ${DPRT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/dprtConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/dprtConfigVersion.cmake
  DESTINATION
  ${DPRT_CMAKE_INSTALL_DESTINATION}
  )


install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cmake
  DESTINATION
    ${DPRT_CMAKE_INSTALL_DESTINATION}
  FILES_MATCHING
    PATTERN *.cmake
    PATTERN Finddprt.cmake EXCLUDE
)

install(
  TARGETS
  dprt
  EXPORT dprt-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT dprt-config
  DESTINATION ${DPRT_CMAKE_INSTALL_DESTINATION}
  NAMESPACE dprt::
  )

configure_file(
  # the _source_ file still has old name, but this file will only
  # ever get seen by cmake, so this can be renamed at a later point.
  ${PROJECT_SOURCE_DIR}/include/dprt/dprt.in.h
  include/dprt/dprt.h
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/dprt
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
)


