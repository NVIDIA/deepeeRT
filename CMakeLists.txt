# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(DEEPEERT_VERSION_MAJOR 0)
set(DEEPEERT_VERSION_MINOR 3)
set(DEEPEERT_VERSION_PATCH 2)
set(DEEPEERT_VERSION ${DEEPEERT_VERSION_MAJOR}.${DEEPEERT_VERSION_MINOR}.${DEEPEERT_VERSION_PATCH})
  
set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0048 NEW)
project(deepeeRT VERSION ${DEEPEERT_VERSION} LANGUAGES C CXX CUDA)

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

option(DEEPEERT_USE_LOCAL_CUBQL "Use *local* cuBQL" OFF)
mark_as_advanced(DEEPEERT_USE_LOCAL_CUBQL)
if (DEEPEERT_USE_LOCAL_CUBQL)
  add_subdirectory(../cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
else()
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
endif()

add_subdirectory(dp)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  # don't build miniapp if this is a subproject
else()
#  add_subdirectory(miniapp)
endif()


# ==================================================================
# install target(s)
# ==================================================================

include(GNUInstallDirs)

set(DEEPEERT_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/deepeeRT-${DEEPEERT_VERSION}#${PROJECT_VERSION}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/dp/deepeeRTConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/deepeeRTConfig.cmake"
  INSTALL_DESTINATION
  ${DEEPEERT_CMAKE_INSTALL_DESTINATION}
)

message("DEEPEERT_VERSION ${DEEPEERT_VERSION}")
write_basic_package_version_file(
    "deepeeRTConfigVersion.cmake"
    VERSION ${DEEPEERT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/deepeeRTConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/deepeeRTConfigVersion.cmake
  DESTINATION
  ${DEEPEERT_CMAKE_INSTALL_DESTINATION}
  )


install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cmake
  DESTINATION
    ${DEEPEERT_CMAKE_INSTALL_DESTINATION}
  FILES_MATCHING
    PATTERN *.cmake
    PATTERN FinddeepeeRT.cmake EXCLUDE
)

install(
  TARGETS
  deepeeRT
#  EXPORT deepeeRT
  EXPORT deepeeRT-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT deepeeRT-config
#install(EXPORT deepeeRT
  DESTINATION ${DEEPEERT_CMAKE_INSTALL_DESTINATION}
  NAMESPACE deepeeRT::
  )

# configures the public api
configure_file(
  ${PROJECT_SOURCE_DIR}/include/deepeeRT/deepeeRT.in.h
  include/deepeeRT/deepeeRT.h
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/deepeeRT
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
)

