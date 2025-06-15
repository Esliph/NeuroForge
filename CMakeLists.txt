cmake_minimum_required(VERSION 3.15)
project(NeuroForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/packages/NeuroForge-${PROJECT_VERSION}" CACHE PATH "Install path prefix" FORCE)

option(CONSTRUCT_INSTALLATION "Build the lib installer" OFF)
option(ENABLE_TEST "Build unit tests" OFF)

file(GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  src/*.cpp
)

add_library(NeuroForge STATIC ${SOURCES})

if(CONSTRUCT_INSTALLATION)
  target_include_directories(NeuroForge
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  add_library(NeuroForge::NeuroForge ALIAS NeuroForge)

  install(TARGETS NeuroForge
    EXPORT NeuroForgeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  install(DIRECTORY include/
    DESTINATION include
  )

  install(EXPORT NeuroForgeTargets
    FILE NeuroForgeTargets.cmake
    NAMESPACE NeuroForge::
    DESTINATION lib/cmake/NeuroForge
  )

  include(CMakePackageConfigHelpers)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NeuroForgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NeuroForgeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/NeuroForge
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NeuroForgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NeuroForgeConfigVersion.cmake"
    DESTINATION lib/cmake/NeuroForge
  )
else()
  include_directories(include)
endif()

if(ENABLE_TEST)
  add_subdirectory(tests)
elseif(!CONSTRUCT_INSTALLATION)
  add_executable(CampoMinado ${SOURCES} src/main.cpp)
endif()
