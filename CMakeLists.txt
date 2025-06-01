cmake_minimum_required(VERSION 3.15)
project(NeuroForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/packages/NeuroForge-${PROJECT_VERSION}" CACHE PATH "Install path prefix" FORCE)

option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

if(ENABLE_ASAN)
  message("-- ASAN ENABLED")
  add_compile_options(-fsanitize=address -g)
  add_link_options(-fsanitize=address)
endif()

file(GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  src/neuro/*.cpp
)

add_library(NeuroForge STATIC ${SOURCES})

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
