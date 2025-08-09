cmake_minimum_required(VERSION 3.15)
project(NeuroForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/packages/NeuroForge-${PROJECT_VERSION}" CACHE PATH "Install path prefix" FORCE)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ENVIRONMENT "production" CACHE STRING "Define the environment")

file(GLOB_RECURSE LIB_SOURCES
  CONFIGURE_DEPENDS
  src/neuro/*.cpp
)

add_library(NeuroForge STATIC ${LIB_SOURCES})

target_include_directories(NeuroForge
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/internal
)

if(ENVIRONMENT STREQUAL "production")
  message(STATUS "Building on production environment")
  target_compile_definitions(NeuroForge PUBLIC ENVIRONMENT_PRODUCTION)

  set(INSTALL_TARGET ON)

elseif(ENVIRONMENT STREQUAL "tests")
  message(STATUS "Building on tests environment")
  target_compile_definitions(NeuroForge
    PUBLIC
    ENVIRONMENT_TESTS
    LOG_LEVEL_INFO LOG_LEVEL_DEBUG LOG_LEVEL_WARNING LOG_LEVEL_ERROR
  )

  set(ENABLE_TEST ON)
  set(VERBOSE_CODE_COMPILATION ON)
  set(ENABLE_COVERAGE ON)

elseif(ENVIRONMENT STREQUAL "development")
  message(STATUS "Building on development environment")
  target_compile_definitions(NeuroForge
    PUBLIC
    ENVIRONMENT_DEVELOPMENT
    LOG_LEVEL_INFO LOG_LEVEL_DEBUG LOG_LEVEL_WARNING LOG_LEVEL_ERROR
  )

  set(VERBOSE_CODE_COMPILATION ON)
  set(MAKE_EXECUTABLE ON)

endif()

if(VERBOSE_CODE_COMPILATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")
endif()

if(INSTALL_TARGET)
  add_library(NeuroForge::NeuroForge ALIAS NeuroForge)

  install(TARGETS NeuroForge
    EXPORT NeuroForgeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  install(DIRECTORY include/neuro/
    DESTINATION include/neuro
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
endif()

if(ENABLE_TEST)
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()

if(MAKE_EXECUTABLE)
  add_executable(NeuroForgeMain ${PROJECT_SOURCE_DIR}/src/main.cpp)

  target_link_libraries(NeuroForgeMain
    PRIVATE
    NeuroForge
  )
endif()
