cmake_minimum_required(VERSION 3.15)
project(NeuroForge VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

file(GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  src/*.cpp
)

option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

if(ENABLE_ASAN)
  message("-- Debug ON")
  add_compile_options(-fsanitize=address -g)
  add_link_options(-fsanitize=address)
endif()

add_executable(NeuroForge ${SOURCES})
