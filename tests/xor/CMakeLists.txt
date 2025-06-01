cmake_minimum_required(VERSION 3.15)
project(NeuroForgeTestXOR VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../packages/NeuroForge-${PROJECT_VERSION}")

find_package(NeuroForge REQUIRED)

add_executable(xor main.cpp)

target_link_libraries(xor PRIVATE NeuroForge::NeuroForge)
