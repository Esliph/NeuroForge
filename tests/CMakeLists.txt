set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TEST_TYPE "unit" CACHE STRING "Test type: unit | perf | opt")
set(TEST_NAME "" CACHE STRING "Test name when test type is \"perf\" or \"opt\"")

message(STATUS "Executing test type ${TEST_TYPE} - test name \"${TEST_NAME}\"")

if(TEST_TYPE STREQUAL "unit")
  include(FetchContent)

  FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
  )

  FetchContent_MakeAvailable(doctest)

  file(GLOB_RECURSE TEST_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp
  )

  add_executable(NeuroForgeTests ${TEST_SOURCES})

  target_include_directories(NeuroForgeTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
  )

  target_link_libraries(NeuroForgeTests
    PRIVATE doctest::doctest
    PRIVATE NeuroForge
  )

  enable_testing()
  add_test(NAME NeuroForgeTests
    COMMAND NeuroForgeTests --success --durations yes --exitcode
  )

else()
  if(TEST_NAME STREQUAL "")
    message(FATAL_ERROR "To run performance or optimization tests you must enter the \"TEST_NAME\"")
  endif()

  if(TEST_TYPE STREQUAL "perf")
    set(TEST_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/performance/${TEST_NAME}.cpp)

  elseif(TEST_TYPE STREQUAL "opt")
    set(TEST_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/optimization/${TEST_NAME}.cpp)
  endif()

  if(NOT EXISTS ${TEST_FILE_PATH})
    message(FATAL_ERROR "Test \"${TEST_FILE_PATH}\" not found")
  endif()

  add_executable(${TEST_NAME} ${TEST_FILE_PATH})
  target_link_libraries(${TEST_NAME} PRIVATE NeuroForge)

endif()
