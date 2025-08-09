set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

message(STATUS "Executing unit tests")

include(FetchContent)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12
)

FetchContent_MakeAvailable(doctest)

file(GLOB_RECURSE TEST_SOURCES
  CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp
)

add_executable(NeuroForgeTests ${TEST_SOURCES})

target_link_libraries(NeuroForgeTests
  PRIVATE
  doctest::doctest
  NeuroForge
)

enable_testing()
add_test(NAME NeuroForgeTests
  COMMAND NeuroForgeTests --success --durations yes --exitcode
)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "Test coverage activated")

  target_compile_options(NeuroForge PUBLIC --coverage -O0 -g)
  target_link_options(NeuroForge PUBLIC --coverage)

  target_compile_options(NeuroForgeTests PRIVATE --coverage -O0 -g)
  target_link_options(NeuroForgeTests PRIVATE --coverage)
endif()
