include(FetchContent)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12
)

FetchContent_MakeAvailable(doctest)

file(GLOB_RECURSE TEST_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
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
