file(GLOB_RECURSE TEST_SOURCES
  CONFIGURE_DEPENDS
  *.cpp
)

add_executable(NeuroForgeTests ${TEST_SOURCES})

target_link_libraries(NeuroForgeTests
  PRIVATE
  Catch2::Catch2WithMain
)

target_include_directories(NeuroForgeTests
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)

include(CTest)
include(Catch)
catch_discover_tests(NeuroForgeTests)
