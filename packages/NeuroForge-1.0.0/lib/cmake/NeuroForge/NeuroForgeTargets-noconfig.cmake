#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "NeuroForge::NeuroForge" for configuration ""
set_property(TARGET NeuroForge::NeuroForge APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NeuroForge::NeuroForge PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libNeuroForge.a"
  )

list(APPEND _cmake_import_check_targets NeuroForge::NeuroForge )
list(APPEND _cmake_import_check_files_for_NeuroForge::NeuroForge "${_IMPORT_PREFIX}/lib/libNeuroForge.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
