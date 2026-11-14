CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

PROJECT (llm_system)


if(WIN32)
  add_library(llm_system STATIC)
  add_compile_definitions(FMT_CONSTEVAL=)
else()
  add_library(llm_system SHARED)
  add_compile_options(-fPIC)
endif()


include_directories(src)
include_directories(src/dram/ramulator2/src)
add_subdirectory(src)
add_subdirectory(eval)


target_link_libraries(
    llm_system
    PRIVATE hardware
    PRIVATE module
    PRIVATE model
    PRIVATE scheduler
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

configure_file(dram_config_HBM3_80GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3_80GB.yaml COPYONLY)
configure_file(dram_config_HBM3E_192GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3E_192GB.yaml COPYONLY)
if(WIN32)
  target_link_libraries(llm_system PRIVATE dram)
  target_link_libraries(run PRIVATE "-Wl,--whole-archive" "$<TARGET_FILE:dram>" "-Wl,--no-whole-archive")
  # Static factory registrations must be retained by the executable's final link.
  get_target_property(ramulator_parts dram LINK_LIBRARIES)
  foreach(part IN LISTS ramulator_parts)
    if(TARGET "${part}")
      get_target_property(part_type "${part}" TYPE)
      if(part_type STREQUAL "OBJECT_LIBRARY")
        target_sources(dram PRIVATE $<TARGET_OBJECTS:${part}>)
      endif()
    endif()
  endforeach()
else()
  target_link_libraries(llm_system PRIVATE -Wl,--whole-archive dram -Wl,--no-whole-archive)
endif()

configure_file(config.yaml ${EXECUTABLE_OUTPUT_PATH}/config.yaml COPYONLY)