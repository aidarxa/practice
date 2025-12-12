cmake_minimum_required(VERSION 3.30)

project(practice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ACPP_SOURCE_ROOT}/cmake)

find_package(AdaptiveCpp CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
   set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

include_directories(${ACPP_BINARY_ROOT} ${ACPP_SOURCE_ROOT})

file(GLOB SRC_CPP_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

foreach(cpp_file IN LISTS SRC_CPP_FILES)
  get_filename_component(target_name "${cpp_file}" NAME_WE)

  add_executable(${target_name} "${cpp_file}")
  target_include_directories(${target_name} PRIVATE "${CMAKE_SOURCE_DIR}/include")

  add_sycl_to_target(TARGET ${target_name} SOURCES "${cpp_file}")
endforeach()