cmake_minimum_required(VERSION 3.30)

project(practice)

set(CMAKE_CXX_STANDART 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ACPP_SOURCE_ROOT}/cmake)

set(SRC_FILES 
    "src/main.cpp")

set(HEADER_FILES
)
set(SYCL_FILES
    "src/main.cpp"
)

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

add_executable(practice ${SRC_FILES} ${HEADER_FILES})
add_sycl_to_target(TARGET practice SOURCES ${SYCL_FILES})