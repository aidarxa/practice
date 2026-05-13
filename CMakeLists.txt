cmake_minimum_required(VERSION 3.16)

project(crystal-sycl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing -Wno-deprecated-declarations")

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

# --- linenoise (FetchContent) ---
include(FetchContent)
FetchContent_Declare(
    linenoise
    GIT_REPOSITORY https://github.com/antirez/linenoise.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(linenoise)

add_library(linenoise_lib STATIC ${linenoise_SOURCE_DIR}/linenoise.c)
target_include_directories(linenoise_lib PUBLIC ${linenoise_SOURCE_DIR})

find_package(AdaptiveCpp REQUIRED)

set(CRYSTAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" CACHE PATH
    "Include directory with Crystal project headers for JIT compilation.")
set(CRYSTAL_DEPS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/deps/include" CACHE PATH
    "Include directory with Crystal third-party headers for JIT compilation.")

foreach(path_var CRYSTAL_INCLUDE_DIR CRYSTAL_DEPS_INCLUDE_DIR)
    if("${${path_var}}" STREQUAL "")
        message(FATAL_ERROR "${path_var} must not be empty.")
    endif()
    if(NOT IS_DIRECTORY "${${path_var}}")
        message(FATAL_ERROR "${path_var} does not exist or is not a directory: ${${path_var}}")
    endif()
endforeach()

# --- REPL executable ---
add_executable(db_cli
    src/main.cpp
    src/cli/meta_command_handler.cpp
    src/cli/terminal_app.cpp
    src/core/execution.cpp
    src/core/config.cpp
    src/core/translator.cpp
    src/core/optimizer_rules.cpp
    src/core/visitor.cpp
    src/db/database_instance.cpp
)

target_include_directories(db_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/deps/include"
)

target_link_directories(db_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/deps/lib"
)

target_link_libraries(db_cli PRIVATE
    linenoise_lib
    sqlparser
    dl
)

add_sycl_to_target(TARGET db_cli)
target_compile_definitions(db_cli PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)

# --- Test target (AST → CodeGen, no SYCL) ---
add_executable(test_optimizer
    test/test_optimizer.cpp
    src/core/translator.cpp
    src/core/optimizer_rules.cpp
    src/core/visitor.cpp
)

target_include_directories(test_optimizer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/deps/include"
)

target_link_directories(test_optimizer PRIVATE
    "${CMAKE_SOURCE_DIR}/deps/lib"
)

target_link_libraries(test_optimizer PRIVATE
    sqlparser
)

add_sycl_to_target(TARGET test_optimizer)
target_compile_definitions(test_optimizer PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)

# --- Test target (Expression Tree + Operator Tree + Translator) ---
add_executable(test_translator
    test/test_translator.cpp
    src/core/translator.cpp
)

target_include_directories(test_translator PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/deps/include"
)

target_link_directories(test_translator PRIVATE
    "${CMAKE_SOURCE_DIR}/deps/lib"
)

target_link_libraries(test_translator PRIVATE
    sqlparser
)

target_compile_definitions(test_translator PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)


# --- Test target (nullable bitmap decoding and NULL ABI helpers) ---
add_executable(test_nullable_semantics
    test/test_nullable_semantics.cpp
)

target_include_directories(test_nullable_semantics PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)
