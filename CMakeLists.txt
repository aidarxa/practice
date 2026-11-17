cmake_minimum_required(VERSION 3.16)

project(crystal-pcuda LANGUAGES CXX)

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

option(CRYSTAL_BUILD_GPU "Build the PCUDA runtime, services and db_cli" ON)
if(CRYSTAL_BUILD_GPU)
    find_package(AdaptiveCpp REQUIRED)
endif()

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
add_library(cli_startup_timing STATIC src/cli/startup_timing.cpp)
target_include_directories(cli_startup_timing PUBLIC include)
# Reuse the independent native spawn/pipe observer; no Python timer in the measured chain.
add_executable(pcuda_acceptance_first_result test/pcuda_compile_perf/first_result.cpp)
add_library(checked_sizes STATIC src/core/size_expression.cpp)
target_include_directories(checked_sizes PUBLIC include)
add_subdirectory(bench/loader)
add_library(query_values STATIC src/core/parameter_value.cpp src/core/expression_codegen.cpp src/core/result_layout.cpp)
target_include_directories(query_values PUBLIC include)
target_link_libraries(query_values PUBLIC checked_sizes ssb_storage)
add_library(physical_execution STATIC
    src/core/execution_plan.cpp
    src/core/physical_planner.cpp
    src/core/content_hash.cpp
    src/core/code_symbols.cpp
    src/core/module_codegen.cpp
    src/core/aggregate_codegen.cpp
    src/core/text_plan.cpp
    src/core/text_plan_join.cpp
    src/core/text_plan_group.cpp
    src/core/text_plan_sort.cpp
    src/core/text_plan_memory.cpp
)
target_include_directories(physical_execution PUBLIC include)
target_link_libraries(physical_execution PUBLIC query_values)
find_package(Threads REQUIRED)
add_library(pcuda_modules STATIC src/core/module_compiler.cpp src/core/module_cache.cpp src/core/module_build.cpp src/core/config.cpp)
target_link_libraries(pcuda_modules PUBLIC physical_execution ${CMAKE_DL_LIBS} Threads::Threads)

if(CRYSTAL_BUILD_GPU)
include(cmake/pcuda_jit.cmake)

# Resource management is ordinary C++17; only kernel translation units need acpp.
add_library(pcuda_runtime STATIC src/core/pcuda_runtime.cpp src/core/pcuda_compat.cpp src/core/memory_guard.cpp src/core/text_storage.cpp)
target_include_directories(pcuda_runtime PUBLIC include)
target_link_libraries(pcuda_runtime PUBLIC checked_sizes AdaptiveCpp::acpp-rt)
add_library(pcuda_services STATIC src/core/pcuda_services.cpp)
target_link_libraries(pcuda_services PUBLIC pcuda_runtime)
add_sycl_to_target(TARGET pcuda_services)
set_property(TARGET pcuda_services PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")
add_library(pcuda_scan STATIC src/core/pcuda_scan.cpp)
target_link_libraries(pcuda_scan PUBLIC pcuda_runtime)
add_sycl_to_target(TARGET pcuda_scan)
set_property(TARGET pcuda_scan PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")
add_library(pcuda_plan_services STATIC src/core/pcuda_plan_services.cpp)
target_link_libraries(pcuda_plan_services PUBLIC pcuda_scan)
add_sycl_to_target(TARGET pcuda_plan_services)
set_property(TARGET pcuda_plan_services PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")
add_library(pcuda_plan_runtime STATIC src/core/pcuda_plan_runtime.cpp)
target_link_libraries(pcuda_plan_runtime PUBLIC pcuda_plan_services pcuda_modules)
add_library(pcuda_text_fetch STATIC src/core/text_fetch.cpp)
target_link_libraries(pcuda_text_fetch PUBLIC pcuda_scan)
add_sycl_to_target(TARGET pcuda_text_fetch)
set_property(TARGET pcuda_text_fetch PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")

# Independent handwritten algorithms: no SQL parser/codegen/executor linkage.
add_library(manual_ssb STATIC test/queries/example/support/runner.cpp test/queries/example/support/dataset.cpp)
target_link_libraries(manual_ssb PUBLIC pcuda_scan manual_ssb_spec ssb_storage)
target_sources(manual_ssb PRIVATE src/core/config.cpp)
add_sycl_to_target(TARGET manual_ssb)
set_property(TARGET manual_ssb PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")
foreach(query 11 12 13 21 22 23 31 32 33 34 41 42 43)
    add_executable(ssb_q${query} test/queries/example/q${query}.cpp)
    target_link_libraries(ssb_q${query} PRIVATE manual_ssb)
endforeach()
add_executable(test_manual_ssb_gpu test/pcuda_stage7/test_manual_gpu.cpp)
target_link_libraries(test_manual_ssb_gpu PRIVATE manual_ssb)
target_compile_options(test_manual_ssb_gpu PRIVATE -UNDEBUG)

# Historical stage-2 ABI-negative fixture only; not linked into the engine or new modules.
add_library(legacy_plan_runtime STATIC EXCLUDE_FROM_ALL src/core/legacy_plan_runtime.cpp)
target_link_libraries(legacy_plan_runtime PUBLIC pcuda_runtime)
add_sycl_to_target(TARGET legacy_plan_runtime)
set_property(TARGET legacy_plan_runtime PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")

add_executable(db_cli
    src/main.cpp
    src/cli/meta_command_handler.cpp
    src/cli/terminal_app.cpp
    src/core/execution.cpp
    src/core/config.cpp
    src/core/translator.cpp
    src/core/query_template.cpp
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
    cli_startup_timing
    physical_execution
    pcuda_plan_runtime
    pcuda_services
    pcuda_text_fetch
    linenoise
    sqlparser
    dl
)
add_dependencies(db_cli pcuda_jit_pch)
set_target_properties(db_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;$<TARGET_FILE_DIR:AdaptiveCpp::acpp-rt>")
install(TARGETS db_cli RUNTIME DESTINATION bin)
install(FILES "${CMAKE_SOURCE_DIR}/deps/lib/libsqlparser.so" DESTINATION lib)

target_compile_definitions(db_cli PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)

# Historical ABI-negative adapter fixture, never a production engine dependency.
add_executable(test_gpu_plan
    test/pcuda_stage2/test_gpu_plan.cpp
    src/core/execution.cpp src/core/config.cpp
    src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp
)
target_include_directories(test_gpu_plan PRIVATE include deps/include)
target_link_directories(test_gpu_plan PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_gpu_plan PRIVATE pcuda_plan_runtime legacy_plan_runtime sqlparser dl)
add_dependencies(test_gpu_plan pcuda_jit_pch)
target_compile_options(test_gpu_plan PRIVATE -UNDEBUG)
target_compile_definitions(test_gpu_plan PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)

# Additional hardware fixture uses the real SQL compiler/runtime, with independent tiny inputs.
add_executable(test_pcuda_primitives_sql test/pcuda_stage4/test_gpu_sql.cpp
    src/core/execution.cpp src/core/config.cpp src/core/translator.cpp src/core/query_template.cpp
    src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_primitives_sql PRIVATE include deps/include)
target_link_directories(test_pcuda_primitives_sql PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_primitives_sql PRIVATE pcuda_plan_runtime sqlparser dl)
add_dependencies(test_pcuda_primitives_sql pcuda_jit_pch)
target_compile_options(test_pcuda_primitives_sql PRIVATE -UNDEBUG)
target_compile_definitions(test_pcuda_primitives_sql PRIVATE
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}")
add_executable(test_pcuda_sql_modules test/pcuda_stage5/test_gpu_modules.cpp
    src/core/execution.cpp src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_sql_modules PRIVATE include deps/include)
target_link_directories(test_pcuda_sql_modules PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_sql_modules PRIVATE pcuda_plan_runtime sqlparser)
target_compile_options(test_pcuda_sql_modules PRIVATE -UNDEBUG)
add_dependencies(test_pcuda_sql_modules pcuda_jit_pch)
add_executable(test_pcuda_runtime_parameters test/pcuda_stage6/test_gpu_parameters.cpp
    src/core/execution.cpp src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_runtime_parameters PRIVATE include deps/include)
target_link_directories(test_pcuda_runtime_parameters PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_runtime_parameters PRIVATE pcuda_plan_runtime sqlparser)
target_compile_options(test_pcuda_runtime_parameters PRIVATE -UNDEBUG)
target_compile_definitions(test_pcuda_runtime_parameters PRIVATE CRYSTAL_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_dependencies(test_pcuda_runtime_parameters pcuda_jit_pch)
add_executable(test_pcuda_text_gpu test/pcuda_stage7_text/test_gpu.cpp
    src/core/execution.cpp src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_text_gpu PRIVATE include deps/include)
target_link_directories(test_pcuda_text_gpu PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_text_gpu PRIVATE pcuda_plan_runtime pcuda_text_fetch sqlparser)
target_compile_options(test_pcuda_text_gpu PRIVATE -UNDEBUG)
add_dependencies(test_pcuda_text_gpu pcuda_jit_pch)
add_executable(test_pcuda_text_resources test/pcuda_stage7_text/test_resources.cpp)
target_link_libraries(test_pcuda_text_resources PRIVATE pcuda_text_fetch query_values)
target_compile_options(test_pcuda_text_resources PRIVATE -UNDEBUG)
endif()

# --- Test target (AST → CodeGen, no SYCL) ---
add_executable(test_optimizer
    test/test_optimizer.cpp
    src/core/translator.cpp
    src/core/query_template.cpp
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
    physical_execution
    sqlparser
)

# The optimizer lowers strings/contracts only; no GPU compiler/runtime is needed.
target_compile_definitions(test_optimizer PRIVATE
    CRYSTAL_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    CRYSTAL_INCLUDE_DIR_DEFAULT="${CRYSTAL_INCLUDE_DIR}"
    CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT="${CRYSTAL_DEPS_INCLUDE_DIR}"
)

# --- Test target (Expression Tree + Operator Tree + Translator) ---
add_executable(test_translator
    test/test_translator.cpp
    src/core/translator.cpp
    src/core/query_template.cpp
)

target_include_directories(test_translator PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/deps/include"
)

target_link_directories(test_translator PRIVATE
    "${CMAKE_SOURCE_DIR}/deps/lib"
)

target_link_libraries(test_translator PRIVATE
    query_values
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

# --- Test target (SQL three-valued logic truth table model) ---
add_executable(test_sql_3vl
    test/test_sql_3vl.cpp
)

# Assertions in tests must remain active in Release builds.
add_executable(test_query_template
    test/test_query_template.cpp
    src/core/translator.cpp
    src/core/query_template.cpp
    src/core/optimizer_rules.cpp
)
target_include_directories(test_query_template PRIVATE include deps/include)
target_link_directories(test_query_template PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_query_template PRIVATE sqlparser query_values)
target_compile_definitions(test_query_template PRIVATE CRYSTAL_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_testing()
add_executable(test_pcuda_acceptance_timing test/pcuda_stage8/test_timing.cpp)
target_link_libraries(test_pcuda_acceptance_timing PRIVATE cli_startup_timing)
target_compile_options(test_pcuda_acceptance_timing PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_acceptance_timing COMMAND test_pcuda_acceptance_timing)
add_executable(test_pcuda_text_values test/pcuda_stage7_text/test_values.cpp)
target_link_libraries(test_pcuda_text_values PRIVATE query_values)
target_compile_options(test_pcuda_text_values PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_text_values COMMAND test_pcuda_text_values "${CMAKE_CURRENT_BINARY_DIR}/text-values-fixture.bin")
add_executable(test_pcuda_text_frontend test/pcuda_stage7_text/test_frontend.cpp
    src/core/translator.cpp src/core/query_template.cpp)
target_include_directories(test_pcuda_text_frontend PRIVATE include deps/include)
target_link_directories(test_pcuda_text_frontend PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_text_frontend PRIVATE query_values sqlparser)
target_compile_options(test_pcuda_text_frontend PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_text_frontend COMMAND test_pcuda_text_frontend)
add_library(manual_ssb_spec STATIC test/queries/example/support/query_spec.cpp)
target_include_directories(manual_ssb_spec PUBLIC include test/queries/example)
target_link_libraries(manual_ssb_spec PUBLIC checked_sizes)
add_executable(test_manual_ssb_spec test/pcuda_stage7/test_manual_spec.cpp)
target_link_libraries(test_manual_ssb_spec PRIVATE manual_ssb_spec)
target_compile_options(test_manual_ssb_spec PRIVATE -UNDEBUG)
add_test(NAME test_manual_ssb_spec COMMAND test_manual_ssb_spec)
add_executable(test_pcuda_storage test/pcuda_stage7/test_storage.cpp)
target_include_directories(test_pcuda_storage PRIVATE include)
target_compile_options(test_pcuda_storage PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_storage COMMAND test_pcuda_storage)
add_executable(test_pcuda_storage_frontend test/pcuda_stage7/test_frontend.cpp
    src/core/translator.cpp src/core/query_template.cpp)
target_include_directories(test_pcuda_storage_frontend PRIVATE include deps/include)
target_link_directories(test_pcuda_storage_frontend PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_storage_frontend PRIVATE query_values sqlparser)
target_compile_options(test_pcuda_storage_frontend PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_storage_frontend COMMAND test_pcuda_storage_frontend)
add_executable(test_pcuda_primitives_host test/pcuda_stage4/test_primitives.cpp src/core/size_expression.cpp)
target_include_directories(test_pcuda_primitives_host PRIVATE include)
target_compile_options(test_pcuda_primitives_host PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_primitives_host COMMAND test_pcuda_primitives_host)
if(CRYSTAL_BUILD_GPU)
add_executable(test_pcuda_primitives_gpu test/pcuda_stage4/test_gpu_primitives.cpp)
target_link_libraries(test_pcuda_primitives_gpu PRIVATE pcuda_scan)
target_compile_options(test_pcuda_primitives_gpu PRIVATE -UNDEBUG)
add_sycl_to_target(TARGET test_pcuda_primitives_gpu)
set_property(TARGET test_pcuda_primitives_gpu PROPERTY RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE} --acpp-pcuda")
add_executable(test_pcuda_runtime
    test/pcuda_stage3/test_runtime.cpp test/pcuda_stage3/fake_pcuda.cpp
    src/core/pcuda_runtime.cpp src/core/memory_guard.cpp src/core/size_expression.cpp)
target_include_directories(test_pcuda_runtime PRIVATE include
    $<TARGET_PROPERTY:AdaptiveCpp::acpp-rt,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(test_pcuda_runtime PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_runtime COMMAND test_pcuda_runtime)
add_executable(test_pcuda_storage_config test/pcuda_stage3/test_storage_config.cpp src/core/config.cpp)
target_include_directories(test_pcuda_storage_config PRIVATE include)
target_compile_options(test_pcuda_storage_config PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_storage_config COMMAND test_pcuda_storage_config)
# Manual hardware fixture; keep GPU tests sequential and separate from CPU CTest.
add_executable(test_pcuda_gpu_runtime test/pcuda_stage3/test_gpu_runtime.cpp)
target_link_libraries(test_pcuda_gpu_runtime PRIVATE pcuda_services)
target_compile_options(test_pcuda_gpu_runtime PRIVATE -UNDEBUG)
endif()
add_executable(test_module_abi test/test_module_abi.cpp)
target_include_directories(test_module_abi PRIVATE include)
add_executable(test_execution_plan test/test_execution_plan.cpp)
target_link_libraries(test_execution_plan PRIVATE physical_execution)
foreach(test_target test_translator test_optimizer test_nullable_semantics test_sql_3vl test_query_template test_module_abi test_execution_plan)
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()

add_executable(test_pcuda_parameter_plan test/pcuda_stage5/test_parameter_plan.cpp
    src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_parameter_plan PRIVATE include deps/include)
target_link_directories(test_pcuda_parameter_plan PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_parameter_plan PRIVATE physical_execution sqlparser)
target_compile_options(test_pcuda_parameter_plan PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_parameter_plan COMMAND test_pcuda_parameter_plan)
add_executable(test_pcuda_modules test/pcuda_stage5/test_modules.cpp)
target_link_libraries(test_pcuda_modules PRIVATE pcuda_modules)
target_compile_options(test_pcuda_modules PRIVATE -UNDEBUG)
target_compile_definitions(test_pcuda_modules PRIVATE DB_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_pcuda_modules COMMAND test_pcuda_modules)
add_executable(test_pcuda_runtime_contracts test/pcuda_stage6/test_contracts.cpp
    src/core/translator.cpp src/core/query_template.cpp src/core/optimizer_rules.cpp src/core/visitor.cpp)
target_include_directories(test_pcuda_runtime_contracts PRIVATE include deps/include)
target_link_directories(test_pcuda_runtime_contracts PRIVATE "${CMAKE_SOURCE_DIR}/deps/lib")
target_link_libraries(test_pcuda_runtime_contracts PRIVATE physical_execution sqlparser)
target_compile_options(test_pcuda_runtime_contracts PRIVATE -UNDEBUG)
add_test(NAME test_pcuda_runtime_contracts COMMAND test_pcuda_runtime_contracts)
