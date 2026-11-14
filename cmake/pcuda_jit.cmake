# Stock acpp only. The PCH is a build/install artifact, not a SQL-path cache.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(CRYSTAL_PCUDA_COMPILER NAMES acpp REQUIRED)
set(CRYSTAL_PCUDA_FLAGS "$ENV{CRYSTAL_PCUDA_FLAGS}" CACHE STRING "Additional PCUDA/PCH flags (CMake list, e.g. a compatible --gcc-install-dir)")
set(CRYSTAL_PCUDA_LINK_FLAGS "-fuse-ld=lld;-Wl,-O0" CACHE STRING "PCUDA shared-module link flags (CMake list)")
set(CRYSTAL_PCUDA_JIT_DIR "${CMAKE_CURRENT_BINARY_DIR}/pcuda-jit" CACHE PATH "Prebuilt PCUDA JIT artifact directory")
file(MAKE_DIRECTORY "${CRYSTAL_PCUDA_JIT_DIR}")
set(_pcuda_compile --acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC
    -Wno-narrowing -Wno-deprecated-declarations "-I${CRYSTAL_INCLUDE_DIR}" ${CRYSTAL_PCUDA_FLAGS})
set(_pcuda_link --acpp-targets=generic --acpp-pcuda ${CRYSTAL_PCUDA_FLAGS} ${CRYSTAL_PCUDA_LINK_FLAGS})
set(_pch "${CRYSTAL_PCUDA_JIT_DIR}/jit_headers.pch")
set(_manifest "${CRYSTAL_PCUDA_JIT_DIR}/toolchain.manifest")
set(_metadata_args)
foreach(_flag IN LISTS _pcuda_compile)
    list(APPEND _metadata_args "--compile-arg=${_flag}")
endforeach()
foreach(_flag IN LISTS _pcuda_link)
    list(APPEND _metadata_args "--link-arg=${_flag}")
endforeach()
set(_device_headers)
foreach(_header core/module_abi.h core/module_contract.h core/module_abi_v2.h core/module_contract_v2.h core/text.h core/pcuda_index.h core/tagged_numeric.h
        core/inline_math.h core/device_atomics.h crystal/device.h crystal/load.h crystal/pred.h
        crystal/join.h crystal/hash.h crystal/collectives.h)
    list(APPEND _device_headers "${CRYSTAL_INCLUDE_DIR}/${_header}")
endforeach()
set(_depfile_option)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20 OR CMAKE_GENERATOR MATCHES "Ninja")
    set(_depfile_option DEPFILE "${_pch}.d")
else()
    message(FATAL_ERROR "PCUDA PCH toolchain dependency tracking requires CMake >=3.20 or the Ninja generator")
endif()
add_custom_command(OUTPUT "${_pch}" "${_manifest}"
    COMMAND "${CRYSTAL_PCUDA_COMPILER}" ${_pcuda_compile} -fpch-instantiate-templates
        -c -MD -MF "${_pch}.d" -x c++-header "${CRYSTAL_INCLUDE_DIR}/jit_headers.h" -o "${_pch}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/pcuda_pch_manifest.py"
        --compiler "${CRYSTAL_PCUDA_COMPILER}" --pch "${_pch}" --depfile "${_pch}.d"
        --header "${CRYSTAL_INCLUDE_DIR}/jit_headers.h" --output "${_manifest}" ${_metadata_args}
    DEPENDS "${CRYSTAL_INCLUDE_DIR}/jit_headers.h" ${_device_headers} "${CRYSTAL_PCUDA_COMPILER}" "${CMAKE_SOURCE_DIR}/tools/pcuda_pch_manifest.py"
    ${_depfile_option}
    VERBATIM COMMENT "Preparing and fingerprinting the minimal PCUDA JIT PCH")
add_custom_target(pcuda_jit_pch DEPENDS "${_pch}" "${_manifest}")
target_compile_definitions(pcuda_modules PRIVATE CRYSTAL_PCUDA_MANIFEST_DEFAULT="${_manifest}")
install(DIRECTORY "${CRYSTAL_INCLUDE_DIR}/" DESTINATION include/crystal FILES_MATCHING PATTERN "*.h")
configure_file("${CMAKE_SOURCE_DIR}/cmake/install_pcuda_jit.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/install_pcuda_jit.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_pcuda_jit.cmake")
