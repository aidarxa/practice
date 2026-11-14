#pragma once

#include <cstddef>
#include <string>

namespace db {

struct CrystalConfig {
    std::string include_dir;
    std::string deps_include_dir;

    /** @brief Prebuilt compiler/flags/PCH/header manifest; no SQL-path PCH generation is allowed. */
    std::string pcuda_manifest;
    /** @brief Parent of private session artifacts; empty uses the platform temporary directory. */
    std::string jit_artifact_dir;
    std::size_t jit_jobs = 4;
    bool jit_keep_artifacts = false;
    bool jit_traces = false;
    /** @brief Research-only S1/execution overlap; disabled by default and uses a non-additive timing model. */
    bool jit_overlap_experiment = false;

    /** @brief Empty selects the first enumerated GPU; otherwise require one case-insensitive name match. */
    std::string device_name;

    bool dump_generated_code = false;
    bool extended_timing = false;
    bool memory_guard_enabled = true;
    bool reuse_scratch_buffers = true;

    std::size_t output_row_limit = 1000;
    bool output_row_limit_enabled = true;

    std::size_t memory_guard_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
    double memory_guard_reserve_fraction = 0.10;

    std::string source_path;
};

CrystalConfig loadCrystalConfig();

} // namespace db
