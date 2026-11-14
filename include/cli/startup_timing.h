#pragma once
/** @file @brief Optional process-first-result observation; never a query timer or SQL executor. */
#include <chrono>

namespace db::cli {
/** @brief Disjoint main-entry startup/data intervals and a direct main-entry-to-CPU-ready interval, in ms. */
struct StartupTimingStats {
    double startup_ms = 0;
    double data_load_ms = 0;
    double main_to_first_result_ms = 0;
};

/**
 * @brief Observe startup once and optionally notify a native parent through PERF_READY_FD.
 * @details Construct at main entry, mark database construction and completed data loading,
 * then announce only a successfully returned CPU QueryResult, before formatting/printing.
 * The inherited pipe writer is opt-in, must have a live reader, and becomes this object's
 * responsibility. CLOEXEC prevents compiler/linker children from keeping it alive.
 * Without the environment variable no descriptor is opened and no marker is written.
 * The native parent measures spawn-to-marker separately, including loader and marker delivery.
 */
class StartupTiming {
public:
    StartupTiming();
    ~StartupTiming() noexcept;
    StartupTiming(const StartupTiming&) = delete;
    StartupTiming& operator=(const StartupTiming&) = delete;
    void databaseReady();
    void dataReady();
    /** @return True exactly once; later successful results do not overwrite the first interval. */
    bool resultReady();
    const StartupTimingStats& stats() const { return stats_; }
private:
    using Clock = std::chrono::steady_clock;
    enum class Phase { Startup, Loading, Query, Complete };
    const Clock::time_point start_ = Clock::now();
    Clock::time_point database_ready_;
    StartupTimingStats stats_;
    Phase phase_ = Phase::Startup;
    int ready_fd_ = -1;
};
} // namespace db::cli
