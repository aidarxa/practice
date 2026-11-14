/** @file @brief Behavioral checks of opt-in first-result timing, independent of GPU/SQL execution. */
#include "cli/startup_timing.h"
#include "../assertions_enabled.h"
#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

/** @brief The caller grants ownership of one pipe writer; no marker precedes CPU-ready. */
static void firstResult() {
    int pipefd[2];
    assert(pipe(pipefd) == 0);
    assert(setenv("PERF_READY_FD", std::to_string(pipefd[1]).c_str(), 1) == 0);
    {
        db::cli::StartupTiming timing;
        assert(fcntl(pipefd[1], F_GETFD) & FD_CLOEXEC);
        timing.databaseReady();
        timing.dataReady();
        assert(timing.resultReady());
        assert(!timing.resultReady());
        const auto& stats = timing.stats();
        assert(stats.startup_ms >= 0 && stats.data_load_ms >= 0);
        assert(stats.main_to_first_result_ms >= stats.startup_ms + stats.data_load_ms);
        char marker = 0;
        assert(read(pipefd[0], &marker, 1) == 1 && marker == 'R');
        assert(read(pipefd[0], &marker, 1) == 0);
    }
    assert(close(pipefd[0]) == 0);
    unsetenv("PERF_READY_FD");
}

/** @brief Failure/teardown closes the writer without falsely announcing a successful query. */
static void noResult() {
    int pipefd[2];
    assert(pipe(pipefd) == 0);
    setenv("PERF_READY_FD", std::to_string(pipefd[1]).c_str(), 1);
    { db::cli::StartupTiming timing; timing.databaseReady(); timing.dataReady(); }
    char marker = 0;
    assert(read(pipefd[0], &marker, 1) == 0);
    assert(close(pipefd[0]) == 0);
    unsetenv("PERF_READY_FD");
    db::cli::StartupTiming ordinary;
    ordinary.databaseReady(); ordinary.dataReady();
    assert(ordinary.resultReady());
}

/** @brief Parse an opt-in descriptor once, rejecting stdio, garbage and closed descriptors. */
static void invalidDescriptors() {
    for (const auto* value : {"", "0", "1", "2", "-1", "+3", "3x", " 3", "999999999999999999999", "999999"}) {
        setenv("PERF_READY_FD", value, 1);
        bool rejected = false;
        try { db::cli::StartupTiming timing; } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    int pipefd[2];
    assert(pipe(pipefd) == 0);
    const auto malformed = std::to_string(pipefd[1]) + "x";
    setenv("PERF_READY_FD", malformed.c_str(), 1);
    bool rejected = false;
    try { db::cli::StartupTiming timing; } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected); // The valid numeric prefix must not hide a failed whole-string parse.
    assert(close(pipefd[0]) == 0 && close(pipefd[1]) == 0);
    unsetenv("PERF_READY_FD");
}

/** @brief Invalid boundary order must not produce a plausible but false timing sample. */
static void invalidOrder() {
    const auto rejects = [](auto action) {
        bool rejected = false;
        try { action(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    };
    db::cli::StartupTiming timing;
    rejects([&] { timing.dataReady(); });
    rejects([&] { timing.resultReady(); });
    timing.databaseReady();
    rejects([&] { timing.databaseReady(); });
    timing.dataReady();
    rejects([&] { timing.dataReady(); });
    assert(timing.resultReady());
}

#ifdef DB_ACCEPTANCE_COMBINED_TEST
int startupTimingChecks() {
#else
int main() {
#endif
    firstResult(); noResult(); invalidDescriptors(); invalidOrder();
    std::cout << "First CPU-ready marker, startup boundaries and descriptor lifetime passed\n";
    return 0;
}
