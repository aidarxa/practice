#include "cli/startup_timing.h"
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace db::cli {
namespace {
/** @brief Parse the explicitly granted pipe writer, never borrowing stdin/stdout/stderr. */
int readyDescriptor() {
    const char* value = std::getenv("PERF_READY_FD");
    if (!value) return -1;
    int fd = -1;
    const auto end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, fd);
    if (parsed.ec != std::errc{} || parsed.ptr != end || fd <= STDERR_FILENO)
        throw std::runtime_error("PERF_READY_FD must name an inherited pipe writer above stderr");
    const int flags = fcntl(fd, F_GETFD);
    if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
        throw std::runtime_error("PERF_READY_FD is not an accessible inherited descriptor");
    return fd;
}
double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
/** @brief The one-byte notification follows CPU readiness, not query submission or display output. */
void notifyReady(int fd) {
    if (fd < 0) return;
    ssize_t bytes;
    do { bytes = write(fd, "R", 1); } while (bytes < 0 && errno == EINTR);
    if (bytes != 1) throw std::runtime_error("Cannot notify first CPU-ready result");
}
} // namespace

StartupTiming::StartupTiming() : ready_fd_(readyDescriptor()) {}
StartupTiming::~StartupTiming() noexcept { if (ready_fd_ >= 0) close(ready_fd_); }
void StartupTiming::databaseReady() {
    if (phase_ != Phase::Startup) throw std::runtime_error("Database-ready boundary is out of order");
    database_ready_ = Clock::now();
    stats_.startup_ms = milliseconds(database_ready_ - start_);
    phase_ = Phase::Loading;
}
void StartupTiming::dataReady() {
    if (phase_ != Phase::Loading) throw std::runtime_error("Data-ready boundary is out of order");
    stats_.data_load_ms = milliseconds(Clock::now() - database_ready_);
    phase_ = Phase::Query;
}
bool StartupTiming::resultReady() {
    if (phase_ == Phase::Complete) return false;
    if (phase_ != Phase::Query) throw std::runtime_error("CPU-ready boundary precedes data-ready");
    stats_.main_to_first_result_ms = milliseconds(Clock::now() - start_);
    notifyReady(ready_fd_);
    if (ready_fd_ >= 0) { close(ready_fd_); ready_fd_ = -1; }
    phase_ = Phase::Complete;
    return true;
}
} // namespace db::cli
