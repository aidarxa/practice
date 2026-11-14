// Native process-start boundary: spawn the host, stop on its CPU-result marker.
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: first_result TIMING.json PROGRAM ARGS...");
        int pipefd[2];
        if (pipe(pipefd)) throw std::runtime_error("pipe failed");
        const auto fd = std::to_string(pipefd[1]);
        if (setenv("PERF_READY_FD", fd.c_str(), 1)) throw std::runtime_error("setenv failed");
        posix_spawn_file_actions_t actions;
        if (posix_spawn_file_actions_init(&actions)) throw std::runtime_error("spawn actions failed");
        posix_spawn_file_actions_addclose(&actions, pipefd[0]);
        pid_t pid;
        const auto start = std::chrono::steady_clock::now();
        const int error = posix_spawn(&pid, argv[2], &actions, nullptr, argv + 2, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(pipefd[1]);
        if (error) { close(pipefd[0]); throw std::runtime_error("spawn failed: " + std::to_string(error)); }
        char marker = 0;
        ssize_t bytes;
        do { bytes = read(pipefd[0], &marker, 1); } while (bytes < 0 && errno == EINTR);
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        close(pipefd[0]);
        int status;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) || bytes != 1 || marker != 'R')
            throw std::runtime_error("host failed or first-result marker absent");
        std::ofstream output(argv[1]);
        output << std::setprecision(12) << "{\"process_start_to_first_result_ms\":" << ms << "}\n";
        if (!output) throw std::runtime_error("cannot write first-result timing");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
