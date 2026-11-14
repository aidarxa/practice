// Native first-result timer, also retaining abnormal child exit status.
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
        if (argc < 3) throw std::runtime_error("usage: first_result JSON PROGRAM ARGS...");
        int pipefd[2];
        if (pipe(pipefd)) throw std::runtime_error("pipe");
        const auto fd = std::to_string(pipefd[1]);
        if (setenv("PERF_READY_FD", fd.c_str(), 1)) throw std::runtime_error("setenv");
        posix_spawn_file_actions_t actions;
        if (posix_spawn_file_actions_init(&actions)) throw std::runtime_error("spawn actions");
        posix_spawn_file_actions_addclose(&actions, pipefd[0]);
        pid_t pid;
        const auto start = std::chrono::steady_clock::now();
        const int error = posix_spawn(&pid, argv[2], &actions, nullptr, argv + 2, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(pipefd[1]);
        if (error) { close(pipefd[0]); throw std::runtime_error("spawn error " + std::to_string(error)); }
        char marker = 0;
        ssize_t bytes;
        do { bytes = read(pipefd[0], &marker, 1); } while (bytes < 0 && errno == EINTR);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        close(pipefd[0]);
        int status = 0;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        const bool ready = bytes == 1 && marker == 'R';
        const bool success = waited >= 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0 && ready;
        std::ofstream output(argv[1]);
        output << std::setprecision(12) << "{\"success\":" << (success ? "true" : "false")
            << ",\"ready\":" << (ready ? "true" : "false") << ",\"raw_wait_status\":" << status
            << ",\"exit_code\":" << (WIFEXITED(status) ? WEXITSTATUS(status) : -1)
            << ",\"signal\":" << (WIFSIGNALED(status) ? WTERMSIG(status) : 0)
            << ",\"process_start_to_first_result_ms\":";
        if (ready) output << elapsed;
        else output << "null";
        output << "}\n";
        if (!output) throw std::runtime_error("write first-result metadata");
        if (!success) std::cerr << "host failed: wait status " << status << ", CPU-ready " << ready << '\n';
        return success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
