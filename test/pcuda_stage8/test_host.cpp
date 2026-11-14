/** @file @brief One instrumented executable avoids multi-main LLVM profile identity collisions. */
int startupTimingChecks();
int moduleChecks(int, char**);
int main(int argc, char** argv) {
    if (argc > 1) return moduleChecks(argc, argv);
    startupTimingChecks();
    return moduleChecks(argc, argv);
}
