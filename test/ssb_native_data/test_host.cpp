/** @file @brief One host test executable avoids unrelated main() LLVM profile collisions. */
int datasetChecks();
int manualSpecChecks();
int main() {
    if (datasetChecks()) return 1;
    return manualSpecChecks();
}
