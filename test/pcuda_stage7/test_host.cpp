/** @file @brief One instrumented executable avoids merging unrelated main() profile identities. */
int storageChecks();
int manualSpecChecks();
int frontendChecks();
int main() {
    if (storageChecks()) return 1;
    if (manualSpecChecks()) return 1;
    return frontendChecks();
}
