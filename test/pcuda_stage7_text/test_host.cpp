/** @file @brief One instrumented client of real storage/value/frontend code; not device coverage. */
int textValueChecks(int argc,char** argv);
int textFrontendChecks();
int main(int argc,char** argv) {
    if (textValueChecks(argc,argv)) return 1;
    return textFrontendChecks();
}
