/** @file @brief Manual PCUDA Q1.1: fused scan/filter/SUM, with the SF10 reference's 1993-only window.
 * @details The SQL corpus also includes 1994-01-01; that known difference is not hidden by this oracle.
 */
#include "support/runner.h"
int main(int argc, char** argv) { return ssb::run(11, argc, argv); }
