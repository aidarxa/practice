#pragma once
/** @file @brief A Release test must fail to build rather than silently discard assert-based checks. */
#ifdef NDEBUG
#error "This behavioral test requires active assertions; compile with -UNDEBUG after Release flags"
#endif
