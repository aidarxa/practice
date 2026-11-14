#pragma once

// Experimental process-local controls, NOT an AdaptiveCpp public API.
extern "C" {
void jit_poc_set_policy(int ir_level, int tool_level, int use_clang);
unsigned long long jit_poc_compile_count();
void jit_poc_arm_notification();
int jit_poc_wait_notification(unsigned timeout_ms);
int jit_poc_dump(const char* path);
}
