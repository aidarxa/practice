set -u
export LC_ALL=C
TIMEFORMAT='%6R %6U %6S'
pids=()
failed=0
unset 'ACPP_CLANG'
unset 'ACPP_CLANG_INCLUDE_PATH'
unset 'ACPP_CONFIG_FILE_DIR'
unset 'ACPP_CPU_CXX'
unset 'ACPP_CUDA_CXX_FLAGS'
unset 'ACPP_CUDA_LINK_LINE'
unset 'ACPP_CUDA_PATH'
unset 'ACPP_DEPLOY'
unset 'ACPP_DRYRUN'
unset 'ACPP_DRYRUN_ONLYSTDFLAGS'
unset 'ACPP_EXPLICIT_MULTIPASS'
unset 'ACPP_EXPORT_ALL'
unset 'ACPP_GPU_ARCH'
unset 'ACPP_NO_WARN_LEGACY_FLOWS'
unset 'ACPP_NVCXX'
unset 'ACPP_OMP_CXX_FLAGS'
unset 'ACPP_OMP_LINK_LINE'
unset 'ACPP_PCUDA'
unset 'ACPP_PCUDA_CHEVRON_LAUNCH'
unset 'ACPP_PLATFORM'
unset 'ACPP_ROCM_CXX_FLAGS'
unset 'ACPP_ROCM_LINK_LINE'
unset 'ACPP_ROCM_PATH'
unset 'ACPP_SAVE_TEMPS'
unset 'ACPP_SEQUENTIAL_CXX_FLAGS'
unset 'ACPP_SEQUENTIAL_LINK_LINE'
unset 'ACPP_STDPAR'
unset 'ACPP_STDPAR_MQS'
unset 'ACPP_STDPAR_PREFETCH_MODE'
unset 'ACPP_STDPAR_SYSTEM_USM'
unset 'ACPP_STDPAR_UNCONDITIONAL_OFFLOAD'
unset 'ACPP_TARGETS'
unset 'ACPP_USE_ACCELERATED_CPU'
unset 'COMPILER_PATH'
unset 'CPATH'
unset 'CPLUS_INCLUDE_PATH'
unset 'GCC_EXEC_PREFIX'
unset 'HIPSYCL_CLANG'
unset 'HIPSYCL_CLANG_INCLUDE_PATH'
unset 'HIPSYCL_CONFIG_FILE_DIR'
unset 'HIPSYCL_CPU_CXX'
unset 'HIPSYCL_CUDA_CXX_FLAGS'
unset 'HIPSYCL_CUDA_LINK_LINE'
unset 'HIPSYCL_CUDA_PATH'
unset 'HIPSYCL_DEPLOY'
unset 'HIPSYCL_DRYRUN'
unset 'HIPSYCL_DRYRUN_ONLYSTDFLAGS'
unset 'HIPSYCL_EXPLICIT_MULTIPASS'
unset 'HIPSYCL_EXPORT_ALL'
unset 'HIPSYCL_GPU_ARCH'
unset 'HIPSYCL_NO_WARN_LEGACY_FLOWS'
unset 'HIPSYCL_NVCXX'
unset 'HIPSYCL_OMP_CXX_FLAGS'
unset 'HIPSYCL_OMP_LINK_LINE'
unset 'HIPSYCL_PCUDA'
unset 'HIPSYCL_PCUDA_CHEVRON_LAUNCH'
unset 'HIPSYCL_PLATFORM'
unset 'HIPSYCL_ROCM_CXX_FLAGS'
unset 'HIPSYCL_ROCM_LINK_LINE'
unset 'HIPSYCL_ROCM_PATH'
unset 'HIPSYCL_SAVE_TEMPS'
unset 'HIPSYCL_SEQUENTIAL_CXX_FLAGS'
unset 'HIPSYCL_SEQUENTIAL_LINK_LINE'
unset 'HIPSYCL_STDPAR'
unset 'HIPSYCL_STDPAR_MQS'
unset 'HIPSYCL_STDPAR_PREFETCH_MODE'
unset 'HIPSYCL_STDPAR_SYSTEM_USM'
unset 'HIPSYCL_STDPAR_UNCONDITIONAL_OFFLOAD'
unset 'HIPSYCL_TARGETS'
unset 'HIPSYCL_USE_ACCELERATED_CPU'
export 'LD_LIBRARY_PATH=/opt/rocm_sdk_612/x86_64/lib:/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/x86_64/lib:/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/apps/aie/xrt/lib:/opt/rocm_sdk_612/apps/aie/lib:/opt/rocm_sdk_612/apps/aie/xrt/lib:/opt/rocm_sdk_612/apps/aie/lib'
unset 'LIBRARY_PATH'
export 'PATH=/opt/rocm_sdk_612/bin:/home/aidar/go/bin:/opt/rocm_sdk_612/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/lib/jvm/default/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl:/opt/AdaptiveCpp/bin/:/opt/rocm_sdk_612/apps/aie/xrt/bin:/opt/rocm_sdk_612/apps/aie/bin:/opt/AdaptiveCpp/bin/:/opt/rocm_sdk_612/apps/aie/xrt/bin:/opt/rocm_sdk_612/apps/aie/bin'
unset 'SDKROOT'
unset 'SOURCE_DATE_EPOCH'
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/4170886016dae97984e214ac1a7e674360871b1cf29c1a2368aaafd0bad14493.link.status'; exit "$status"; fi
) &
pids+=("$!")
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6.link.status'; exit "$status"; fi
) &
pids+=("$!")
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q11-plain/db-pcuda-dlaIHa/0/57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b.link.status'; exit "$status"; fi
) &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
if ((failed)); then exit 1; fi
