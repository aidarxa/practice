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
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/0bf0b2da66c668832e2af5da6cb76186e9892198ff5ec18ee138137df38c6639.link.status'; exit "$status"; fi
) &
pids+=("$!")
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34.link.status'; exit "$status"; fi
) &
pids+=("$!")
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/170f70d6c009f9b6bd0b0b23e7d198f3d5a5dcc330d7b33fcde18746fd39f725.link.status'; exit "$status"; fi
) &
pids+=("$!")
(
set -e
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '-std=c++17' '-O0' '-g0' '-fPIC' '-Wno-narrowing' '-Wno-deprecated-declarations' '-I/home/aidar/Проекты/practice/include' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-include-pch' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/engine/pcuda-jit/jit_headers.pch' '-c' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.cpp' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.o'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.compile.status'; exit "$status"; fi
: >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.started'
if { time '/opt/AdaptiveCpp/bin/acpp' '--acpp-targets=generic' '--acpp-pcuda' '--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1' '-fuse-ld=lld' '-Wl,-O0' '-shared' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.o' '-o' '/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.so'  >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.stdout' 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.stderr'; } 2>'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.time'; then echo 0 >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.status'; else status=$?; echo "$status" >'/home/aidar/Проекты/practice/build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/0-q21-plain/db-pcuda-zMVtjr/0/7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80.link.status'; exit "$status"; fi
) &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
if ((failed)); then exit 1; fi
