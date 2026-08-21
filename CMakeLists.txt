# tflite-rocket — TFLite external delegate for the RK3588 NPU (rocket).
# Links the standalone rocket-userspace driver library.
#
# The delegate is a classic C TfLiteDelegate, so it builds against the TFLite C-API
# HEADERS ALONE — no C++ TFLite library, no flatbuffers. Point -DTFLITE_DIR at any tree
# carrying tensorflow/lite/core/c/{common,builtin_op_data}.h + tensorflow/lite/builtin_ops.h
# (the set Mesa's Teflon build ships at <mesa>/include works as-is):
#   -DTFLITE_DIR=/path/to/mesa/include   # TFLite C-API header root
#   [-DTFLITE_LIB=/path/to/libtensorflowlite.so]          # OPTIONAL link-time resolution
#
#   cmake -S . -B build -DTFLITE_DIR=/path/to/mesa/include
#   cmake --build build -j           # -> libtflite_rocket.so (TfLiteIntArray* bind at dlopen)
#
# WITHOUT TFLite you can still build + run `convert_test`: it links only the
# rocket-userspace driver and validates the delegate's NHWC<->NCHW + SAME/VALID pad +
# bias/activation glue (rocket_convert.h) against an independent NHWC oracle. With
# an NPU present it runs that exact glue ON HARDWARE, so the delegate's data path
# is gate-able on RK3588 hardware before the TFLite build env is solved:
#   cmake -S . -B build && cmake --build build -j -t convert_test && ./build/convert_test
cmake_minimum_required(VERSION 3.16)
project(tflite-rocket CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- driver lib (shared foundation) ---
# Prefer an installed rocketnpu CMake package (cmake --install of rocket-userspace, which
# exports rocketnpu::rocketnpu); otherwise build it from a sibling checkout. Override
# the source location with -DROCKETNPU_DIR=/path/to/rocket-userspace.
find_package(rocketnpu QUIET CONFIG)
if(rocketnpu_FOUND)
    message(STATUS "tflite-rocket: using installed rocketnpu package")
else()
    if(NOT DEFINED ROCKETNPU_DIR)
        set(ROCKETNPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rocket-userspace CACHE PATH "Path to a rocket-userspace checkout")
    endif()
    if(NOT EXISTS ${ROCKETNPU_DIR}/CMakeLists.txt)
        message(FATAL_ERROR
            "rocketnpu package not found and no checkout at ROCKETNPU_DIR=${ROCKETNPU_DIR}. "
            "Install rocket-userspace (cmake --install) or set -DROCKETNPU_DIR=/path/to/rocket-userspace.")
    endif()
    message(STATUS "tflite-rocket: building rocketnpu from ${ROCKETNPU_DIR}")
    set(ROCKETNPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${ROCKETNPU_DIR} ${CMAKE_BINARY_DIR}/rocketnpu)
endif()

# --- the delegate .so (needs only the TFLite C-API HEADERS; -DTFLITE_DIR=<header root>) ---
# rocket_delegate.cpp is a classic C TfLiteDelegate (it does NOT use the C++ SimpleDelegate
# helper), so it builds against the TFLite C-API headers alone — e.g. the set Mesa's Teflon
# build ships at <mesa>/include. No libtensorflowlite.so is required: the only TFLite symbols
# the .so references (TfLiteIntArrayCreate/Free) bind at dlopen from the host interpreter,
# exactly like Mesa's libteflon.so. So the headers-only invocation is just:
#   cmake -S . -B build -DTFLITE_DIR=/path/to/mesa/include
#   cmake --build build -j            # -> libtflite_rocket.so
# Pass -DTFLITE_LIB=<libtensorflowlite.so> only if you have one and prefer link-time resolution.
if(DEFINED TFLITE_DIR)
    add_library(tflite_rocket SHARED rocket_delegate.cpp)
    set_target_properties(tflite_rocket PROPERTIES PREFIX "lib" OUTPUT_NAME tflite_rocket)
    target_include_directories(tflite_rocket PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}                     # rocket_convert.h
        ${TFLITE_DIR})                                  # tensorflow/lite/... C-API headers
    # tflite_plugin_create_delegate keeps default visibility (no -fvisibility=hidden) so
    # the external-delegate loader can dlsym it.
    # rocketgraph is the RK3576 GRAPH PLANNER — the placement and linking rules that decide
    # which tensors stay in the part's cube layout, which producers write slices of one
    # buffer and which runs of layers go out as one hardware kick. It ships with the
    # rocketnpu package rather than living here because a second copy of those rules would
    # fork silently; see rocket_rk3576_net.h.
    target_link_libraries(tflite_rocket PRIVATE rocketnpu::rocketgraph rocketnpu::rocketnpu)
    target_compile_options(tflite_rocket PRIVATE -Wall -Wextra)

    # Portability: internalize the glibc 2.38 __isoc23_strtol/fscanf symbols so the .so
    # loads on an older-glibc runtime (e.g. a Debian-bookworm Frigate container) even
    # when built on a newer host. Off by default (normal builds are byte-unchanged); see
    # glibc_compat.c. The shim must be C11 so its own strtol/fscanf aren't redirected.
    option(TFLITE_ROCKET_PORTABLE_GLIBC
           "Link a shim so the .so loads on glibc < 2.38 (e.g. the Frigate image)" OFF)
    if(TFLITE_ROCKET_PORTABLE_GLIBC)
        target_sources(tflite_rocket PRIVATE glibc_compat.c)
        set_source_files_properties(glibc_compat.c PROPERTIES COMPILE_OPTIONS "-std=gnu11")
        message(STATUS "tflite-rocket: PORTABLE_GLIBC on -> .so floors at glibc 2.34")
    endif()

    if(DEFINED TFLITE_LIB)
        target_link_libraries(tflite_rocket PRIVATE ${TFLITE_LIB})
        message(STATUS "tflite-rocket: building libtflite_rocket.so (TFLITE_DIR=${TFLITE_DIR}, linking ${TFLITE_LIB})")
    else()
        message(STATUS "tflite-rocket: building libtflite_rocket.so headers-only "
                       "(TFLITE_DIR=${TFLITE_DIR}; C-API symbols bind at dlopen, no TFLITE_LIB)")
    endif()

    # Test-harness shim: provides the two TFLite C-API symbols (TfLiteIntArrayCreate/Free) for host
    # interpreters that hide them (LiteRT / ai_edge_litert). LD_PRELOAD it before the interpreter so
    # the headers-only delegate resolves them at dlopen — see tools/tflite_c_shim.c, tools/run_delegate.py.
    add_library(tflite_cshim SHARED tools/tflite_c_shim.c)
    set_target_properties(tflite_cshim PROPERTIES PREFIX "lib" OUTPUT_NAME tflite_cshim)
    target_include_directories(tflite_cshim PRIVATE ${TFLITE_DIR})
else()
    message(STATUS "tflite-rocket: TFLITE_DIR unset -> skipping libtflite_rocket.so "
                   "(build/run `convert_test` to gate the glue without TFLite).")
endif()

# --- glue test (driver-only; no TFLite) ---
# convert_test exercises the delegate's exact conv layout/pad/bias/activation path
# (rocket_convert.h) plus the aux host ops (rocket_ops.h: ADD/POOL/CONCAT/RESHAPE) and
# compares to independent NHWC oracles. For conv, fd<0 => CPU oracle (x86), fd>=0 =>
# the glue runs on the NPU (RK3588 HW gate); the aux ops are host kernels (no NPU).
option(ROCKET_TFLITE_BUILD_TESTS "Build the tflite-rocket glue test" ON)
if(ROCKET_TFLITE_BUILD_TESTS)
    enable_testing()
    add_executable(convert_test tests/convert_test.cpp)
    target_include_directories(convert_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(convert_test PRIVATE rocketnpu::rocketnpu)
    target_compile_options(convert_test PRIVATE -Wall -Wextra)
    add_test(NAME convert_test COMMAND convert_test)   # CPU oracle on x86, HW glue on RK3588 hardware

    # Host-only box-sum gate: forks a child per ROCKET_BOXSUM_MODE, cross-checks the three
    # NEON realizations are byte-identical, and microbenches them. No NPU needed (pure host
    # code in rocket_convert.h), so it always runs — exit 1 on any cross-mode divergence.
    add_executable(boxsum_bench tests/boxsum_bench.c)
    target_include_directories(boxsum_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(boxsum_bench PRIVATE rocketnpu::rocketnpu)
    target_compile_options(boxsum_bench PRIVATE -Wall -Wextra)
    add_test(NAME boxsum_bench COMMAND boxsum_bench)

    # Driver-only HW correctness/regression gates. They need an NPU at runtime; absent
    # one they print "no NPU" and exit 2, so 2 is the CTest skip code. Building them by
    # default keeps them from bit-rotting (the int8 resonance / conv-bank-slack /
    # matmul-vs-conv gates), rather than relying on a hand-built one-off.
    foreach(t conv_bank_slack conv_bisect fp16_res mm_nt_det mm_vs_conv_acc)
        add_executable(${t} tests/${t}.c)
        target_link_libraries(${t} PRIVATE rocketnpu::rocketnpu)
        target_compile_options(${t} PRIVATE -Wall -Wextra)
        add_test(NAME ${t} COMMAND ${t})
        set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 2)
    endforeach()
endif()
