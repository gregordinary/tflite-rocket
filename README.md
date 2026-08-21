# tflite-rocket

## AI Disclosure

With the exception of prior work this may build on, tflite-rocket was developed by AI, primarily Claude Code (Opus 4.8). Human involvement was mostly limited to setting project goals and providing hardware access. This is a side project for curiosity's sake and comes with no guarantee of quality, accuracy, or update frequency.

## About tflite-rocket

A TensorFlow Lite external delegate for Rockchip NPUs (validated on the RK3588) via the mainline
`rocket` DRM-accel driver — the detection frontend of the stack. It offloads supported TFLite ops to
the NPU through the standalone `rocket-userspace` driver library; everything else falls back to the
CPU, exactly like ggml's scheduler. Any TFLite-based application can load it at runtime and run an
unmodified `.tflite` model; the motivating use case is real-time detection (for example, Frigate) on
mainline RK3588, *without* the proprietary RKNN toolkit.

It links the standalone `librocketnpu` driver library (the `rocket-userspace` project) and plugs into
the TensorFlow Lite runtime.

```
.
  rocket-userspace/   # driver lib + regcmd op generators (dependency): git clone https://github.com/gregordinary/rocket-userspace
  tflite-rocket/      # this project: the TFLite delegate (detection)
```

Real `.tflite` detectors run on the NPU through this delegate, including native int8 and native uint8
convolution. The delegate (`rocket_delegate.cpp`, a classic C `TfLiteDelegate`) partitions the graph
and runs general `CONV_2D`, `DEPTHWISE_CONV_2D`, and the surrounding `ADD`/`POOL`/`CONCAT`/`RESHAPE`
seams — float and int8/uint8 — so a whole MobileNet/SSD partition is NPU-resident. The 1×1 pointwise
fast path uses the resident prepacked matmul; the general / strided / stem / depthwise convs use the
HW-validated `rocket_conv2d_fp16`. int8/uint8 has two paths: the default dequant↔fp16 boundary (an
fp16 approximation) and the opt-in `native_int8=1` path — a real int8×int8→int32 conv with host
requant and exact int8/uint8 semantics. The complete op mapping, the quantization internals, the
delegate-option reference, and the host-side/gaps analysis are in [API.md](API.md).

## Architecture

A TFLite external delegate.

- TFLite loads an external delegate `.so` at runtime. The `.so` exports two C symbols —
  `tflite_plugin_create_delegate` / `tflite_plugin_destroy_delegate` — and implements
  `tflite::SimpleDelegateInterface`. The runtime auto-partitions the graph: ops our
  `IsNodeSupportedByDelegate` accepts run on the NPU; the rest stay on CPU.
- Another external delegate (Teflon) already targets this same driver; tflite-rocket is a separate
  implementation focused on op coverage, perf, the fp16 path, and reuse of the `rocket-userspace`
  driver library. Both are interchangeable `.so`s from the host application's point of view.

The delegate partitions and runs general `CONV_2D` (KxK / stride / `SAME`|`VALID` pad / dilation),
`DEPTHWISE_CONV_2D` (depth_multiplier 1), and the surrounding host seams. The only genuinely new glue
(NHWC↔NCHW transpose, pad materialization, weight reorder, bias + fused activation, the int8/uint8
requant) is validated off-hardware by `tests/convert_test.cpp` against independent oracles and on the
NPU; `convert_test` links only the driver, so the data path gates on the NPU without TFLite. The full
op-by-op table is in [API.md](API.md#tflite-op-mapping).

### Using it from an application

Any TFLite-based application loads the delegate at runtime — in Python via
`tflite.load_delegate(".../libtflite_rocket.so")`, or through the equivalent C external-delegate API
— and runs an unmodified `.tflite` model. Nothing in the delegate is detector- or
application-specific.

Some applications need a thin adapter. Frigate, for example, compiles its detectors as Python plugins
(it has no external-detector loader and ships an `rknn` detector but no `rocket` one), so shipping
there is two pieces: the delegate `.so` (this project) plus a small `rocket.py` detector plugin —
modeled on Frigate's own `cpu_tfl.py` — that loads a model with the delegate via
`tflite.load_delegate`. That plugin is the only application-specific piece; the delegate itself is
generic. A complete Frigate deployment (the plugin, a Docker image, compose, and config) lives in
[frigate/](frigate/), validated running SSDLite-MobileDet on the NPU inside Frigate.

### The RK3576: a delegated partition is a planned GRAPH

The RK3576 carries the same NPU IP with a different register encoding, and on it the delegate takes a
second path: a delegated partition is lowered, planned and run as one network rather than as a
sequence of independent ops. That is not a tuning choice. On the same MobileNetV1-224 the op entries
with transient weights cost ~115 ms, resident weights ~21 ms, keeping the tensors in the part's own
cube layout between layers 10.4 ms, and submitting the whole run as ONE hardware kick 5.0 ms — and
all three of those levers are properties of the graph, so none can be expressed one op at a time.

The path is selected from the detected part, not from an option (`rk3576=0` is the A/B arm). It
claims `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `ADD`, `CONCATENATION`,
`QUANTIZE` and a foldable `PAD`, at int8 or uint8; everything else stays on the CPU. Inside the
partition every tensor is CHW int8, so the only transposes a fully linked graph pays are the two at
the partition boundary. Five ImageNet classifiers and two COCO detectors run end to end, warm, on an
RK3576 (H96 MAX M9, kernel 7.1.6, 786 MHz), each against the same interpreter with no delegate:

| model | delegate | CPU | | claimed | joins | kicks |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV1-224 | 5.2 ms | 71.7 ms | 13.9× | 29/31 | 28/28 | 1 |
| MobileNetV2-224 | 6.8 ms | 45.3 ms | 6.6× | 64/66 | 63/63 | 1 |
| ResNet-18-224 | 8.8 ms | 167.1 ms | 19.0× | 36/38 | 27/27 | 1 |
| Inception V1-224 | 7.4 ms | 180.5 ms | 24.4× | 81/83 | 53/53 | 1 |
| Inception V3-299 | 34.5 ms | 669.3 ms | 19.4× | 130/132 | 90/94 | 16 |
| SSD-MobileNetV2-COCO | 17.0 ms | 109.0 ms | 6.4× | 87/111 | 70/82 | 6 |
| EfficientDet-Lite0-320 | 98.9 ms | 121.9 ms | 1.2× | 238/267 | 85/89 | 10 |

Four of the five classifiers go out as a SINGLE hardware kick covering the whole partition. All five
return the interpreter's own top-1 label; they differ from it on 0.2-1.3% of the output elements,
which is this DPU's requant rather than a defect — it carries a 15-bit multiplier and rounds ties to
even where TFLite carries 31 bits and rounds half away from zero, and the disagreement compounds down
a chain because layer *n*+1 is fed the part's output rather than TFLite's. The two detectors run but
are **not** validated: a detection list is NMS-ordered, so one count of drift reorders it, and
scoring one is a mAP question (`tools/coco_map.py`).

The placement and linking rules — which tensors stay in cube layout, which producers write slices of
one shared buffer, which runs of layers are one submit — are the driver library's `rocketgraph`
component, not this project's, so a second frontend cannot fork them. What lives here is the TFLite
lowering: the claim, the geometry (including TFLite's asymmetric `SAME` as an output extent, since
the hardware derives the pad its last window consumes rather than taking a trailing one), the uint8
→ int8 rebase, the weight transposes, the buffers and the boundary layout. See
[rocket_rk3576_net.h](rocket_rk3576_net.h) and `tools/rk3576_net_ab.py`.

## Performance

Warm, RK3588 @ 600 MHz, `native_int8=1` (the exact-int8 path — opt-in for the delegate, and the
default in the Frigate `rocket.py` plugin). At a glance:

| Metric | Result |
|---|---|
| COCO-val mAP@[.5:.95] | 0.3321 NPU vs 0.3318 CPU int8 (parity, 500 images) |
| Warm single-stream latency | ~336 ms (host cube-gather-bound, not submit-bound) |
| Multi-camera pool, P=1→4 | 3.20 → 9.55 detection_fps (2.98× at P=4, live Frigate) |

A single inference is host cube-scatter/gather-bound, so the NPU's value is throughput under a
multi-camera pool (Frigate's regime), not single-stream latency. The levers behind the ~336 ms
single-stream figure (on MobileDet / MobileNetV2):

- **Resident device state.** Packing weights once in `Prepare` and reusing a resident 5-BO conv pool
  (`rocket_conv_ctx`) across calls/tiles removes the dominant per-op cost. Warm MobileNetV2 block
  ~28→12 ms (~3×), numerics bit-identical.
- **Multicore DIRECT conv** (`rocket_conv_pool` + `rocket_conv2d_int8_mt`) fans the conv's independent
  OC/OH/OW tiles across the 3 NPU cores, bit-exact. Warm MobileDet 560→458 ms (1.21×).
- **1×1 int8/uint8 → resident matmul** (`mm_int8`, default-on under `native_int8`): a 1×1 conv is a
  matmul, so routing it to `rocket_matmul_int8_prepacked` is +8% warm (366→336 ms), bit-exact.
- **COCO-val mAP: CPU parity.** MobileDet mAP@[.5:.95] = 0.3321 vs CPU 0.3318 (Δ +0.0002, 500 val
  images, `tools/coco_map.py`). The fp16-approx depthwise and the missing per-channel DW int8 cost
  ~0 mAP.

Throughput comes from running several detection contexts concurrently — one process per stream
(exactly how Frigate runs cameras), each pinned to a distinct A76 core via `ROCKET_CPU_AFFINITY`. The
host-requant NEON vectorization, the multi-camera throughput recipe, and the op-coverage gaps are
detailed in [API.md](API.md#host-side-work-and-remaining-gaps).

## Build

**Prerequisites.** An RK3588 board on a mainline kernel carrying the `rocket` DRM-accel driver, with
`/dev/accel/accel0` present (`lsmod | grep rocket`). Build and install the sibling `rocket-userspace`
driver first — `cd ../rocket-userspace && cmake -S . -B build && cmake --build build && sudo cmake
--install build` — since the delegate links it (see the reinstall note below). Bring your own
`.tflite` detector (e.g. `ssdlite_mobiledet_coco_qat_postprocess.tflite` from the Coral / MediaPipe
model zoo). The NPU boots at 200 MHz and the delegate is correct there, but the figures above are at
600 MHz — apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`.

The delegate is a classic C `TfLiteDelegate` (it does not use the C++ `SimpleDelegate` helper), so it
builds against the TFLite C-API headers alone. The
handful of TFLite symbols it references (`TfLiteIntArrayCreate`/`Free`) bind at `dlopen` from the host
interpreter, exactly like Mesa's `libteflon.so`. Point `-DTFLITE_DIR` at any tree carrying
`tensorflow/lite/core/c/{common,builtin_op_data}.h` + `tensorflow/lite/builtin_ops.h` — the set
Mesa's Teflon build ships at `<mesa>/include` works as-is, so no separate TFLite build is needed.
Without one to hand, that whole set is four files and their transitive closure is themselves:

```sh
D=/tmp/tflhdr; B=https://raw.githubusercontent.com/tensorflow/tensorflow/v2.18.0
mkdir -p $D/tensorflow/lite/core/c
for f in tensorflow/lite/builtin_ops.h tensorflow/lite/core/c/common.h \
         tensorflow/lite/core/c/builtin_op_data.h tensorflow/lite/core/c/c_api_types.h; do
  curl -fsSL -o $D/$f $B/$f
done
```

```sh
cmake -S . -B build -DTFLITE_DIR=/path/to/mesa/include
cmake --build build -j           # -> libtflite_rocket.so (no TFLITE_LIB required)
# validate: load a .tflite with tflite.load_delegate("build/libtflite_rocket.so") and
# compare vs no-delegate (tools/run_delegate.py --compare; tools/rk3576_net_ab.py scores a
# classifier's label and reports its wall against the same interpreter with no delegate).
```

A host interpreter is the other half. `ai-edge-litert` is the maintained wheel and is the one these
tools are driven with; it hides `TfLiteIntArrayCreate`/`Free`, so `LD_PRELOAD` the
`libtflite_cshim.so` this build also produces before the interpreter, or the delegate will not
resolve them at `dlopen`.

(Pass `-DTFLITE_LIB=/path/to/libtensorflowlite.so` only if you have a full C++ TFLite build and
prefer link-time symbol resolution.)

Pass `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON` to run the `.so` on an older-glibc target than the build host
(e.g. a Debian-bookworm container from a trixie/forky host): it internalizes the glibc-2.38
`__isoc23_strtol`/`__isoc23_fscanf` symbols a newer GCC emits, dropping the `.so`'s glibc floor to
2.34. Off by default (normal builds are byte-unchanged). See `glibc_compat.c` and
[frigate/README.md](frigate/README.md), which uses it to deploy into the Frigate image.

> **If the driver is installed** (`find_package(rocketnpu)` resolves to a `cmake --install`d package,
> e.g. `/usr/local`), the delegate links that installed header/lib, not the source tree. So after any
> change in `rocket-userspace/`, reinstall the driver before rebuilding the delegate:
> `cd ../rocket-userspace && cmake --build build && sudo cmake --install build`, then `cmake --build
> build` here. (`convert_test` uses the source headers directly, so it can pass while the delegate
> `.so` still links a stale installed lib — don't be fooled.)

**Gate the glue without TFLite.** `convert_test` links only the driver and runs the delegate's exact
layout/pad/bias/activation path against an independent NHWC oracle — on the CPU oracle off-device
(x86), or on the NPU when one is present (the hardware gate):

```sh
cmake -S . -B build                          # TFLITE_DIR/LIB optional; .so is skipped
cmake --build build -j -t convert_test
./build/convert_test                         # conv glue + aux ops, all max_abs=0 / max|dq|=0
```

`convert_test` covers the conv glue (11 float + 8 quant `CONV_2D` shapes, plus 7 float + 4 int8/uint8
`DEPTHWISE_CONV_2D` shapes) and the aux host ops (`ADD` / `AVERAGE`+`MAX_POOL_2D` / `CONCATENATION` /
`RESHAPE`, float + int8/uint8) against independent NHWC oracles; the conv shapes also run on the NPU
when one is present (the HW gate), while the aux ops are pure host kernels (their proof is
device-independent).

The full external-delegate option reference (`native_int8` / `mm_int8` / `nthreads` / `aux_ops` / the
`*_npu` routes / `nchw_resident` / …) and the `ROCKET_PROF_POOL` probe are in
[delegate options](API.md#delegate-options).

## The rocket NPU stack

This delegate is one frontend of an open source stack for Rockchip NPUs — three userspace projects
plus a set of optional kernel patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`) — the userspace driver, matmul, and on-NPU op library. The
  dependency; usable on its own.
- **`tflite-rocket`** (this project) — a TFLite external delegate for detection models.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)** — a ggml backend `.so` for `llama.cpp` / `whisper.cpp`, linking the same driver.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope) — optional out-of-tree kernel-module patches
  (clock / voltage / IOMMU). They raise the NPU clock from its 200 MHz boot default to 600 MHz; the
  performance figures here assume them.

## License & credits

`tflite-rocket` is GPL-3.0-or-later (it links the GPL-3 `rocket-userspace` driver library). It targets
[TensorFlow Lite](https://www.tensorflow.org/lite) (Apache-2.0) as an external delegate and reuses the
`rocket-userspace` driver for the NPU op path.
