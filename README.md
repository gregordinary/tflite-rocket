# tflite-rocket

## AI disclosure

Except for the prior work it builds on, tflite-rocket was developed by AI, primarily Claude. Human
involvement was mostly limited to setting project goals and providing hardware access. This is a
side project for curiosity's sake, and it comes with no guarantee of quality, accuracy, or update
frequency.

## About tflite-rocket

tflite-rocket is a TensorFlow Lite external delegate for Rockchip NPUs, validated on the RK3588 and
the RK3576. It runs the TFLite ops it [claims](#terms) on the NPU through the mainline `rocket`
DRM-accel driver. It uses the `librocketnpu` driver library, which the `rocket-userspace` project
builds. The interpreter's CPU kernels run every other op.

Any TFLite application can load the delegate at runtime and run an unmodified `.tflite` model.
The motivating use is real-time detection, for example in Frigate, on a mainline RK3588 or RK3576
without the proprietary RKNN toolkit.

The build expects the checkouts side by side:

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  tflite-rocket/      # this project: the TFLite delegate (detection)
```

[API.md](API.md) is the reference. It holds:

- The op-by-op mapping and the [route](#terms) each op takes
- The quantized convolution routes and their agreement with TFLite
- The RK3576 lowering
- Every delegate option and environment variable
- The multi-camera throughput recipe and the host-side work

A complete Frigate deployment is in [frigate/](frigate/).

## What runs on the NPU

On the RK3588 the delegate runs one op at a time:

- **`CONV_2D` and `DEPTHWISE_CONV_2D`**, float, int8 and uint8, with a depth multiplier of 1.
  A float 1×1 at stride 1 with aligned channels runs on the resident fp16 matmul. Every other
  convolution runs on the fp16 convolution, and a quantized one is dequantized at its boundary
  by default.
- **The native int8 routes**, under `native_int8=1`. The convolution accumulates int8 exactly on
  the NPU. The int8-out routes, on by default under `native_int8`, also requantize on the NPU.
  They differ from TFLite by one count on 0.037-0.156% of the outputs they write, never by more
  [HW sweep]. The int32 route requantizes on the host, exactly as TFLite does.
- **The ops between the convolutions**: `ADD`, the pools, `CONCATENATION`, `RESHAPE`, the
  activations, the elementwise and layout ops, resize, softmax and more. The delegate claims them
  and computes them with host kernels inside its own kernel, so a MobileNet or SSD backbone is one
  partition. Opt-in options move some of them to the NPU.
- **`FULLY_CONNECTED`**, float: a host matmul by default, and the NPU under `fc_npu=1`.

On the RK3576 the delegate runs each delegated partition as one planned network, the
[RK3576 graph](#the-rk3576), at int8 and uint8.

The delegate is a classic C `TfLiteDelegate` that exports `tflite_plugin_create_delegate` and
`tflite_plugin_destroy_delegate`. Its Prepare callback decides which nodes it claims and hands
them to TFLite (`ReplaceNodeSubsetsWithDelegateKernels`), which groups them into partitions.
Nothing in it is specific to one detector or application. Mesa's Teflon is another external
delegate for the same driver, and the two are interchangeable `.so` files to a host.

Some applications need a thin adapter. Frigate compiles its detectors as Python plugins and has
no external-detector loader. It includes an `rknn` detector and no `rocket` one. The plugin in
[frigate/](frigate/), `rocket.py`, is modeled on Frigate's `cpu_tfl.py` and loads the model with
this delegate.

## Terms

The terms this README, [API.md](API.md) and [frigate/README.md](frigate/README.md) use that
TFLite's documentation does not define:

| Term | Definition | Source | Nearest standard term |
|---|---|---|---|
| Claim | The delegate's acceptance of a TFLite node, made in its Prepare callback. TFLite then runs a claimed node inside the delegate's kernel. Its opposite is decline. | Coined | Supported node |
| CPU parity | A detector's COCO mAP on the delegate matching the CPU interpreter's within the difference stated beside it. | Coined | Accuracy parity |
| Cube layout | The NPU's tiled tensor layout, in which channels are grouped and packed innermost. | Coined | NC1HWC2 |
| Decline | The delegate's refusal of a TFLite node. The interpreter's CPU kernels then run it. Its opposite is claim. | Coined | Unsupported node |
| fp16 route | The route that dequantizes a quantized convolution's input to fp16, convolves in fp16 on the NPU and requantizes the output on the host. | Coined | |
| Int32 route | A native int8 route that writes the int32 accumulator and requantizes it on the host. | Coined | |
| Int8-out route | A native int8 route that requantizes on the NPU and writes int8 output. | Coined | On-chip requantization |
| Kick | One start of the NPU on a submitted job. | Coined | Dispatch |
| Native int8 routes | The routes `native_int8=1` enables, which accumulate int8 products exactly on the NPU rather than in fp16. | Coined | Integer convolution |
| One count | One step of a quantized output's integer value. | Coined | One LSB |
| Resident weights | Weights packed into NPU buffers once, at Prepare, and reused by every invoke. | Coined | Prepacked weights |
| Route | The way the delegate computes one claimed op, as `profile=1` names it: for example `fp16`, `native-int8` or `native-uint8-i8out`. | Coined | Code path |
| The RK3576 graph | The RK3576's execution of a delegated partition, lowered, planned and run as one network rather than op by op. | Coined | Graph compilation |

## Performance

On the RK3588, warm, under `native_int8=1` with its default routes:

| Model | Latency, `performance` | Latency, `ondemand` | COCO mAP@[.5:.95] | CPU interpreter's mAP |
|---|---:|---:|---:|---:|
| SSDLite-MobileDet | 51.4 ms | 73.5 ms | 0.3325 | 0.3301 |
| EfficientDet-Lite0 | 94.5 ms | 127.8 ms | 0.2810 | 0.2823 |
| SSD MobileNet v2 | 47.1 ms | 64.3 ms | 0.2634 | 0.2631 |

**Conditions:** RK3588 (Turing RK1), the mainline `rocket` driver, NPU at 928 MHz (the 900 MHz
request). The `performance` column has the CPU governor pinned to `performance` and the process
on the Cortex-A76 cores. The `ondemand` column has the default governor, with
`scaling_min_freq` at 1.2 GHz on the A76 cores, and the process unpinned. Each latency is the
median of 30 warm invokes on one image. mAP is over 500 COCO val2017 images [HW sweep 2026-10-29].

A detector's accuracy is stated as mAP against the CPU interpreter. Its output list is
NMS-ordered, so one count of arithmetic drift reorders or drops a box. An element-wise comparison
against the CPU therefore has no meaning.

A deployment with several cameras needs throughput more than single-stream latency. Run one
process per stream, each pinned to its own A76 core with `ROCKET_CPU_AFFINITY`, as Frigate runs
one process per detector. Four such processes run SSDLite-MobileDet at 3.33-3.51x one process's
rate, with every process's output identical to the single process's [HW sweep 2026-10-29]. In
live Frigate, four detectors gave 2.98x one detector's `detection_fps` on a 2026-07-01 build at
~336 ms an inference. [Multi-camera throughput](API.md#multi-camera-throughput) has the recipe and
the figures.

> [!CAUTION]
> A load-sampling CPU governor slows the delegate. The process blocks while the NPU runs, so
> `ondemand` reads the cores as idle and lowers them toward `scaling_min_freq`. The host half of
> the next inference then runs there. MobileDet is 1.43x slower at a `scaling_min_freq` of
> 1.2 GHz.
>
> On a BSP-kernel board whose `scaling_min_freq` is 408 MHz, a 2026-08-25 build measured
> 199.7 ms on `performance` and 639.4 ms on `ondemand`, 3.2x. The CPU interpreter was 183 ms
> under both. For a deployment, raise `scaling_min_freq` on the cores the detector runs on. For a
> benchmark, pin the governor before quoting any figure.

The changes behind the single-stream figure, and what each measured, are in
[Host-side work](API.md#host-side-work).

### Evidence

Each latency is a median over four passes, one process per configuration, with the governor order
and the model order rotated. Memory was reset before every process. Per pass, `ondemand` measured
1.29-1.50x `performance` on SSDLite-MobileDet, 1.34-1.40x on SSD MobileNet v2 and 1.34-1.37x on
EfficientDet-Lite0. The pool figure is two passes of `tools/pool_hash.py`, one process per A76 core
at `nthreads=1`, on `performance`. It measured 15.3-15.9 inferences a second at one process and
52.9-53.6 at four.

On 2026-09-27 and 2026-09-28, at ~816 MHz (the 600 MHz request), SSDLite-MobileDet measured ~56 ms
on `performance` and ~76 ms on `ondemand`. EfficientDet-Lite0 measured 101 ms on `performance`.
Their mAP over 100 images was 0.4040 against the CPU interpreter's 0.3980 (SSDLite-MobileDet),
0.3023 against 0.2996 (EfficientDet-Lite0) and 0.3229 against 0.3215 (SSD MobileNet v2).

## Hardware support

| Part | Status | How the delegate runs |
|---|---|---|
| RK3588 | Supported | One op at a time, float, int8 and uint8 |
| RK3576 | Supported | [The RK3576 graph](#the-rk3576), int8 and uint8 |
| Other Rockchip parts | Not characterized | |

### The RK3576

The RK3576 integrates the same NPU design with a different register encoding. On it, the
delegate lowers, plans and runs each delegated partition as one network. The delegate selects
this from the detected part, and `rk3576=0` turns it off.

On the RK3576 the delegate claims `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`,
`MAX_POOL_2D`, `ADD`, `CONCATENATION`, `QUANTIZE` and a `PAD` that folds into a convolution, at
int8 or uint8. The CPU runs everything else. Inside a partition every tensor is CHW int8, so a
fully linked graph transposes only at the partition's two boundaries.

The gain comes from the graph, so an op-at-a-time execution cannot reach it. On the same
MobileNetV1-224:

| Configuration | Time |
|---|---:|
| Op entries, weights packed per call | ~115 ms |
| Resident weights | ~21 ms |
| Tensors kept in the part's own cube layout between layers | 10.4 ms |
| The whole run submitted as one hardware kick | 5.0 ms |

Five ImageNet classifiers and two COCO detectors run end to end. `Joins` counts the adjacent
layer pairs the plan links in cube layout, and `Kicks` the hardware kicks per inference:

| Model | Delegate | CPU | Speedup | Claimed | Joins | Kicks |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV1-224 | 5.2 ms | 71.7 ms | 13.8x | 29/31 | 28/28 | 1 |
| MobileNetV2-224 | 6.8 ms | 45.3 ms | 6.7x | 64/66 | 63/63 | 1 |
| ResNet-18-224 | 8.8 ms | 167.1 ms | 19.0x | 36/38 | 27/27 | 1 |
| Inception V1-224 | 7.4 ms | 180.5 ms | 24.4x | 81/83 | 53/53 | 1 |
| Inception V3-299 | 34.5 ms | 669.3 ms | 19.4x | 130/132 | 90/94 | 16 |
| SSD-MobileNetV2-COCO | 17.0 ms | 109.0 ms | 6.4x | 87/111 | 70/82 | 6 |
| EfficientDet-Lite0-320 | 55.5 ms | 123.1 ms | 2.2x | 238/267 | Not characterized | Not characterized |

**Conditions:** H96 MAX M9, kernel 7.1.6, NPU 786 MHz, warm, measured 2026-08-04. The CPU column
is the same interpreter with no delegate, and the delegate column includes the transposes at the
partition boundaries. The EfficientDet-Lite0 row was measured on 2026-09-27, on `rocket` 1.6.0
and kernel 7.2.3, with both CPU clusters on `performance`.

All five classifiers return the interpreter's own top-1 label, and four of them run as a single
hardware kick covering the whole partition. Their outputs differ from the interpreter's on
0.2-1.3% of elements. The RK3576's DPU requantizes with a 15-bit multiplier and rounds ties to
even, where TFLite uses 31 bits and rounds half away from zero. The difference compounds down a
chain, because each layer is fed the NPU's output rather than TFLite's.

Both detectors are at [CPU parity](#terms). mAP@[.5:.95] over 500 COCO val2017 images is 0.2617
against the CPU's 0.2631 for SSD-MobileNetV2 (-0.0014), and 0.2818 against 0.2823 for
EfficientDet-Lite0 (-0.0005), from `tools/coco_map.py`.

EfficientDet-Lite0 has per-axis filter scales. The library programs each output channel's scale
on chip. It refuses only a live channel whose scale is below 1/65534 of the largest in its task.
The delegate asks the library at claim time, where a refusal declines one node. The same refusal
in the node's own Prepare would fail the whole model. On EfficientDet-Lite0 nothing is refused,
and its 29 unclaimed nodes are op kinds the RK3576 graph does not claim.

The driver library's `rocketgraph` component holds the placement and linking rules. It decides
which tensors stay in cube layout, which producers write slices of one shared buffer and which
runs of layers are one submit. A second frontend therefore cannot fork those rules. The delegate does the TFLite lowering, in
[rocket_rk3576_net.h](rocket_rk3576_net.h).
[The RK3576 graph](API.md#the-rk3576-graph) has the detail.

> [!CAUTION]
> On the RK3576 the delegate needs the NPU device. With no device, Prepare fails and the model
> does not load, where the RK3588 routes fall back to a CPU computation.

## Requirements

- An RK3588 or RK3576 board on a mainline kernel with the `rocket` DRM-accel driver loaded and
  `/dev/accel/accel0` present (`lsmod | grep rocket`). On the RK3576 the driver must have that
  part's support (device-tree compatible `rockchip,rk3576-rknn-core`), so the node appears only
  once it binds.
- CMake 3.16 or later, and a compiler that accepts C++ `_Float16` for the target: GCC 13 or
  later on Arm64. GCC 12 accepts it only on x86.
- The TFLite C-API headers (four files, fetched in [Build](#build)).
- A host interpreter: `ai-edge-litert` (LiteRT), which these tools are driven with, or
  `tflite_runtime`.
- `rocket-userspace` built and installed, since the delegate links it:

```sh
cd ../rocket-userspace && cmake -S . -B build && cmake --build build && sudo cmake --install build
```

### Clock

The RK3588's NPU boots at 200 MHz, and the delegate computes correctly there. Above 200 MHz the
NPU clock follows the rail voltage, so a requested rate is not the rate the NPU runs at.

Apply the `patches/rocket` clock and voltage patches, and load the module with
`rocket_npu_clk_hz=900000000`. On the RK1's 0.80 V rail that request runs the NPU at 928 MHz. The
[Performance](#performance) figures were measured there. The other RK3588 figures in this README
were measured with the 600 MHz request, which runs at ~816 MHz on the same rail.

> [!CAUTION]
> With the NPU cores pinned on (`power/control=on`), an `rmmod` at the 900 MHz request
> hard-locks the board, and only a power cycle recovers it. Unload the module only while all
> three cores are runtime-suspended.

## Build and run

### Build

The delegate is a classic C `TfLiteDelegate` and does not use the C++ `SimpleDelegate` helper,
so it builds against the TFLite C-API headers alone. The TFLite symbols it references,
`TfLiteIntArrayCreate` and `TfLiteIntArrayFree`, bind at `dlopen` from the host interpreter, as
Mesa's `libteflon.so` does. Point `-DTFLITE_DIR` at any tree that has
`tensorflow/lite/core/c/{common,builtin_op_data}.h` and `tensorflow/lite/builtin_ops.h`. The set
Mesa's Teflon build installs at `<mesa>/include` works as it is. Otherwise fetch the four files,
which include nothing beyond each other:

```sh
D=/tmp/tflhdr; B=https://raw.githubusercontent.com/tensorflow/tensorflow/v2.18.0
mkdir -p $D/tensorflow/lite/core/c
for f in tensorflow/lite/builtin_ops.h tensorflow/lite/core/c/common.h \
         tensorflow/lite/core/c/builtin_op_data.h tensorflow/lite/core/c/c_api_types.h; do
  curl -fsSL -o $D/$f $B/$f
done
```

```sh
cmake -S . -B build -DTFLITE_DIR=/tmp/tflhdr
cmake --build build -j           # -> libtflite_rocket.so and libtflite_cshim.so
```

`-DTFLITE_LIB=/path/to/libtensorflowlite.so` is optional, for a full C++ TFLite build and
link-time symbol resolution.

The option `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON` builds a `.so` for a target with an older glibc
than the build host. An example is a Debian bookworm container built from a trixie or forky host. It
links in forwarders for the symbols a newer host versions above glibc 2.34. Those are glibc 2.38's
`__isoc23_*` and glibc 2.43's `sqrtf`. The `.so` then needs only glibc 2.34.

If the `.so` still needs a newer glibc, or a newer libstdc++ than bookworm's GCC 12 provides, the
build fails. The option is off by default, and a normal build is unchanged. The forwarders are in
`glibc_compat.c`. [frigate/README.md](frigate/README.md) uses it to deploy into the
Frigate image.

The driver library resolves through `find_package(rocketnpu)` when it is installed, and from a
sibling `rocket-userspace` checkout otherwise. `-DROCKETNPU_DIR=/path/to/rocket-userspace`
overrides both.

### Run from Python

```sh
LD_PRELOAD=$PWD/build/libtflite_cshim.so python3 - <<'EOF'
from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
d = load_delegate("build/libtflite_rocket.so", options={"native_int8": "1"})
it = Interpreter(model_path="ssdlite_mobiledet_coco_qat_postprocess.tflite",
                 experimental_delegates=[d],
                 experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors(); it.invoke()
EOF
```

Preload `libtflite_cshim.so` under a Python interpreter. It supplies the two C-API symbols the
delegate binds at `dlopen`. LiteRT does not export them, and `tflite_runtime` loads its extension
module `RTLD_LOCAL`, so the delegate cannot see that wheel's copies either.

> [!CAUTION]
> `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES` is required. It holds XNNPACK back, so the
> convolutions are offered to this delegate first. Under the default resolver XNNPACK claims them,
> and the model runs entirely on the CPU while its log still reports a delegate. The `profile=1`
> option prints each convolution's route, which shows where it ran.

`tools/run_delegate.py --compare` runs a model with and without the delegate and compares the
outputs, with delegate options passed as `--option native_int8=1`. `tools/rk3576_net_ab.py` scores a classifier's label and times it against the same
interpreter with no delegate. The process-wide [delegate counts](API.md#delegate-counts) say how
many ops reached an NPU route.

Bring your own `.tflite` detector, for example `ssdlite_mobiledet_coco_qat_postprocess.tflite`
from the Coral or MediaPipe model zoo.

### Host tests

`convert_test` links only the driver library, so it builds without TFLite. It runs the delegate's
layout, padding, bias, activation and requantization code against independent NHWC reference
implementations. On a host with no NPU the convolutions run on the library's CPU reference. Where
an NPU is present, they run on it.

```sh
cmake -S . -B build                          # TFLITE_DIR optional; the .so is skipped without it
cmake --build build -j -t convert_test
./build/convert_test
```

It covers:

- The convolution glue: 13 float and 9 quantized `CONV_2D` shapes, and 11 float and 7 int8 or
  uint8 `DEPTHWISE_CONV_2D` shapes
- The native direct convolutions: 11 int8 and 13 uint8 shapes
- The native 1×1 convolutions through the matmul, the native depthwise glue, the resident-context
  shapes and the NCHW helpers
- Every op `aux_ops` claims, and the `FULLY_CONNECTED` host matmul
- The byte-table, transposed-conversion and broadcast kernels

The convolution glue and the layout ops are exact (`max_abs=0.0000`, `max|dq|=0`, `max|byte|=0`).
`SOFTMAX` and `LOG_SOFTMAX` pass within 1e-5, float `L2_NORMALIZATION` within 1e-6, and on a
device the PPU average pool within 0.06. `RESIZE_*`, `SOFTMAX`, `LOG_SOFTMAX` and `CUMSUM` have
float cases only. `boxsum_bench` checks the three box-sum implementations against each other.

The delegate options, `nthreads`, `aux_ops`, the opt-in `*_npu` routes, `nchw_resident` and the
rest, are in [Delegate options](API.md#delegate-options).

### On a BSP kernel

The delegate also runs on the BSP kernel most Rockchip images ship, through
[`rknpu-submit`](https://github.com/gregordinary/rknpu-submit). `librocketnpu` makes every kernel
call through one submit seam, and `rknpu-submit` implements it against the Rockchip BSP
`rknpu` driver. Building the delegate against it takes three more cache entries and no source
change:

```sh
cmake -S . -B build -DTFLITE_DIR=/path/to/tflite-c-headers \
      -DROCKETNPU_DIR=/path/to/rocket-userspace \
      -DROCKETNPU_PROVIDER=external \
      -DROCKETNPU_PROVIDER_LIB=/path/to/build-rknpu/librknpu-submit.a
```

The two drivers give byte-identical detector outputs. On 2026-08-26 an MD5 over every output
tensor of SSDLite-MobileDet and EfficientDet-Lite0 matched between them [HW sweep]. It matched in
the CPU-only, `native_int8=1` and `native_int8=0` configurations, with every `*_npu` route enabled
and with `nchw_resident=1`. `convert_test` and the six driver-level tests under `tests/` passed
on both.

Warm single-inference latency was within 2%: MobileDet 202.1 ms against the mainline driver's
197.9, and EfficientDet-Lite0 299.6 against 293.6. The workload is host-bound, and the two boards'
CPUs differ by as much on their own. A four-process pool reached 3.51x one process's throughput,
against 3.61x on the mainline driver, with every process returning the single-process output
hash.

**Conditions:** two RK3588 (Turing RK1) boards, one per driver, both NPUs at the 600 MHz request,
the delegate as built on 2026-08-26.

**Not characterized:** the NPUs' physical clocks, which follow each kernel's rail voltage, so the
latency comparison is not clock-matched. Also the on-chip requantization routes that `native_int8`
now takes by default (`direct_i8out`, `direct_perc`, `pw_i8out_min_m`, `dw_perc` and the uint8
depthwise), which have not run on the BSP driver through the delegate.

## Troubleshooting

### The model runs on the CPU

Most often XNNPACK claimed the convolutions first. Load the interpreter with
`OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES`, as in [Run from Python](#run-from-python), and
run once with `profile=1` to see each convolution's route.

If the delegate prints `an on-NPU path was requested but no NPU device opened`, the process could
not open `/dev/accel/accel0`. On the RK3588 the delegate then computes on the CPU, correctly and
slowly. Check that the node exists and that the process can open it:

```sh
ls -l /dev/accel/accel0       # typically root:render, mode 0660
id -nG | tr ' ' '\n' | grep -x render
```

### `undefined symbol: TfLiteIntArrayCreate`

The host interpreter does not make the two C-API symbols visible to the delegate. Preload the shim
the build produces:

```sh
LD_PRELOAD=$PWD/build/libtflite_cshim.so python3 your_script.py
```

### `version 'GLIBC_2.38' not found` at load

The `.so` was built on a host with a newer glibc than the target. Rebuild it with
`-DTFLITE_ROCKET_PORTABLE_GLIBC=ON`, as in [Build](#build).

### The delegate runs an older `librocketnpu`

When `find_package(rocketnpu)` resolves to an installed package (a `cmake --install`, typically
under `/usr/local`), the delegate links that installed library rather than a sibling source tree.
`convert_test` links the same installed library. After a change in `rocket-userspace/`, reinstall
it before rebuilding the delegate, or name the tree with `-DROCKETNPU_DIR`, which overrides the
installed package:

```sh
cd ../rocket-userspace && cmake --build build && sudo cmake --install build
cd ../tflite-rocket && cmake --build build
```

### Slower than the figures above

Check the CPU governor and `scaling_min_freq` on the A76 cores, per the governor Caution under
[Performance](#performance). Discard the warm-up invokes: the NPU clock parks at idle, so the
first inferences after a pause run slower.

## The rocket NPU stack

This delegate is one frontend of an open source stack for Rockchip NPUs. The stack is a driver
library, three frontends, a submit provider for the BSP kernel, and a set of optional kernel
patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`): the
  userspace driver, matmul, and on-NPU op library. It is the dependency, and usable on its own.
- **`tflite-rocket`**: this delegate, a TFLite external delegate for detection models.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)**: a ggml backend `.so` for
  `llama.cpp` and `whisper.cpp`, linking the same library.
- **[`ort-rocket`](https://github.com/gregordinary/ort-rocket)**: an ONNX Runtime execution
  provider for transformer encoders and per-layer convolutions, linking the same library.
- **[`rknpu-submit`](https://github.com/gregordinary/rknpu-submit)**: a submit provider that runs
  `librocketnpu` on the `rknpu` BSP kernel driver instead of the mainline `rocket` driver.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional out-of-tree
  kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock above its 200 MHz
  boot default. The RK3588 figures here were measured with them: the Performance table at
  928 MHz (the 900 MHz request), and the rest at ~816 MHz (the 600 MHz request).

## License and credits

`tflite-rocket` is GPL-3.0-or-later (it links the GPL-3 `rocket-userspace` driver library). It
targets [TensorFlow Lite](https://www.tensorflow.org/lite) (Apache-2.0) as an external delegate
and uses the `rocket-userspace` driver for the NPU.
