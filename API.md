# tflite-rocket: API and tuning reference

The reference for tuning the delegate and checking what it does. It covers the recommended
configuration, the op-by-op mapping, the quantized convolution routes and the RK3576 lowering. It
also covers every option and environment variable, the delegate counts, multi-camera throughput
and the host-side work. The [README](README.md) covers what the delegate is, the build and a
first run, and its [Terms](README.md#terms) define the words this page uses.

## Recommended configuration

For a detector, as in Frigate:

- **`native_int8=1`** runs the quantized convolutions on the [native int8 routes](#native-int8-routes),
  and is the default in the Frigate plugin, `rocket.py`. Under its default routes, MobileDet's
  COCO mAP@[.5:.95] over 500 images is 0.3325 against the CPU interpreter's 0.3301
  [HW sweep 2026-10-29]. Under it, the default `mm_int8=1` sends each aligned 1×1 convolution to
  the resident int8 matmul. The exception is a 1×1 that an int8-out route takes at
  `pw_i8out_min_m` pixels or more.
- **`aux_ops=1`**, the default, claims the ops between the convolutions and computes them with
  host kernels, so the partitions stay contiguous. The opt-in on-NPU routes for those ops
  (`act_npu`, `pool_npu` and the rest) are for comparison and fusion, because the host kernels
  are exact and add no NPU round trip.
- **A process pool** gives throughput: one delegate process per camera, each pinned to its own
  Cortex-A76 core with `ROCKET_CPU_AFFINITY` ([Multi-camera throughput](#multi-camera-throughput)).
  `nthreads`, default 4, spreads one convolution across the three NPU cores. The pool across
  streams gives more.
- **A raised `scaling_min_freq`** on the cores the detector runs on. The inference blocks while
  the NPU runs, so a load-sampling governor reads the cores as idle and lowers them. The host
  half of the next inference then runs at `scaling_min_freq`. Raising it on those cores is
  enough, and `performance` everywhere is not needed. Pin the governor before quoting any
  NPU-against-CPU figure.

At a `scaling_min_freq` of 1.2 GHz, MobileDet runs 51.4 ms with the governor pinned and 73.5 ms on
`ondemand`, 1.43x, at 928 MHz (the 900 MHz request) [HW sweep 2026-10-29]. At the same minimum, a
2026-09-28 build at ~816 MHz measured 55.7 and 75.8 ms, 1.36x, and a 2026-08-25 build at ~199 ms
measured 1.27x. On a BSP-kernel board whose `scaling_min_freq` is 408 MHz, that build measured 199.7
ms pinned and 639.4 ms on `ondemand`, 3.2x. The CPU interpreter measured 183 ms under both.

**Not characterized:** the 408 MHz board on the current build.

## TFLite op mapping

The table is the RK3588's, where the delegate runs one op at a time. On an RK3576 the delegate
claims the [RK3576 graph's](#the-rk3576-graph) op set instead. Every op not listed is declined
and runs on the interpreter's CPU kernels.

| TFLite op | Route | Status |
|---|---|---|
| `CONV_2D`, float, 1×1, stride 1, dilation 1, `IC%32==0`, `OC%16==0`, `IH·IW%4==0` | The resident fp16 matmul, `rocket_matmul_fp16_prepacked`: weights packed once at Prepare, `nthreads` workers. A shape the pack refuses runs on the fp16 convolution | Supported (default) |
| `CONV_2D`, float, every other shape: K×K, stride, `SAME` or `VALID`, dilation | The fp16 convolution, `rocket_conv2d_fp16_ctx`, falling back to `rocket_conv2d_fp16` | Supported (default) |
| `DEPTHWISE_CONV_2D`, float, depth multiplier 1 | The fp16 convolution's depthwise mode (`depthwise=1`, channel group 32), bit-exact [HW sweep] | Supported (default) |
| `CONV_2D` and `DEPTHWISE_CONV_2D`, int8 or uint8 | The fp16 route: dequantized to fp16 at the boundary, the fp16 convolution, requantized on the host ([The fp16 route](#the-fp16-route)) | Supported (default) |
| `CONV_2D`, int8 or uint8, under `native_int8=1` | The native int8 routes. [Native int8 routes](#native-int8-routes) gives which convolution takes which | Supported (opt-in) |
| `DEPTHWISE_CONV_2D`, int8 or uint8, under `native_int8=1` | The native depthwise routes, `rocket_conv2d_dw_int8` and `rocket_conv2d_dw_int8_perc`, in [Native int8 routes](#native-int8-routes) | Supported (opt-in) |
| `FULLY_CONNECTED`, float, constant weight, `M%4==0` or `M==1` | The host matmul, `rocket_fc_f`, which is faster than the NPU for a one-shot layer. `fc_npu=1` sends an `M%4==0` layer to the resident fp16 matmul with K padded to 32 and N to 16, and `M==1` stays on the host. Claimed whatever `aux_ops` is | Supported (default) |
| `HARD_SWISH`, `LOGISTIC`, `TANH`, `ELU`, `LOG`, `RELU`, `RELU6`, `RELU_N1_TO_1`, `LEAKY_RELU`, `EXP`, `SQRT`, `RSQRT`, `ABS`, `NEG`, `SQUARE`, `FLOOR` | Host kernels `rocket_unary_f` and `rocket_unary_q`, the quantized form through a 256-entry table. The `RELU` family, `NEG`, `SQUARE`, `FLOOR` and `ABS` are exact float arithmetic, and `EXP`, `SQRT`, `RSQRT` and `LEAKY_RELU` are libm-exact. `act_npu=1` sends the curved kinds to the DPU LUT (below the table) | Supported (default) |
| `MAXIMUM`, `MINIMUM`, same shape or broadcast | Host kernels `rocket_binary_f` and `rocket_binary_q`. `ew_npu=1` sends a float same-shape pair to the DPU EW ALU, `rocket_ew_max_fp16` and `rocket_ew_min_fp16`, bit-identical to the host | Supported (default) |
| `ADD`, `MUL`, `SUB`, `DIV`, same shape or broadcast, fused activation | Host kernels (`ADD` in `rocket_ops.h`, the others `rocket_arith_f` and `rocket_arith_q`). `SUB` and `DIV` compute `in0 op in1` | Supported (default) |
| `PRELU`, per-channel slope | Host kernels `rocket_prelu_f` and `rocket_prelu_q`, with the slope dequantized to float. `act_npu=1` sends the float form to `rocket_prelu_fp16`, bit-identical | Supported (default) |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | Host kernels in `rocket_ops.h`. `pool_npu=1` sends a float pool to the PPU, an average pool only with `VALID` padding | Supported (default) |
| `MEAN`, `REDUCE_MAX`, `REDUCE_MIN`, axes 1 and 2 of a rank-4 tensor with batch 1 | Host kernels `rocket_reduce_spatial_f` and `rocket_reduce_spatial_q`. `pool_npu=1` sends the float form to `rocket_global_{avg,max,min}pool_fp16` | Supported (default) |
| `CONCATENATION`, `RESHAPE` | Host kernels in `rocket_ops.h` | Supported (default) |
| `RESIZE_NEAREST_NEIGHBOR`, `RESIZE_BILINEAR` | Host kernels `rocket_resize_nearest_f/_q` and `rocket_resize_bilinear_f/_q`, with TFLite's `align_corners` and `half_pixel_centers` coordinates. A quantized resize needs equal input and output scale and zero point. `resize_npu=1` sends a float nearest resize at an integer factor, with both flags false, to `rocket_upsample_nearest_fp16`, bit-identical | Supported (default) |
| `L2_NORMALIZATION`, last axis, no fused activation | Host kernels `rocket_l2norm_f` and `rocket_l2norm_q`. `norm_npu=1` sends the float form to `rocket_l2norm_fp16` | Supported (default) |
| `TRANSPOSE_CONV`, float | Host scatter kernel `rocket_transpose_conv_f`, with TFLite's pad mapping | Supported (default) |
| `TRANSPOSE_CONV`, int8 or uint8 | Declined | Not supported |
| `SOFTMAX`, `LOG_SOFTMAX`, last axis | Host kernels `rocket_softmax_f/_q`, with the `beta` logit scale, and `rocket_logsoftmax_f/_q`, with the row maximum subtracted | Supported (default) |
| `CUMSUM`, last axis | Host kernels `rocket_cumsum_f` and `rocket_cumsum_q` | Supported (default) |
| `TRANSPOSE`, `PAD`, `PADV2`, `SLICE`, `STRIDED_SLICE`, `SPLIT`, `SPLIT_V`, rank 8 or less | Host byte kernels (`rocket_transpose_bytes`, `rocket_slice_bytes`, `rocket_pad_bytes`, `rocket_strided_slice_bytes`). Values pass through and the quantization is unchanged | Supported (default) |

Every host-kernel row takes float, int8 and uint8 except where the row says otherwise, and needs
`aux_ops=1`. `PAD` and `PADV2` take int32 paddings, and `SPLIT` and `SPLIT_V` an int32 axis.
`STRIDED_SLICE` takes any nonzero stride, positive or negative, and all five masks with at most
one ellipsis. An empty slice is declined. TFLite's converter lowers the ellipsis and new-axis
masks into explicit begin and end values and a `RESHAPE`. Those two masks therefore arrive only
from other producers.

No layout op has an NPU route, because the NPU has no layout-conversion engine.

`act_npu=1` sends `ELU` to `rocket_elu_fp16` and `LEAKY_RELU` to `rocket_leaky_relu_fp16`. It
sends `HARD_SWISH`, `LOGISTIC`, `TANH`, `LOG`, `EXP`, `SQRT`, `RSQRT` and `ABS` to the DPU LUT,
`rocket_activation_fp16`, whose input domain is limited. A quantized input is dequantized, passed
through the LUT and requantized. `RELU`, `RELU6`, `RELU_N1_TO_1`, `NEG`, `SQUARE` and `FLOOR`
stay on the host. A `RELU` the converter fused into a `CONV_2D` is applied in the convolution.

A float MobileNet's or MobileDet's 1×1 convolutions, most of its multiply-accumulates, run on the
resident fp16 matmul. A quantized model's 1×1 convolutions take the fp16 route by default, and
the int8 matmul or an int8-out route under `native_int8=1`. With the pointwise, direct and
depthwise convolutions and the ops between them claimed, a MobileNet or SSD backbone is one
partition.

### The ops between the convolutions

A detector has more than convolutions. Residual `ADD` (MobileNetV2 and V3, FPN), the pools, the
`CONCATENATION` that joins an SSD head's box and class predictors, and the `RESHAPE` in that head
sit between them. Left unclaimed, each would end a partition.

The delegate claims them and computes them with host NHWC kernels. They are memory-bound
elementwise and reduction ops. A convolution on the fp16 route transposes NHWC to NCHW and back
itself, so claiming the ops around it does not save a transpose. The 1×1 matmul routes read and
write NHWC directly. The gain is fewer, larger partitions. [`nchw_resident`](#nchw_resident)
keeps some tensors in fp16 NCHW between convolutions and skips the transpose.

A quantized op between the convolutions uses per-tensor input and output scales and zero points
and requantizes to the output type. `ADD` keeps each input's own scale, `CONCATENATION`
requantizes each input to the output scale, and a pool dequantizes, reduces and requantizes.

A broadcasting two-tensor op, per-channel `[C]`, scalar or general right-aligned, has each operand
materialized to the output shape (`rocket_broadcast_copy`). The same-shape host kernel then runs,
bit-exact. Those are the SE-block scale and bias-add forms. `convert_test` checks each host op
against an independent NHWC reference.

## Quantized convolutions

### The fp16 route

The default for an int8 or uint8 convolution is the [fp16 route](README.md#terms). The delegate
dequantizes the filter once at Prepare, with its per-axis scales and zero points, and the input
per inference. It runs the fp16 convolution and requantizes the output on the host, clamped to
the output type. The int32 bias is dequantized by `in_scale * w_scale[oc]`.

The input dequantization runs on 16-channel by 16-pixel NEON blocks. A 256-entry fp16 table,
with the scale and zero point applied, handles the remainder and narrow planes. Both are
bit-identical to the per-element arithmetic.

The fp16 route approximates TFLite's int32-accumulating kernel and is not bit-identical to it.
`convert_test`'s quantized shapes therefore check the glue against a reference that performs the
same dequantize, fp16 convolution and requantize.

### Native int8 routes

`native_int8=1` runs int8 and uint8 convolutions with an exact int8 accumulate on the NPU. CPU
TFLite is the reference for every native route:

- **The int32 route** writes the int32 accumulator and requantizes on the host, which matches
  TFLite exactly. An SSD head is bit-identical to CPU TFLite (`max|delegate-CPU|=0`), and a
  MobileNetV2 block's mean deviation is half the fp16 route's.
- **The int8-out routes** requantize on the NPU. They match TFLite's accumulator exactly and its
  output within one count at a rounding boundary.

Under the default options a convolution takes this route:

| Convolution | Route | Entry |
|---|---|---|
| 1×1, aligned (K and N padded to 32, `IH·IW%4==0`), plane below `pw_i8out_min_m` or not taken by an int8-out route | Resident int8 matmul, int32 out, host requant | `rocket_matmul_int8_prepacked` (`mm_int8`) |
| uint8, per-tensor weights, direct or 1×1 at `pw_i8out_min_m` pixels or more | Int8-out | `rocket_conv2d_int8_q` (`direct_i8out`) |
| int8, per-axis symmetric weights, output plane of `pw_i8out_min_m` pixels or more | Int8-out | `rocket_conv2d_int8_q_perc` (`direct_perc`) |
| int8 per-tensor, int8 per-axis below the plane threshold, uint8 per-axis, or any shape the int8-out planner refuses | Int32 | `rocket_conv2d_int8`, multicore `rocket_conv2d_int8_mt` |
| Depthwise, int8 or uint8, per-tensor | Int8-out | `rocket_conv2d_dw_int8` |
| Depthwise, int8, per-axis symmetric weights | Int8-out | `rocket_conv2d_dw_int8_perc` (`dw_perc`) |
| Depthwise, uint8 per-axis, or past the int8 programs' limits | The fp16 route | `rocket_conv2d_fp16` |

With `direct_i8out=0` and `direct_perc=0`, every direct and 1×1 convolution takes the int32 route.
With `dw_perc=0`, a per-axis depthwise takes the fp16 route. If the fp16 planner accepts a
convolution past every int8 program's limits, it takes the fp16 route, and otherwise it is
declined.
`rocket_conv2d_int8_q`, `rocket_conv2d_int8_q_perc` and `rocket_conv2d_dw_int8_perc` are RK3588
only.

The int8-out routes differ from TFLite by one count on a small share of their outputs, and none
differs by more [HW sweep]:

- 0.037% of MobileDet's direct-convolution outputs
- 0.156% of the outputs of EfficientDet-Lite0's 102 convolutions, run on TFLite's own input
- 0.050% and 0.074% of the 1×1 outputs `pw_i8out_min_m` moves in MobileDet and SSD MobileNet v2

The depthwise int8-out route differs by one count on 11 of 4096 outputs of one real model layer
(`conv_dw_int8_runtime`).

> [!NOTE]
> A Mesa Teflon capture is not a reference for the int8 routes. Mesa's `rocket` driver is a uint8
> driver, and its output for an int8 model differs from CPU TFLite by up to 162.

#### uint8 convolutions

A uint8 direct convolution runs the int8 program on recentered bytes, `x=in_q-128` and
`y=w_q-128`, both always in [-128,127]. With α=128−in_zp and β=128−w_zp, the per-channel
constants `α·Wy + N·α·β` fold into the bias. The asymmetric weight zero point leaves a
per-output-pixel box-sum `β·Sx`, which the int32 route adds at requant
(`rocket_in_window_sum_i8_band`, `rocket_out_nchw_to_nhwc_q_per_axis_u8_band`). The delegate
skips it where every weight zero point is 128. The int8-out route programs the weight zero point
on the DPU's CPEND operand instead, so it has no int32 readback, box-sum or host requant.

MobileDet is uint8 per-tensor, with weight zero points spread 91-177 and almost never 128, so
the box-sum is what makes it exact. All 66 of its `CONV_2D` run on the native uint8 routes, and
`profile=1` tags them `[native u8]`. Their scores have 35-40% lower error against the CPU than the
fp16 route's, several bit-identical. All 13 native uint8 shapes in `convert_test` reach
`max|dq|=0` on the NPU.

#### Native depthwise

A per-tensor depthwise takes `rocket_conv2d_dw_int8` at any weight zero point. The input zero
point folds into the bias. The DPU's CPEND operand adds `-w_zp` times each window's input sum,
the term that cannot fold into the bias. uint8 bytes are recentered by 128 with every zero point.
All three of a depthwise's tensors must be int8, or all uint8.

The delegate materializes the `SAME` padding on the host, so a stride-2 layer with an odd pad is
claimed too. `rocket_conv2d_dw_int8_plan` decides the shape at claim time, and the fused
activation is applied as TFLite's quantized clamp after the requant. A plane past one CBUF pass
runs in row bands, and in column bands where needed. SSD MobileNet v2's two 150×150 layers are
therefore claimed too.

A per-axis int8 depthwise takes `rocket_conv2d_dw_int8_perc`, with each channel's scale on the
BS stage's per-channel multiplier. TFLite quantizes a depthwise filter per channel by default.

#### Measured gains

| Route | Model | Warm speedup, `performance` | Warm speedup, `ondemand` | COCO mAP, before -> after | CPU interpreter's mAP |
|---|---|---:|---:|---:|---:|
| Native depthwise, per-tensor | MobileDet | 1.06x | Not characterized | 0.0012 lower | Not characterized |
| Native depthwise, per-tensor | SSD MobileNet v2 | 1.05x | Not characterized | 0.0001 lower | 0.3215 |
| `dw_perc` | EfficientDet-Lite0 | 1.095-1.104x | Not characterized | 0.2998 (fp16 route) -> 0.3019 | 0.2996 |
| `direct_i8out` | MobileDet | 1.22-1.27x | 1.21-1.22x | 0.3990 -> 0.3994 | Not characterized |
| `direct_i8out` | SSD MobileNet v2 | 1.19-1.20x | 1.17x | 0.3197 -> 0.3213 | 0.3215 |
| `pw_i8out_min_m=256` | MobileDet | 1.37-1.41x | 1.95-2.00x | 0.3994 -> 0.4040 | 0.3980 |
| `pw_i8out_min_m=256` | SSD MobileNet v2 | 1.23-1.28x | 1.90x | 0.3213 -> 0.3229 | 0.3215 |
| `direct_perc` | EfficientDet-Lite0 | 1.21-1.25x | 1.29-1.33x | 0.3019 -> 0.3023 | 0.2996 |

**Conditions:** RK3588 (Turing RK1), the mainline `rocket` driver, NPU at ~816 MHz (the 600 MHz
request), warm. Each speedup is against the build before the route, and the routes were added in
the table's order. The `performance` column has the governor pinned and the process on the A76
cores (`taskset -c 4-7`). The `ondemand` column has `scaling_min_freq` at 1.2 GHz. mAP@[.5:.95] is
over 100 COCO val2017 images. Measured 2026-09-27 and 2026-09-28.

With `pw_i8out_min_m=256`, 1×1 convolutions of 16×16 pixels and up move from the int8 matmul to
the int8-out direct convolution. Its output needs no int32 readback or host requant. The matmul
route also loses more to parked cores: on `ondemand` its time is 1.95x its pinned time, against
the direct route's 1.36x.

#### Evidence

Each speedup is from four passes with the configuration order rotated, 30 invokes per process.
The native depthwise rows measure MobileDet 202-206 ms -> 191-193 ms and SSD MobileNet v2
200-201 ms -> 190-192 ms. Their mAP moved by -0.0012 and -0.0001 against a CPU-to-delegate spread
of +0.001 and -0.006. SSD MobileNet v2's two 150×150 depthwise layers stayed on the fp16 route
in that step, so its 0.3157 is the native depthwise row's result. With those two layers moved to
the int8 route in row bands, its mAP measured 0.3197.

On `performance`, EfficientDet-Lite0 measured 133 -> 121 ms under `dw_perc` and 123 -> 101 ms
under `direct_perc`. MobileDet measured 93 -> 75 ms under `direct_i8out` and 77 -> 56 ms under
`pw_i8out_min_m`, and 150 -> 76 ms on `ondemand`. The 256-pixel threshold was measured on both
sides, at 100 and 400 pixels.

### `nchw_resident`

`nchw_resident=1` keeps a partition-internal tensor in fp16 NCHW, after bias and activation,
between two convolutions on the fp16 route. That skips the NHWC-to-NCHW transpose and the
requantize-dequantize round trip at that boundary. It is off by default because it changes the
int8 numerics: a tensor that skips the requant keeps sub-count drift.

A MobileNetV2 block runs 12.66 -> 9.34 ms warm (1.36x), and the SSD head 2.32 -> 1.96 ms.

A tensor stays resident only where no consumer reaches a partition output through transparent ops
without a requantization barrier between them. The transparent ops are `RESHAPE`,
`CONCATENATION`, `TRANSPOSE`, `PAD`, `SLICE` and `STRIDED_SLICE`, and every other consumer is a
barrier. The last convolution before an output therefore always reads a requantized input. The
rule keeps damped chains resident, such as a block's projection into its `ADD`, and drops the
SSD predictors' direct inputs. Residency also excludes the matmul routes and every native int8
producer and consumer.

Against the int8 reference, the barrier rule alone does not keep deep graphs close. High-gain
int8 convolutions compound the drift, to `max|delegate-CPU|` of 52 on the SSD head and 63 on a
convolution stack. Against the fp32 model, through `tools/precision_probe.py` on a paired float
and int8 model, `nchw_resident=1` is neutral on the maximum error. The shared final output
quantization dominates there. It is better on the mean, and moves toward fp32. The large difference
from the int8 reference is distance from a lossy reference rather than error.

## The RK3576 graph

On an RK3576 the delegate lowers each delegated partition onto `librocketnpu`'s neutral
description, and the driver library's `rocketgraph` plans and runs it as one network. The
README's [RK3576 section](README.md#the-rk3576) has the claimed op set and the results. Only the
`profile` and `rk3576` options have an effect on this path.

The delegate does the TFLite lowering, in [rocket_rk3576_net.h](rocket_rk3576_net.h):

- The claim
- The geometry, with TFLite's asymmetric `SAME` padding expressed as an output extent. The
  hardware derives the pad its last window consumes rather than taking a trailing pad.
- The uint8-to-int8 rebase
- The weight transposes, the buffers and the boundary layout

`QUANTIZE` is claimed only as a requantization within one type. `profile=1` prints the plan, as
`[rocket/rk3576] <n> layers: <j> joins, <f> far links, <r> run(s) over <l> layers`.
`tools/rk3576_net_ab.py` scores a classifier against the interpreter with no delegate.

> [!CAUTION]
> Without an NPU device, Prepare fails with `[rocket/rk3576] no NPU device` and the model does not
> load. The RK3588 routes compute on the library's CPU reference instead.

### Per-axis quantization

A convolution runs as one hardware task with one output converter shift. Each output channel
reaches its own scale through a 16-bit multiplier in its coefficient group. A shift in the same
stage lets the task's largest channel use the multiplier's whole range. A channel whose output
cannot vary over its reachable inputs, such as a pruned all-zero filter, is programmed as that
one byte. Only a live channel below 1/65534 of its task's largest live scale is refused.

`rocket_conv2d_int8_perchannel_plan_rk3576()` answers that from the weights, the bias and the
quantization scales, with no device. The delegate asks it at claim time, in its Prepare callback,
where a refusal declines one node. The same refusal in the node kernel's Prepare would fail the
whole model.

On EfficientDet-Lite0 nothing is refused and no layer is split into output-channel tiles. It runs
55.5 ms against 83.6 ms without the shift (0.661x paired). Its mAP@[.5:.95] over 500 val2017
images is 0.2818 against the CPU's 0.2823 [HW sweep 2026-09-27, `rocket` 1.6.0].
`ROCKET_RK3576_PC_SHIFT=0` restores the shift-0 ramp, whose detections are byte-identical to a
build without the shift.

## Delegate options

The delegate reads these external-delegate options. `load_delegate` passes each value as a
string.

| Option | Default | Effect |
|---|---|---|
| `native_int8` | `0` | Runs int8 and uint8 convolutions on the [native int8 routes](#native-int8-routes) instead of the fp16 route. The accumulate is exact, and the default int8-out routes requantize on chip within one count of TFLite |
| `mm_int8` | `1` | Sends an aligned 1×1 int8 or uint8 convolution to the resident int8 matmul. Active only under `native_int8`, with a device open |
| `dw_perc` | `1` | Sends a per-axis int8 depthwise to `rocket_conv2d_dw_int8_perc` (RK3588). Active only under `native_int8`. `0` keeps it on the fp16 route |
| `direct_i8out` | `1` | Sends a uint8 convolution with per-tensor weights to the int8-out entry `rocket_conv2d_int8_q` (RK3588). Active only under `native_int8`. `0` keeps the int32 route, whose host requant is TFLite's |
| `direct_perc` | `1` | Sends a signed int8 convolution with per-axis symmetric weights and an output plane of at least `pw_i8out_min_m` pixels to `rocket_conv2d_int8_q_perc` (RK3588). Active only under `native_int8`. `0` keeps the int32 route |
| `pw_i8out_min_m` | `256` | The plane size, in pixels, from which a 1×1 that `direct_i8out` or `direct_perc` takes runs on the int8-out direct convolution instead of the int8 matmul, and from which `direct_perc` applies. `0` keeps every eligible 1×1 on the matmul and lets `direct_perc` take any plane. Active only under `native_int8` |
| `nthreads` | `4` | Worker count, clamped to 1-8. It sizes the 1×1 matmul fan-out across the three NPU cores, the native direct-convolution pool `rocket_conv_pool`, the host requant and box-sum fan-out, and the matmul context `FULLY_CONNECTED` uses |
| `min_macs` | `0` | The smallest convolution to claim, in `OC·OH·OW·IC·KH·KW` (`OC·OH·OW·KH·KW` for a depthwise). `0` claims every supported convolution |
| `aux_ops` | `1` | Claims the host ops listed below. `0` restricts the delegate to the convolutions and `FULLY_CONNECTED` |
| `profile` | `0` | Prints per-op timing to stderr: the `breakdown in/conv/out` split, the resident set, and each convolution's route and input plane |
| `nchw_resident` | `0` | Keeps tensors between fp16-route convolutions in fp16 NCHW. [`nchw_resident`](#nchw_resident) has the detail |
| `rk3576` | `1` | On an RK3576, runs each delegated partition as the [RK3576 graph](#the-rk3576-graph). `0` turns it off for comparison: the RK3588 routes then run, and their entries refuse on the RK3576. No effect on another part |

An option name the delegate does not know is ignored, with one line on stderr naming it. A
misspelled option runs the delegate at that option's default.

`profile=1` names each convolution's route as one of `native-dw-perc`, `native-dw-uint8`,
`native-dw-int8`, `native-int8-perc`, `native-int8`, `native-uint8-i8out`, `native-uint8` or
`fp16`. The route name does not tell a 1×1 on the int8 matmul from one on the direct convolution.
The breakdown line's tag does: `[mm i8]` for the matmul.

`aux_ops` claims these host ops:

- Elementwise: `ADD`, `SUB`, `MUL`, `DIV`, `MAXIMUM`, `MINIMUM` and `PRELU`
- Activations: `HARD_SWISH`, `LOGISTIC`, `TANH`, `ELU`, `LOG`, the `RELU` family, `LEAKY_RELU`,
  `EXP`, `SQRT`, `RSQRT`, `ABS`, `NEG`, `SQUARE` and `FLOOR`
- Pooling and reductions: `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `MEAN`, `REDUCE_MAX` and
  `REDUCE_MIN`
- Layout: `RESHAPE`, `CONCATENATION`, `TRANSPOSE`, `PAD`/`PADV2`, `SLICE`, `STRIDED_SLICE` and
  `SPLIT`/`SPLIT_V`
- Other: `RESIZE_*`, `L2_NORMALIZATION`, `TRANSPOSE_CONV`, `SOFTMAX`, `LOG_SOFTMAX` and `CUMSUM`

Some claimed ops also have an opt-in NPU route, each defaulting to `0`:

| Option | Sends to the NPU |
|---|---|
| `act_npu` | The LUT activations listed under the [op mapping](#tflite-op-mapping), and `PRELU` |
| `ew_npu` | `MAXIMUM` and `MINIMUM`, float and same shape, on the DPU EW ALU |
| `pool_npu` | Float `AVERAGE_POOL_2D` (`VALID` padding only), `MAX_POOL_2D`, `MEAN` and `REDUCE_*`, on the PPU |
| `resize_npu` | A float nearest resize at an integer factor |
| `norm_npu` | Float `L2_NORMALIZATION` |
| `fc_npu` | `FULLY_CONNECTED` with `M%4==0`, with a device open |

## Environment variables

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `ROCKET_CPU_AFFINITY` | `librocketnpu` | unset: the big cores | A CPU list to pin worker threads to. `off`, `none`, `0` or an empty value turns pinning off. Unset, every one-thread context's worker pins to the first big core, CPU 4 on the RK3588, so a pool needs one distinct core per process |
| `ROCKET_CONV_BATCH` | `librocketnpu` | off | A value above `0` submits a native int8 direct convolution's tiles as one job. It helps only a convolution-tile-heavy model under a contended pool |
| `ROCKET_REQUANT_SCALAR` | the delegate | unset | A value other than empty or `0` selects the scalar requant epilogues instead of the NEON ones. NEON builds only |
| `ROCKET_BOXSUM_MODE` | the delegate | `0` | The box-sum implementation: `0` separable, `1` per-window, `2` scalar. It applies on NEON builds at unit stride and dilation, and a `KW` of 1 always uses `1` |
| `ROCKET_PROF_POOL` | the delegate | unset | Any value other than `0` counts the host `parallel_oh` fan-out and prints one `[rocket-pool-probe]` line at exit |
| `ROCKET_RK3576_PC_SHIFT` | `librocketnpu` | unset | `0` restores the RK3576 per-axis shift-0 ramp ([Per-axis quantization](#per-axis-quantization)) |

`ROCKET_CONV_BATCH` was neutral single-stream, and on MobileDet and EfficientDet under a pool. It
gave 7.6% more aggregate throughput only on a convolution-tile-heavy test unit at four processes
[HW sweep, June 2026]. Single-stream latency on that build was host-bound, not submit-bound.

`ROCKET_PROF_POOL` prints `fanout_calls`, `threads_spawned` and `fanout_wall`: the fan-out calls,
the threads created and the wall-clock time in fan-out regions. It prints nothing with
`nthreads=1` or on the RK3576 graph. On SSDLite-MobileDet on 2026-07-01, thread creation and
joining measured ~3.1 ms of a ~380 ms inference, ~0.83% [HW sweep].

**Not characterized:** the thread cost on the current build. With an unchanged fan-out count, the
same cost would be ~6% of a ~51 ms inference [expected].

## Delegate counts

The outputs alone cannot show whether the NPU ran. Without a device on the RK3588, the delegate
still claims its nodes, and every convolution runs on the library's CPU reference. That reference
is close enough to TFLite's output to pass a comparison. The delegate therefore exports three process-wide counts:

```c
void rocket_delegate_counts(long *nodes_claimed, long *ops_npu, long *ops_host);
```

`nodes_claimed` is the nodes the delegate claimed in its Prepare callback. Per invoke, `ops_npu`
counts the claimed ops that completed on an NPU route. That is a convolution with a device open,
or an op under a `*_npu` option whose result came back from the device. `ops_host` counts the rest,
including an NPU attempt that fell back to the host kernel. A null pointer is skipped, and the counts accumulate over every
delegate instance in the process. On the RK3576 graph the counts are per planned layer, not per
node.

From Python, `ctypes.CDLL` on the path the delegate was loaded from reaches the same library. Read
the counts before and after the run you score. `tools/delegate_counts.py` does that, and
`coco_map.py`, `rk3576_net_ab.py`, `run_delegate.py` and `dw_native_e2e.py` use it.

## Multi-camera throughput

Throughput comes from running several detection contexts at once, one process per stream, rather
than from a faster single stream. One context's host packing and readback overlap another's NPU
work. Warm single-stream MobileDet takes 51.4 ms with the governor pinned and 73.5 ms on
`ondemand`, about 19 and 14 inferences a second [HW sweep 2026-10-29].

The recipe needs no delegate or driver change. Run one process per stream, as Frigate runs
cameras. In each, set `ROCKET_CPU_AFFINITY` to a distinct big core before the delegate loads:
`4`, `5`, `6` and `7`. The contexts then spread across the four A76 cores rather than all pinning
to the first one.

`tools/pool_throughput.py` runs the delegate on a 1×1-convolution test unit whose cost is the
submit:

| Processes | Aggregate inferences/s, relative |
|---:|---:|
| 1 | 1.00x |
| 2 | 2.17x |
| 3 | 3.11x |
| 4 | 3.56x |

That scaling matches `rocket-userspace`'s own multi-context throughput test. The contexts share one
NPU, so `tools/pool_hash.py` runs the same fan-out and checks that every process's output hash
equals the single-process one. On SSDLite-MobileDet, one process per A76 core at `nthreads=1`, it
measured 3.33-3.51x at four processes, with one hash throughout [HW sweep 2026-10-29]. The aggregate
rate is 52.9-53.6 inferences a second, at 928 MHz on the mainline driver. On 2026-08-26 it measured
3.51-3.61x across the mainline and BSP drivers.

In live Frigate, a pool of detectors `rocket0` to `rocket3`, each pinned to its own A76 core,
served four SSD-MobileDet cameras:

| Detectors | Aggregate `detection_fps` | Scaling |
|---:|---:|---:|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | 2.98x |

**Conditions:** RK3588 (Turing RK1), the mainline `rocket` driver, NPU at ~816 MHz (the 600 MHz
request), `num_threads: 1` per detector, measured 2026-07-01. The NPU IRQs were pinned to the
big cores with `rocket-userspace`'s `tools/npu_set_irq_affinity.sh throughput`. A single inference
took ~336 ms on that build.

**Not characterized:** the live Frigate pool at the current 51.4 ms single stream.

The four A76 cores ran one detector each at ~73% busy, while the Cortex-A55 cores handled video
decode. The live pool scaled below the test unit. That build's inference was bound by the host's
cube packing and gathering, not by the submit. Per-inference latency rose from 338
to 424 ms as the four contexts contended for DRAM bandwidth.

The Frigate plugin is in [frigate/](frigate/): one process per camera at `num_threads: 1`, with
the per-process affinity. Its `config.yml` runs one detector at `num_threads: 4`, and
[frigate/README.md](frigate/README.md) gives the pool layout.

## Host-side work

### Host requantization

The int8-out routes requantize on chip and skip the host requant. On MobileDet all 23 direct
convolutions and 27 of the 43 1×1s, those of 400 pixels and up, take them. On EfficientDet-Lite0
`direct_perc` takes the per-axis convolutions of 256 pixels and up.

The host requant runs where the int32 route remains:

- A 1×1 on the int8 matmul
- A per-axis int8 direct convolution below `pw_i8out_min_m` pixels
- Every per-tensor int8 and per-axis uint8 direct convolution
- Any convolution the int8-out planner refuses
- Every direct and 1×1 convolution with `direct_i8out=0` and `direct_perc=0`

Every host epilogue is NEON-vectorized, and `ROCKET_REQUANT_SCALAR` selects the scalar forms.

**Not characterized:** the host requant's share of the current single-stream time. A `profile=1`
run prints the per-op `breakdown in/conv/out` split that measures it.

#### The 1×1 epilogue

`rocket_out_mn_to_nhwc_q_per_axis_band` and its uint8 form `_u8_band` requantize a matmul's
output. A 1×1's output has the channels contiguous in both the int32 read and the NHWC write. The
epilogue therefore requantizes 8 channels per step with no scatter. It is bit-identical to the scalar
`lrintf` epilogue, because every intermediate fits int32 and `vcvtnq_s32_f32` rounds to nearest
with ties to even. It gave 5.5% on warm MobileDet and 9.3% on EfficientDet-Lite0 in June 2026,
with every aligned 1×1 on the matmul.

#### The box-sum

`rocket_in_window_sum_i8_band` computes the uint8 route's `β·Sx` term. It is NEON-vectorized at
unit stride and dilation, skipped when every weight zero point is 128, and split across the A76
cores by output-row band.

For `KW>1` it is separable. The `IC·KH·KW` window sum splits into a channel and row reduction
`T[iw]=Σ_{ic,kh} x[ic,oh·sy+kh·dy,iw]`, 16 wide, and a horizontal `KW` window sum
`Sx[oh,ow]=Σ_{kw} T[ow+kw]`, 4 wide. That cuts the vector column passes from `IC·KH·KW` to
`IC·KH + KW`, with both passes vectorized. Integer adds are order-independent, so the result is
byte-identical to the per-window and scalar forms. `convert_test` reaches `max|dq|=0` under each
`ROCKET_BOXSUM_MODE`, and `boxsum_bench` cross-checks the three checksums.

| Kernel | Speedup over the per-window NEON form |
|---|---|
| 3×3 | 2.6x |
| 5×5 | 3.3x |
| 7×7 | 4.3x |

The separable form is ~10-16x the scalar one. Its whole-model effect measured below 1% in
June 2026. Under the default routes it runs only on a uint8 direct convolution left on the int32 route.
A uint8 1×1 on the matmul sums its pixels during the recentering instead.

#### The direct-convolution epilogue

`rocket_out_nchw_to_nhwc_q_per_axis_band` and `_u8_band` requantize a direct convolution's NCHW
accumulator into NHWC. They run channel-outer, so the NHWC write `dst[pix·OC+oc]` is strided. The
computation vectorizes over 4 contiguous source pixels and writes the 4 result bytes out one at a
time. It is bit-identical to the scalar epilogue, with the same float order: `convert_test`
reaches `max|dq|=0`, and both detectors are byte-identical with NEON and without. Its function
benchmark is 1.6-3.4x, and its whole-model effect measured below 1% in June 2026.

### Evidence

Single-stream latency by build, each change measured against the build before it, warm, at
~816 MHz (the 600 MHz request). The first two changes are June 2026 builds, and the native
depthwise and everything after it are from 2026-09-27 on. Every change after the native depthwise
left the outputs byte-identical, and the three before it were bit-exact.

| Change | MobileDet | SSD MobileNet v2 | EfficientDet-Lite0 |
|---|---|---|---|
| Resident device state: weights packed at Prepare, a resident convolution buffer pool (`rocket_conv_ctx`) with each buffer role cached by power-of-two size. A warm MobileNetV2 block went ~28 -> 12 ms (~3x) | | | |
| Multicore direct convolution (`rocket_conv_pool`, `rocket_conv2d_int8_mt`) across the three cores | 560 -> 458 ms (1.21x) | | |
| 1×1 int8 and uint8 on the resident int8 matmul (`mm_int8`) | 366 -> 336 ms (8%) | | |
| Native depthwise, per-tensor | 202-206 -> 191-193 ms | 200-201 -> 190-192 ms | |
| The library's depthwise pack blocked rather than per element | 171-172 ms (1.12x) | 172-173 ms (1.10x) | |
| The same blocked pack in the fp16 and int8 direct convolutions | 150-152 ms | 130-132 ms | 307-309 -> 255-257 ms |
| The 1×1 route's matmul buffers sized to the tiles a call has | 130-131 ms | 119-120 ms | 228-230 ms |
| Quantized unary and concatenation through a 256-entry table | 1.04x | 1.04x | 1.22-1.25x (`LOGISTIC` 35 -> 2.6 ms, concatenation 11 -> 0.5 ms) |
| A one-K-tile matmul gathered straight into C | 1.07-1.09x | 1.04-1.07x | 1.10-1.11x |
| `lrintf` inlined by `-fno-math-errno` | 1.02-1.03x | 1.03-1.04x | 1.05-1.08x |
| The library's convolution packs transposed with NEON, 16 pixels of a channel group at a time | 1.07-1.08x | 1.11-1.13x | 1.06-1.07x |
| The delegate's own NHWC and NCHW conversions transposed, a NEON `ADD` and a byte-max pool | 1.04-1.06x, to ~100 ms | 1.09-1.11x, to 85-86 ms | 1.10-1.12x, to 133-135 ms |

The int8-out routes and `dw_perc` followed, in [Measured gains](#measured-gains). The 256-entry
input dequantization table measured ~1.2x the per-element arithmetic on the A76 for
feature-map-shaped inputs. The 2026-06-21 build, with the host requant on every convolution,
measured MobileDet mAP@[.5:.95] 0.3321 against the CPU's 0.3318 over 500 val2017 images.
