# tflite-rocket: API and tuning reference

The full TFLite op mapping, the quantization internals (native int8/uint8, the
`nchw_resident` inter-op-buffer lever), the complete external-delegate option reference,
and the host-side work and remaining-gaps analysis. The [README](README.md) is the guide,
and this is the reference.

## TFLite op mapping

| TFLite op | NPU path | status |
|---|---|---|
| `CONV_2D`, 1×1, stride 1 (**pointwise**), matmul-aligned | **the matmul** (`rocket_matmul_fp16_mt`, multicore) | wired |
| `CONV_2D`, KxK / stride / `SAME`\|`VALID` pad / dilation (float) | fp16 conv (`rocket_conv2d_fp16`) | wired |
| `DEPTHWISE_CONV_2D` (depth_multiplier 1, float + int8/uint8) | native depthwise conv (`rocket_conv2d_fp16` `depthwise=1`, G=32), **HW-validated bit-exact** | wired |
| `CONV_2D`/`DEPTHWISE_CONV_2D`, **signed int8** (`native_int8=1`) | **NATIVE int8**: direct int8xint8->int32 (`rocket_conv2d_int8`, **multicore `_mt`**) + host per-axis requant; per-tensor DW int8-out on-chip requant (`rocket_conv2d_dw_int8`). **Exact int8**, SSD head bit-identical to CPU TFLite | wired (`native_int8`) |
| `CONV_2D` **1×1** int8/uint8 (`native_int8=1`, `mm_int8` default-on) | **resident int8 matmul** (`rocket_matmul_int8_prepacked`, a 1×1 *is* a matmul), K/N pad %32 and M%4 gate; falls back to the conv otherwise. Bit-exact + **nt-deterministic**. +8% warm | wired (`mm_int8`) |
| `CONV_2D` **uint8** DIRECT (`native_int8=1`) | **NATIVE uint8**: recenter to int8 (`x=in_q−128`, `y=w_q−128`), reuse `rocket_conv2d_int8` (**multicore `_mt`**), fold the centering into `eff_bias` + a per-output-pixel box-sum (`rocket_in_window_sum_i8`, **NEON-vectorized**, **separable** for KW>1, skipped when w_zp==128). **Exact uint8**, all 66 MobileDet convs native, scores ~35-40% lower error vs CPU than fp16 | wired (`native_int8`) |
| `CONV_2D`, int8/uint8 quantized (default) | dequant↔fp16 boundary, reuses the fp16 conv (uint8 DEPTHWISE / per-channel DW / asym-pad DW always) | wired |
| `FULLY_CONNECTED` | **host** matmul (`rocket_fc_f`) by default, which beats the dispatch-bound NPU for a one-shot FC; **`fc_npu=1`** routes a large GEMM-shaped FC (M%4, K->%32, N->%16 zero-pad) to the resident fp16 matmul (M==1 GEMV stays host) | wired (`fc_npu` opt-in) |
| `HARD_SWISH` / `LOGISTIC` (sigmoid) / `TANH` / `ELU` / `LOG` / `RELU` / `RELU6` / `RELU_N1_TO_1` / `LEAKY_RELU` / `EXP` / `SQRT` / `RSQRT` / `ABS` / `NEG` / `SQUARE` / `FLOOR` (standalone builtins) | **host** kernel (`rocket_unary_f/_q`, float + int8/uint8), which keeps `conv->act->conv` one contiguous partition. The `RELU` family and `NEG`/`SQUARE`/`FLOOR`/`ABS` are exact float arithmetic (byte-identical to CPU TFLite through the full delegate); `EXP`/`SQRT`/`RSQRT`/`LEAKY_RELU` are libm-exact. **`act_npu=1`** routes the curved kinds to the on-NPU DPU LUT (`ELU`->`rocket_elu_fp16`, `LEAKY_RELU`->`rocket_leaky_relu_fp16`; `EXP`/`SQRT`/`RSQRT`/`ABS`/`LOG` to the domain-limited LUT; `RELU`/`RELU6`/`RELU_N1_TO_1`/`NEG`/`SQUARE`/`FLOOR` are host-only, exact). `RELU`/`RELU6`/`RELU_N1_TO_1` are also fused into a preceding `CONV_2D` when the converter folds them. | wired (`aux_ops`) |
| `MAXIMUM` / `MINIMUM` (two-tensor elementwise) | **host** NHWC kernel (`rocket_binary_f/_q`), which keeps partitions contiguous; **`ew_npu=1`** routes float to the on-NPU DPU EW ALU (`rocket_ew_max/min_fp16`, **bit-identical** to host) | wired (`aux_ops`) |
| `MUL` / `SUB` / `DIV` (two-tensor elementwise, same-shape **or broadcast**, fused act) | **host** kernel (`rocket_arith_f/_q`, float + int8/uint8), siblings of `ADD` (`SUB`/`DIV` are non-commutative: `in0 op in1`); a broadcasting operand (per-channel `[C]`, scalar, general right-aligned) is materialized to the output shape first (`rocket_broadcast_copy`); keeps partitions contiguous | wired (`aux_ops`) |
| `PRELU` (per-channel slope) | **host** kernel (`rocket_prelu_f/_q`, float + int8/uint8), the alpha constant dequantized to float[C]; **`act_npu=1`** routes float to the NPU (`rocket_prelu_fp16`, ew_max/mul, **bit-identical**) | wired (`aux_ops`) |
| `MEAN` / `REDUCE_MAX` / `REDUCE_MIN` (axes [1,2] = spatial) | **host** kernel (`rocket_reduce_spatial_f/_q`), GlobalAvg/Max/MinPool; **`pool_npu=1`** routes float to the NPU PPU (`rocket_global_{avg,max,min}pool_fp16`) | wired (`aux_ops`) |
| `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR` (float + int8/uint8) | **host** kernel (`rocket_resize_*`, TFLite-exact align_corners/half_pixel coordinate math); **`resize_npu=1`** routes an integer-factor align_corners=false/half_pixel=false **nearest** resize to the NPU (`rocket_upsample_nearest_fp16` = block replication, **bit-identical**) | wired (`aux_ops`) |
| `L2_NORMALIZATION` (over the channel axis) | **host** kernel (`rocket_l2norm_f/_q`, float + int8/uint8); **`norm_npu=1`** routes float to the NPU (`rocket_l2norm_fp16`) | wired (`aux_ops`) |
| `TRANSPOSE_CONV` (float) | **host** scatter kernel (`rocket_transpose_conv_f`, TFLite pad mapping), which keeps `conv->tconv->conv` contiguous; int8 + the on-NPU route (`rocket_conv_transpose2d_fp16`) are the follow-on | wired (`aux_ops`) |
| `SOFTMAX` (last axis), `LOG_SOFTMAX` (last axis), `CUMSUM` (last axis) | **host** kernel (`rocket_softmax_f/_q` with the `beta` logit scale, `rocket_logsoftmax_f/_q`, `rocket_cumsum_f/_q`, float + int8/uint8), stable row-max subtraction, keeps partitions contiguous; the NPU routes (`rocket_softmax_fp16` / `rocket_logsoftmax_fp16` / `rocket_cumsum_fp16`) are the follow-on | wired (`aux_ops`) |
| `ADD` (residual), `AVERAGE`/`MAX_POOL_2D`, `CONCATENATION`, `RESHAPE` (float + int8/uint8) | **host** NHWC kernel (`rocket_ops.h`), which keeps partitions contiguous (`AVERAGE`/`MAX_POOL_2D` opt-in `pool_npu`) | wired (`aux_ops`) |
| `TRANSPOSE`, `PAD`/`PADV2`, `SLICE`, `STRIDED_SLICE`, `SPLIT`/`SPLIT_V` (float + int8/uint8) | **host** byte-exact kernel (`rocket_transpose/slice/pad_bytes`, `rocket_ops.h`), pure layout moves (values pass through, quant scale/zp unchanged), so they keep `conv->layout->conv` ONE partition instead of splitting around a CPU node. Generic rank <= 8. `STRIDED_SLICE` handles any **non-zero** per-axis stride, positive or negative (the signed strided gather `rocket_strided_slice_bytes`, begin[] = the first/HIGH index when stride<0), with all five masks resolved per TFLite's Start/Stop rules: begin/end/shrink_axis plus **ellipsis_mask** (expands to full-range axes) and **new_axis_mask** (an output-only size-1 dim) via a spec->input-axis expansion that feeds the same byte gather. (The standard TF->TFLite converter lowers ellipsis/new-axis away, into explicit begin/end+masks and a separate `RESHAPE`, so these masks only arrive from hand-authored or non-standard producers; supported defensively.) No NPU route (the NPU has no on-chip layout-conversion engine) | wired (`aux_ops`) |
| everything else | CPU fallback (automatic) | n/a |

MobileNet and MobileDet are **pointwise-heavy**, so the matmul-aligned 1×1 fast path
offloads a real fraction of the network at multicore matmul speed. The general conv path
covers the strided, KxK and dilated convs, such as the RGB stem.

The mainline-rocket conv path is **fp16**, using the same `(target<<48)|(value<<16)|reg`
encoding and NC1HWC2 packing the matmul uses. The general conv is therefore an extension of
the matmul's domain rather than a rewrite.

Depthwise, the MobileNet workhorse, is **claimed in the delegate**. It runs on the native
`DW_EN` path at group G=32, cracked against a Mesa Teflon regcmd capture. The delegate
reorders TFLite's `[1,KH,KW,C]` filter to the
driver's `[C,KH,KW]` (per-channel-axis-3 quant for int8/uint8) and routes to
`rocket_conv2d_fp16` with `depthwise=1`. With pointwise (1×1=matmul), strided/stem
direct conv, depthwise, and the host ADD/POOL/CONCAT seams all claimed, a
MobileNet/SSD backbone is **NPU-resident**.

### The ops around the conv (ADD / POOL / CONCAT / RESHAPE)

A real detector is more than conv. Residual `ADD` (MobileNetV2/V3 + FPN), `AVERAGE_POOL_2D` and `MAX_POOL_2D` all sit
*between* the convs. So do the `CONCATENATION` that joins an SSD head's box and class
predictors, and the `RESHAPE` in that head. Left unclaimed, each is a partition boundary.

The delegate claims them, float and int8/uint8, so the partitioner takes **contiguous
subgraphs**. It computes them with thin **host** NHWC kernels (`rocket_ops.h`) rather than
on the NPU. These are memory-bound elementwise and reduction ops, for which a dedicated
regcmd is not warranted. Each conv also transposes NHWC↔NCHW itself, so claiming them does
not save a transpose.

The win is fewer and larger partitions. **NCHW-resident inter-op buffers**, which would
also let conv skip the per-op transpose, are the documented follow-up.

Quant mirrors the conv boundary: per-tensor input and output scale and zp, requant to the
output type. `ADD` keeps each input's own scale, `CONCATENATION` requantizes each input to
the output scale, and pooling runs dequant, reduce, requant. It is gated by `aux_ops`,
default on, for HW A/B.

Broadcasting two-tensor elementwise is claimed: per-channel `[C]`, scalar, and general
right-aligned. Each operand is materialized to the output shape (`rocket_broadcast_copy`),
then the same-shape host kernel runs, bit-exact. Padding that is neither `SAME` nor `VALID`
stays on the CPU. Each op is validated
off-hardware by `convert_test` against an independent NHWC oracle, the same discipline as the
conv glue.

## Quantization

Detection models are typically **UINT8/INT8** quantized TFLite, and the NPU path is fp16.
So the delegate **dequantizes int8->fp16 at the partition boundary**, runs fp16 on the NPU,
then requantizes the output. The filter is dequantized once at `Prepare`, and the
activations per inference. That is the same trick as ggml's f32↔fp16.

The per-inference input dequant maps each stored byte through a 256-entry fp16 lookup
table, with the per-tensor scale and zero-point baked in once per call. It is bit-identical
to the per-element arithmetic and ~1.2x faster on the A76 for feature-map-shaped inputs.

This is **wired**. The gate accepts int8/uint8 `CONV_2D` and honours per-axis filter
scales and zero points, with the int32 bias dequantized by `in_scale * w_scale[oc]`. The
requant epilogue clamps to the output type.

Being fp16, it is an arithmetic approximation of TFLite's int32-accumulate int8 kernel
rather than bit-identical to it. So `convert_test`'s quantized shapes validate the glue
against an independent oracle
that performs the *same* dequant->fp16-conv->requant path.

A true native int8 conv on the NPU, with its own regcmd encoding, removes both the fp16
approximation *and* the host dequant and requant round-trip. It is the same perf direction
as the `nchw_resident` path, with **exact** int8 semantics, default-safe by construction.
This runs end-to-end and is HW-validated behind the default-off `native_int8` option:
- **DIRECT / 1×1** = int8xint8->int32 raw (`gen_conv2d_int8` -> `rocket_conv2d_int8` runtime
  with OC/IC pad and OC/OH/OW tiling), plus host per-axis requant
  (`rocket_out_nchw_to_nhwc_q_per_axis`, with the input zero-point correction folded into
  the bias). It is a **real int8 accumulate, so an SSD head is BIT-IDENTICAL to CPU TFLite**
  at `max|delegate−CPU|=0`, and a MobileNetV2 block's mean deviation halves.
- **DEPTHWISE** = int8-OUT with **on-chip requant** (`gen_conv2d_dw_int8` + `int8_out=1` ->
  `rocket_conv2d_dw_int8` runtime: uint8-centered cubes plus the Mesa zero-point bias fold).
  **per-tensor** only, and **bit-exact against Teflon ground truth** (`replay_dw_mesa` and
  `conv_dw_int8_runtime`, delegate `max|delegate−Teflon|=0`).
- **UINT8 DIRECT** = the same `rocket_conv2d_int8` runtime fed RE-CENTERED bytes,
  `x=in_q−128` and `y=w_q−128`, both always in [−128,127]. The per-OC centering constants
  `α·Wy + N·α·β`, where α=128−in_zp and β=128−w_zp, fold into `eff_bias`. The
  asymmetric weight zp's residual is a per-output-pixel **box-sum** `β·Sx` added at requant
  (`rocket_in_window_sum_i8` / `rocket_out_nchw_to_nhwc_q_per_axis_u8`, **skipped when
  w_zp==128**). No new regcmd.

  The decisive fact is that real MobileDet is uint8 per-tensor with w_zp spread 91..177,
  almost never 128. The symmetric-only trick fails there, and the box-sum is what makes it
  exact. **All 66 MobileDet CONV_2D run native uint8** (`[native u8]`). Scores are ~35-40%
  lower error against the CPU than fp16, several bit-identical, and 8/8 `convert_test` nu8
  shapes reach `max|dq|=0` on the NPU.

**Run it with `--option native_int8=1`.** Signed-int8 symmetric-weight convs and uint8
DIRECT convs at any weight zp take the native path. **uint8 DEPTHWISE, per-channel
depthwise, and asymmetric-pad depthwise stay on the dequant↔fp16 boundary.** Per-channel
DW on-chip requant needs the BS_MUL per-OC multiply path. uint8 DW native would match Teflon
rather than CPU TFLite, a perf play with an accuracy caveat, and it is not wired.

The oracle subtlety is that DIRECT int32-raw, both int8 and uint8, is a real accumulate and
matches CPU TFLite. DW int8-OUT replicates Teflon's on-chip requant, so its oracle is
Teflon, which itself differs from CPU TFLite by <=143.

### fp16-NCHW resident inter-op buffers (`nchw_resident`, opt-in)

A perf lever on the int8 conv chain. A partition-internal tensor produced and consumed
only by general-conv-path ops is kept in **fp16-NCHW**, post bias and activation, between
the ops. That skips the NHWC↔NCHW transpose **and** the int8 requant-to-dequant round-trip
at that boundary.

The win is a MobileNetV2 block warm at `12.66->9.34 ms` (1.36x), and the SSD head at
`2.32->1.96 ms`. It is **off by default**, and `nchw_resident=1` opts in, because it
changes int8 numerics: skipping the requant lets a tensor carry sub-int8 drift. Two guards
and findings:

- **Re-quantization-barrier rule**, default-on when `nchw_resident=1`. A tensor stays
  resident only if no consuming conv reaches a partition output through transparent ops
  with no requant barrier in between. The transparent ops are `RESHAPE` and `CONCAT`, and a
  barrier is another conv, `ADD` or `POOL`. The last conv before an output therefore always
  reads requantized input.
  That keeps damped chains resident, such as a block's project-to-`ADD`, and drops the SSD
  predictors' direct inputs.
- **Precision against the int8 reference.** The barrier rule alone does *not* keep the
  delegate bit-close to the int8 *reference* in deep graphs. High-gain int8 convs compound
  the drift, at SSD `max|delegate−CPU|` 52 and stack 63.

  Measured against the **fp32 ground truth**, a paired float and int8 model through
  `tools/precision_probe.py`, `nchw_resident=1` is neutral on max. The shared final output
  quant dominates there. It is **better on the mean**, and moves *toward* fp32. The big
  delta against int8 is distance from the lossy int8 reference rather than error. That is
  why `nchw_resident` stays opt-in: native int8 is the path that gets both the perf and
  exact semantics.

## Delegate options

The delegate reads these external-delegate options:

| Option | Default | What it does |
|---|---|---|
| `native_int8` | 0 | Run int8/uint8 convs on the real int8 datapath instead of the dequant↔fp16 boundary. Opt-in, and exact int8/uint8 semantics |
| `mm_int8` | 1 | Route 1×1 int8/uint8 convs to the resident int8 matmul. Active only under `native_int8` |
| `nthreads` | 4 | Worker fan-out across the 3 NPU cores. It sizes **both** the 1×1 matmul fan-out **and** the native int8/uint8 DIRECT conv worker pool `rocket_conv_pool` |
| `min_macs` | 0 | Minimum `OC*OH*OW*IC*KH*KW` to offload. 0 offloads every supported conv |
| `aux_ops` | 1 | Claim the host elementwise, pooling, activation and shape ops. Set 0 to restrict the delegate to conv only |
| `profile` | 0 | Per-op timing to stderr, including the `breakdown in/conv/out` sub-step split and the resident set |
| `nchw_resident` | 0 | Keep conv-to-conv intermediates in fp16-NCHW between ops, skipping the per-boundary transpose and the int8 requant and dequant. Opt-in, and Quantization above has the detail |

`aux_ops` claims this set: `ADD`, `MAXIMUM`, `MINIMUM`, pool, concat, reshape, the
activations (the `RELU` family, `LEAKY_RELU`, `EXP`, `SQRT`, `RSQRT`, `ABS`, `NEG`,
`SQUARE`, `FLOOR`), `PRELU`, `MEAN`, `REDUCE_*`, `RESIZE_*`, `L2_NORMALIZATION`,
`TRANSPOSE_CONV`, `SOFTMAX`, `LOG_SOFTMAX` and `CUMSUM`.

The claimed ops also have opt-in **on-NPU routes**, all defaulting to 0. The host kernel is
exact and free, and the NPU route is an extra round-trip kept for A/B and cube-resident
fusion:

| Option | Routes to the NPU |
|---|---|
| `act_npu` | DPU LUT activations, and `PRELU` |
| `ew_npu` | `MAXIMUM` and `MINIMUM`, on the DPU EW ALU |
| `pool_npu` | `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `MEAN` and `REDUCE_*`, on the PPU |
| `resize_npu` | Integer-factor nearest resize |
| `norm_npu` | `L2_NORMALIZATION` |
| `fc_npu` | `FULLY_CONNECTED` |

The driver dependency resolves via `find_package(rocketnpu)` if installed, else from a
sibling `rocket-userspace` checkout. Override with
`-DROCKETNPU_DIR=/path/to/rocket-userspace`.

### Recommended configuration

For a detector (Frigate's regime), the two settings that matter, plus one that is not a delegate
option at all:

- **`native_int8=1`** runs the int8/uint8 convs on the exact int8 datapath, and is the
  default in the Frigate `rocket.py` plugin. COCO mAP is CPU-parity, at MobileDet 0.3321
  against 0.3318 [HW sweep]. `mm_int8=1`, the default under it, routes 1×1 convs to the
  resident int8 matmul.
- **Leave `aux_ops=1`**, the default, so the host elementwise, pool and activation ops are
  claimed and fused. The opt-in on-NPU routes (`act_npu`, `pool_npu`, and the rest) are A/B
  knobs rather than throughput wins, because the host kernel is exact and free.
- **Throughput comes from a process pool rather than a faster single stream.** A single
  inference is host cube-gather-bound at ~336 ms warm. Scale with **one delegate process per
  camera**, each pinned to a distinct A76 core via the `ROCKET_CPU_AFFINITY` env var. That is
  3.20 to 9.55 detection_fps at P=1 to 4, a 2.98x on live Frigate. `nthreads`, default 4,
  fans a single conv across the 3 NPU cores. The pool fans across streams, and is the larger
  lever.
- **Keep the big cores off their floor.** The inference blocks while the NPU runs, so a
  load-sampling CPU governor sees idle cores and parks them at `scaling_min_freq`. The host
  cube gather then runs at that floor.

  Measured on one MobileDet run, that is 199.7 ms with the cores pinned against 639.4 ms
  under `ondemand`, where the A76 floor is 408 MHz. It is 1.27x where the floor is 1.2 GHz,
  and plain-CPU TFLite is flat in both. Raise the floor for the cores the detector processes
  run on. `performance` everywhere is not needed, and pinning is required before any
  NPU-versus-CPU number is quoted.

`ROCKET_PROF_POOL=1` is an env knob, off by default and zero cost when unset. It tallies the
host `parallel_oh` fan-out, meaning fan-out calls, threads spawned, and total wall in the
fan-out regions, and prints one `[rocket-pool-probe]` line at exit.

It exists to weigh a would-be persistent worker pool against the measured `std::thread`
create-and-join cost before building one. On SSDLite-MobileDet the spawn and join is ~0.83%
of inference, below the host scatter and gather ceiling, so no pool is warranted. Re-check
with this probe on a different detector before revisiting.

## Host-side work and remaining gaps

### Host requant

Host requant is the largest remaining host cost, and every epilogue is NEON-vectorized.

**The M-major 1×1-to-matmul epilogue** is `rocket_out_mn_to_nhwc_q_per_axis*_band`, in both
the signed/symmetric and the uint8-recenter `β·Sx` variants. A 1×1's output has the
channels contiguous on *both* the int32 read and the NHWC write. It therefore requantizes
8 channels per step with no scatter.

It is bit-identical to the scalar `lrintf` epilogue, because every intermediate fits int32
and `vcvtnq_s32_f32` is its round-nearest-ties-even. It is worth **+5.5 % warm MobileDet and
+9.3 % EfficientDet-Lite0**, and `ROCKET_REQUANT_SCALAR=1` restores the scalar path for A/B
or a non-NEON target.

**The per-output-pixel box-sum** is `rocket_in_window_sum_i8`, the asymmetric-uint8 `β·Sx`
residual. It is likewise NEON-vectorized for the contiguous case, skipped when every
`w_zp==128`, and parallelized across the A76s by output-row band.

For `KW>1` it uses a **separable** decomposition. The `IC·KH·KW` window sum splits into a
channel and row reduce `T[iw]=Σ_{ic,kh} x[ic,oh·sy+kh·dy,iw]`, 16-wide, followed by a
horizontal `KW` window-sum `Sx[oh,ow]=Σ_{kw} T[ow+kw]`, 4-wide. That cuts the SIMD
column-pass count from `IC·KH·KW` to `IC·KH + KW` while keeping both passes vectorized.

Pure integer adds are order-independent, so it is byte-identical to the per-window and
scalar forms. `convert_test` reads `max|dq|=0` under `ROCKET_BOXSUM_MODE=0/1/2`, and the
`boxsum_bench` gate cross-checks all three checksums.

| Kernel | Box-sum speedup over the per-window NEON |
|---|---|
| 3×3 | 2.6x |
| 5×5 | 3.3x |
| 7×7 | 4.3x |

That is ~10-16x over scalar. `ROCKET_BOXSUM_MODE` selects `0` for separable (the default),
`1` for per-window and `2` for scalar, for A/B and as the non-NEON fallback.

Whole-model impact stays sub-1 %. The box-sum is a small slice of the gather-bound total,
and most MobileDet convs route to the 1×1 matmul path.

**The NCHW direct-conv requant** is `rocket_out_nchw_to_nhwc_q_per_axis_band` signed and
`_u8_band` uint8, also NEON-vectorized. It is oc-outer, because the conv accumulator is
NCHW, so the NHWC write `dst[pix·OC+oc]` is strided. The per-pixel compute therefore
vectorizes over 4 contiguous source pixels, and the 4 result bytes scatter out
individually.

It is bit-identical to the scalar epilogue, at the same float order, with `convert_test`
`max|dq|=0` and both detectors byte-identical NEON against scalar.

Per-function microbench is **1.6-3.4x**, and whole-model impact is sub-1 %. Most MobileDet
convs route to the already-vectorized 1×1 path, and EfficientDet has no native-direct convs.
That is consistent with the gather-bound envelope.

The remaining single-stream lever is resident NCHW intermediates rather than the requant.

### Throughput pool for multi-camera

Warm single-stream ~336 ms is not video-rate. Throughput comes from running several
detection contexts concurrently rather than from a faster per-conv. One context's host pack
and readback phase overlaps another's NPU phase.

**The recipe needs no delegate or driver change.** Run one process per stream, exactly how
Frigate runs cameras, and set `ROCKET_CPU_AFFINITY=<one distinct big core>` in each before
the delegate loads, for example `4`, `5`, `6`, `7`. The P contexts then spread across the
four A76 cores instead of colliding on one. The auto-affinity otherwise pins every 1-thread
context's worker to big-core 0.

Measured on the RK1 with `tools/pool_throughput.py`, a 1×1-conv submit-bound unit at
600 MHz:

| Processes | Aggregate inferences/s, relative |
|---:|---:|
| 1 | 1.00x |
| 2 | 2.17x |
| 3 | 3.11x |
| 4 | 3.56x |

That is end to end through the delegate, and it matches the driver gate's pool ceiling
(`rocket-userspace/tests/ctx_pool_throughput.c`).

A rate is not the whole gate. Several contexts share one NPU, so `tools/pool_hash.py` runs
the same fan-out asserting that every process's output hash equals the single-process
reference. It measures 3.51-3.61x at P=4 on SSDLite-MobileDet, one hash throughout.

The application glue ships in `frigate/`: **Frigate's `rocket.py` detector plugin**, one
process per camera at `num_threads: 1` with the per-process affinity env, plus config and
compose.

Live-validated on the RK1 with a pool of `rocket0..rocket3` detectors, each pinned to a
distinct A76, serving four SSD-MobileDet cameras. That is 600 MHz, with NPU IRQs pinned to
the big cores via the driver repo's `npu_set_irq_affinity.sh throughput`:

| Detectors | Aggregate `detection_fps` | Scaling |
|---:|---:|---:|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | 2.98x |

[HW sweep] The four A76 cores carry one detector each at ~73 % busy, while the A55s handle
video decode.

The live pool tracks below the submit-bound delegate ceiling, because a real MobileDet
inference is host cube-scatter-and-gather-bound rather than submit-bound. Per-inference
latency rises from 338 to 424 ms as the four contexts contend for DRAM bandwidth, so
aggregate throughput tapers rather than reaching 4x.

Under that contended pool, the `ROCKET_CONV_BATCH=1` knob coalesces a tiled conv's
per-tile submits into one job. That cuts each process's trips through the shared submit and
IOMMU path. It is **neutral single-stream and on conv-tile-light MobileDet and
EfficientDet, and +7.6 % aggregate only on a conv-tile-heavy unit at P=4**. So leave it off
unless a model is conv-tile-heavy.

Single-stream latency is host cube-scatter-and-gather-bound rather than submit-bound. The
single-stream levers are therefore the NEON requant above and resident NCHW intermediates
(`nchw_resident`), rather than submit-coalescing.

### Op-coverage gaps

Measured against the common detection and vision op set. The high-value layout, vision and
arithmetic ops are claimed. That is `RESIZE_*`, `PRELU`, `MEAN` and reduce, `TRANSPOSE`,
`PAD`, `SLICE`, `STRIDED_SLICE`, `SPLIT`, `MUL`, `SUB`, `DIV`, `TRANSPOSE_CONV`, and the
activation and elementwise batch.

The two-tensor elementwise ops (`ADD`, `MUL`, `SUB`, `DIV`, `MAXIMUM`, `MINIMUM`) accept
**NumPy and TFLite broadcast**: per-channel `[C]`, scalar, and general right-aligned shapes,
which are the SE-block scale and bias-add cases. Each is lowered by materializing every
operand to the output shape (`rocket_broadcast_copy`) before the exact same-shape host
kernel.

`STRIDED_SLICE` handles any **non-zero stride, positive or negative**, the signed strided
gather, with begin, end and shrink_axis masks resolved per TFLite's Start and Stop rules.

Two gaps remain:

- Ellipsis and new-axis masks. They remap begin, end and stride off the 1:1 input-axis
  correspondence.
- Moving the host `ADD`, `POOL`, `CONCAT` and layout kernels onto DPU regcmd, once inter-op
  buffers are NCHW-resident.

### Per-channel depthwise int8

BS_MUL per-OC requant is the remaining native-int8 gap. The mAP gate showed it costs about
nothing, so pursue it only if a future model shows DW-quant drift.
