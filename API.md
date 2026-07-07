# tflite-rocket — API and tuning reference

The full TFLite op mapping, the quantization internals (native int8/uint8, the
`nchw_resident` inter-op-buffer lever), the complete external-delegate option reference,
and the host-side work / remaining-gaps analysis. The [README](README.md) is the guide;
this is the reference.

## TFLite op mapping

| TFLite op | NPU path | status |
|---|---|---|
| `CONV_2D`, 1×1, stride 1 (**pointwise**), matmul-aligned | **the matmul** (`rocket_matmul_fp16_mt`, multicore) | wired |
| `CONV_2D`, KxK / stride / `SAME`\|`VALID` pad / dilation (float) | fp16 conv (`rocket_conv2d_fp16`) | wired |
| `DEPTHWISE_CONV_2D` (depth_multiplier 1, float + int8/uint8) | native depthwise conv (`rocket_conv2d_fp16` `depthwise=1`, G=32) — **HW-validated bit-exact** | wired |
| `CONV_2D`/`DEPTHWISE_CONV_2D`, **signed int8** (`native_int8=1`) | **NATIVE int8** — direct int8×int8→int32 (`rocket_conv2d_int8`, **multicore `_mt`**) + host per-axis requant; per-tensor DW int8-out on-chip requant (`rocket_conv2d_dw_int8`). **Exact int8** — SSD head bit-identical to CPU TFLite | wired (`native_int8`) |
| `CONV_2D` **1×1** int8/uint8 (`native_int8=1`, `mm_int8` default-on) | **resident int8 matmul** (`rocket_matmul_int8_prepacked`, a 1×1 *is* a matmul) — K/N pad %32, M%4 gate; falls back to the conv otherwise. Bit-exact + **nt-deterministic**. +8% warm | wired (`mm_int8`) |
| `CONV_2D` **uint8** DIRECT (`native_int8=1`) | **NATIVE uint8** — recenter to int8 (`x=in_q−128`, `y=w_q−128`), reuse `rocket_conv2d_int8` (**multicore `_mt`**), fold the centering into `eff_bias` + a per-output-pixel box-sum (`rocket_in_window_sum_i8`, **NEON-vectorized**, **separable** for KW>1, skipped when w_zp==128). **Exact uint8** — all 66 MobileDet convs native, scores ~35-40% lower error vs CPU than fp16 | wired (`native_int8`) |
| `CONV_2D`, int8/uint8 quantized (default) | dequant↔fp16 boundary, reuses the fp16 conv (uint8 DEPTHWISE / per-channel DW / asym-pad DW always) | wired |
| `FULLY_CONNECTED` | **host** matmul (`rocket_fc_f`) by default — beats the dispatch-bound NPU for a one-shot FC; **`fc_npu=1`** routes a large GEMM-shaped FC (M%4, K→%32, N→%16 zero-pad) to the resident fp16 matmul (M==1 GEMV stays host) | wired (`fc_npu` opt-in) |
| `HARD_SWISH` / `LOGISTIC` (sigmoid) / `TANH` / `ELU` / `LOG` / `RELU` / `RELU6` / `RELU_N1_TO_1` / `LEAKY_RELU` / `EXP` / `SQRT` / `RSQRT` / `ABS` / `NEG` / `SQUARE` / `FLOOR` (standalone builtins) | **host** kernel (`rocket_unary_f/_q`, float + int8/uint8) — keeps `conv→act→conv` one contiguous partition. The `RELU` family and `NEG`/`SQUARE`/`FLOOR`/`ABS` are exact float arithmetic (byte-identical to CPU TFLite through the full delegate); `EXP`/`SQRT`/`RSQRT`/`LEAKY_RELU` are libm-exact. **`act_npu=1`** routes the curved kinds to the on-NPU DPU LUT (`ELU`→`rocket_elu_fp16`, `LEAKY_RELU`→`rocket_leaky_relu_fp16`; `EXP`/`SQRT`/`RSQRT`/`ABS`/`LOG` to the domain-limited LUT; `RELU`/`RELU6`/`RELU_N1_TO_1`/`NEG`/`SQUARE`/`FLOOR` are host-only, exact). `RELU`/`RELU6`/`RELU_N1_TO_1` are also fused into a preceding `CONV_2D` when the converter folds them. | wired (`aux_ops`) |
| `MAXIMUM` / `MINIMUM` (two-tensor elementwise) | **host** NHWC kernel (`rocket_binary_f/_q`) — keeps partitions contiguous; **`ew_npu=1`** routes float to the on-NPU DPU EW ALU (`rocket_ew_max/min_fp16`, **bit-identical** to host) | wired (`aux_ops`) |
| `MUL` / `SUB` / `DIV` (two-tensor elementwise, same-shape **or broadcast**, fused act) | **host** kernel (`rocket_arith_f/_q`, float + int8/uint8) — siblings of `ADD` (`SUB`/`DIV` are non-commutative: `in0 op in1`); a broadcasting operand (per-channel `[C]`, scalar, general right-aligned) is materialized to the output shape first (`rocket_broadcast_copy`); keeps partitions contiguous | wired (`aux_ops`) |
| `PRELU` (per-channel slope) | **host** kernel (`rocket_prelu_f/_q`, float + int8/uint8) — the alpha constant dequantized to float[C]; **`act_npu=1`** routes float to the NPU (`rocket_prelu_fp16`, ew_max/mul, **bit-identical**) | wired (`aux_ops`) |
| `MEAN` / `REDUCE_MAX` / `REDUCE_MIN` (axes [1,2] = spatial) | **host** kernel (`rocket_reduce_spatial_f/_q`) — GlobalAvg/Max/MinPool; **`pool_npu=1`** routes float to the NPU PPU (`rocket_global_{avg,max,min}pool_fp16`) | wired (`aux_ops`) |
| `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR` (float + int8/uint8) | **host** kernel (`rocket_resize_*`, TFLite-exact align_corners/half_pixel coordinate math); **`resize_npu=1`** routes an integer-factor align_corners=false/half_pixel=false **nearest** resize to the NPU (`rocket_upsample_nearest_fp16` = block replication, **bit-identical**) | wired (`aux_ops`) |
| `L2_NORMALIZATION` (over the channel axis) | **host** kernel (`rocket_l2norm_f/_q`, float + int8/uint8); **`norm_npu=1`** routes float to the NPU (`rocket_l2norm_fp16`) | wired (`aux_ops`) |
| `TRANSPOSE_CONV` (float) | **host** scatter kernel (`rocket_transpose_conv_f`, TFLite pad mapping) — keeps `conv→tconv→conv` contiguous; int8 + the on-NPU route (`rocket_conv_transpose2d_fp16`) are the follow-on | wired (`aux_ops`) |
| `SOFTMAX` (last axis), `LOG_SOFTMAX` (last axis), `CUMSUM` (last axis) | **host** kernel (`rocket_softmax_f/_q` with the `beta` logit scale, `rocket_logsoftmax_f/_q`, `rocket_cumsum_f/_q`, float + int8/uint8) — stable row-max subtraction, keeps partitions contiguous; the NPU routes (`rocket_softmax_fp16` / `rocket_logsoftmax_fp16` / `rocket_cumsum_fp16`) are the follow-on | wired (`aux_ops`) |
| `ADD` (residual), `AVERAGE`/`MAX_POOL_2D`, `CONCATENATION`, `RESHAPE` (float + int8/uint8) | **host** NHWC kernel (`rocket_ops.h`) — keeps partitions contiguous (`AVERAGE`/`MAX_POOL_2D` opt-in `pool_npu`) | wired (`aux_ops`) |
| `TRANSPOSE`, `PAD`/`PADV2`, `SLICE`, `STRIDED_SLICE`, `SPLIT`/`SPLIT_V` (float + int8/uint8) | **host** byte-exact kernel (`rocket_transpose/slice/pad_bytes`, `rocket_ops.h`) — pure layout moves (values pass through, quant scale/zp unchanged), so they keep `conv→layout→conv` ONE partition instead of splitting around a CPU node. Generic rank ≤ 8. `STRIDED_SLICE` handles any **non-zero** per-axis stride — positive OR negative (the signed strided gather `rocket_strided_slice_bytes`, begin[] = the first/HIGH index when stride<0) — with all five masks resolved per TFLite's Start/Stop rules — begin/end/shrink_axis plus **ellipsis_mask** (expands to full-range axes) and **new_axis_mask** (an output-only size-1 dim) via a spec→input-axis expansion that feeds the same byte gather. (The standard TF→TFLite converter lowers ellipsis/new-axis away — into explicit begin/end+masks and a separate `RESHAPE` — so these masks only arrive from hand-authored or non-standard producers; supported defensively.) No NPU route (the NPU has no on-chip layout-conversion engine) | wired (`aux_ops`) |
| everything else | CPU fallback (automatic) | — |

MobileNet/MobileDet are **pointwise-heavy**, so the matmul-aligned 1×1 fast path
offloads a real fraction of the network at multicore matmul speed, while the general
conv path covers the strided / KxK / dilated convs (e.g. the RGB stem). The
mainline-rocket conv path is **fp16** (same `(target<<48)|(value<<16)|reg` encoding +
NC1HWC2 packing the matmul uses), so the general conv is an extension of the matmul's
domain, not a rewrite. Depthwise (the MobileNet workhorse) is **claimed in the
delegate** and runs on the native `DW_EN` path (group G=32, cracked against a Mesa
Teflon regcmd capture); the delegate reorders TFLite's `[1,KH,KW,C]` filter to the
driver's `[C,KH,KW]` (per-channel-axis-3 quant for int8/uint8) and routes to
`rocket_conv2d_fp16` with `depthwise=1`. With pointwise (1×1=matmul), strided/stem
direct conv, depthwise, and the host ADD/POOL/CONCAT seams all claimed, a
MobileNet/SSD backbone is **NPU-resident**.

### The ops around the conv (ADD / POOL / CONCAT / RESHAPE)

A real detector is more than conv: residual `ADD` (MobileNetV2/V3 + FPN), `AVERAGE`/
`MAX_POOL_2D`, the `CONCATENATION` that joins an SSD head's box/class predictors, and
the `RESHAPE` in that head all sit *between* the convs. Left unclaimed, each is a
partition boundary. The delegate claims them (float + int8/uint8) so the partitioner
takes **contiguous subgraphs** — but computes them with thin **host** NHWC kernels
(`rocket_ops.h`), not on the NPU: these are memory-bound elementwise / reduction ops
for which a dedicated regcmd is not warranted, and (since each conv transposes
NHWC↔NCHW itself) claiming them does not save a transpose. The win is fewer/larger
partitions, with **NCHW-resident inter-op buffers** — which would also let conv
skip the per-op transpose — as the documented follow-up. Quant mirrors the conv
boundary: per-tensor input/output scale+zp, requant to the output type; `ADD` keeps
each input's own scale, `CONCATENATION` requantizes each input to the output scale,
pooling dequant→reduce→requant. Gated by `aux_ops` (default on) for HW A/B. Broadcasting
two-tensor elementwise (per-channel `[C]`, scalar, general right-aligned) IS claimed — each
operand is materialized to the output shape (`rocket_broadcast_copy`) and then the same-shape
host kernel runs (bit-exact); non-`SAME`/`VALID` padding stays on CPU. Each op is validated
off-hardware by `convert_test` against an independent NHWC oracle, the same discipline as the
conv glue.

## Quantization

Detection models are typically **UINT8/INT8** quantized TFLite. The NPU path is
fp16, so the delegate **dequantizes int8→fp16 at the partition boundary** (filter once at
`Prepare`, activations per inference), runs fp16 on the NPU, requantizes the output — the
same trick as ggml's f32↔fp16. The per-inference input dequant maps each stored byte through a
256-entry fp16 lookup table (the per-tensor scale/zero-point baked in once per call), bit-identical
to the per-element arithmetic and ~1.2× faster on the A76 for feature-map-shaped inputs. This is **wired**: the gate accepts int8/uint8 `CONV_2D`,
honours per-axis filter scales + zero points (int32 bias dequantized with `in_scale *
w_scale[oc]`), and the requant epilogue clamps to the output type. Being fp16, it is an
arithmetic approximation of TFLite's int32-accumulate int8 kernel (not bit-identical to
it), so `convert_test`'s quantized shapes validate the glue against an independent oracle
that performs the *same* dequant→fp16-conv→requant path.

A true native int8 conv on the NPU (its own regcmd encoding) removes both the
fp16 approximation *and* the host dequant/requant round-trip: the same perf direction as the
`nchw_resident` path but with **exact** int8 semantics (default-safe by construction).
This runs end-to-end and is HW-validated behind the default-off `native_int8` option:
- **DIRECT / 1×1** = int8×int8→int32 raw (`gen_conv2d_int8` → `rocket_conv2d_int8` runtime
  with OC/IC pad + OC/OH/OW tiling) + host per-axis requant (`rocket_out_nchw_to_nhwc_q_per_axis`,
  with the input zero-point correction folded into the bias). **Real int8 accumulate → an SSD
  head is BIT-IDENTICAL to CPU TFLite** (`max|delegate−CPU|=0`); a MobileNetV2 block's mean
  deviation halves.
- **DEPTHWISE** = int8-OUT with **on-chip requant** (`gen_conv2d_dw_int8` + `int8_out=1` →
  `rocket_conv2d_dw_int8` runtime: uint8-centered cubes + the Mesa zero-point bias fold) —
  **per-tensor** only, **bit-exact vs Teflon ground truth** (`replay_dw_mesa` + `conv_dw_int8_runtime`;
  delegate `max|delegate−Teflon|=0`).
- **UINT8 DIRECT** = the same `rocket_conv2d_int8` runtime fed RE-CENTERED bytes
  (`x=in_q−128`, `y=w_q−128`, both always in [−128,127]); the per-OC centering constants
  (`α·Wy + N·α·β`, α=128−in_zp, β=128−w_zp) fold into `eff_bias`, and the asymmetric weight zp's
  residual is a per-output-pixel **box-sum** `β·Sx` added at requant (`rocket_in_window_sum_i8` /
  `rocket_out_nchw_to_nhwc_q_per_axis_u8`, **skipped when w_zp==128**). NO new regcmd. The decisive
  fact: real MobileDet is uint8 per-tensor with w_zp spread 91..177 (almost never 128), so the
  symmetric-only trick fails — the box-sum is what makes it exact. **All 66 MobileDet CONV_2D run
  native uint8** (`[native u8]`); scores ~35-40% lower error vs CPU than fp16, several bit-identical;
  8/8 `convert_test` nu8 shapes `max|dq|=0` on the NPU.

**Run it with `--option native_int8=1`.** Signed-int8 symmetric-weight convs AND uint8 (any
weight zp) DIRECT convs take the native path; **uint8 DEPTHWISE, per-channel depthwise, and
asymmetric-pad depthwise stay on the dequant↔fp16 boundary** (per-channel DW on-chip requant needs
the BS_MUL per-OC multiply path; uint8 DW native would match Teflon ≠ CPU TFLite — a perf play with
an accuracy caveat, not wired). The oracle subtlety: DIRECT int32-raw (int8 + uint8) is real
accumulate so it matches CPU TFLite, while DW int8-OUT replicates Teflon's on-chip requant so its
oracle is Teflon (which itself differs from CPU TFLite by ≤143).

### fp16-NCHW resident inter-op buffers (`nchw_resident`, opt-in)

A perf lever on the int8 conv chain: a partition-internal tensor
produced and consumed only by general-conv-path ops is kept in **fp16-NCHW** (post
bias+act) between the ops, skipping the NHWC↔NCHW transpose **and** the int8 requant→dequant
round-trip at that boundary. Win: a MobileNetV2 block warm `12.66→9.34 ms` (1.36×), SSD head
`2.32→1.96 ms`. It is **off by default** (`nchw_resident=1` opts in) because it changes int8
numerics — skipping the requant lets a tensor carry sub-int8 drift. Two guards/findings:

- **Re-quantization-barrier rule** (default-on when `nchw_resident=1`): a tensor stays
  resident only if no consuming conv reaches a partition output through transparent ops
  (RESHAPE/CONCAT) with no requant barrier (another conv / ADD / POOL) in between — so the
  last conv before an output always reads requantized input. Keeps damped chains resident
  (a block's project→ADD), drops the SSD predictors' direct inputs.
- **Precision vs. the int8 reference.** The barrier rule alone does *not* keep the delegate
  bit-close to the int8 *reference* in deep graphs (high-gain int8 convs compound the drift:
  SSD `max|delegate−CPU|` 52, stack 63). But measured against the **fp32 ground truth** (a
  paired float+int8 model, `tools/precision_probe.py`), `nchw_resident=1` is neutral on max
  (the shared final output quant dominates) and **better on the mean** — it moves *toward*
  fp32. The big delta-vs-int8 is distance from the lossy int8 reference, not error. This is
  why `nchw_resident` stays opt-in: native int8 is the path that gets both the perf and
  exact semantics.

## Delegate options

The delegate reads external-delegate options: `native_int8` (run int8/uint8 convs on the
real int8 datapath instead of the dequant↔fp16 boundary; **default 0**, opt-in — exact
int8/uint8 semantics), `mm_int8` (route 1×1 int8/uint8 convs to the resident int8 matmul;
**default 1**, only active under `native_int8`), `nthreads` (worker fan-out across the 3
NPU cores, default 4 — sizes **both** the 1×1 matmul fan-out **and** the native
int8/uint8 DIRECT conv worker pool `rocket_conv_pool`), `min_macs` (minimum
`OC*OH*OW*IC*KH*KW` to offload; default 0 = every supported conv), `aux_ops` (claim the
host elementwise / pooling / activation / shape ops — `ADD`/`MAXIMUM`/`MINIMUM`/pool/
concat/reshape/activations (incl. the `RELU` family / `LEAKY_RELU` / `EXP` / `SQRT` /
`RSQRT` / `ABS` / `NEG` / `SQUARE` / `FLOOR`)/`PRELU`/`MEAN`/`REDUCE_*`/`RESIZE_*`/
`L2_NORMALIZATION`/`TRANSPOSE_CONV`/`SOFTMAX`/`LOG_SOFTMAX`/`CUMSUM`; default 1 — set 0 to restrict the delegate to
conv only). The opt-in **on-NPU routes** for the claimed ops (all default 0; the host
kernel is exact + free, the NPU route is an extra round-trip kept for A/B + cube-resident
fusion): `act_npu` (DPU LUT activations + `PRELU`), `ew_npu` (`MAXIMUM`/`MINIMUM` on the
DPU EW ALU), `pool_npu` (`AVERAGE`/`MAX_POOL_2D` + `MEAN`/`REDUCE_*` on the PPU),
`resize_npu` (integer-factor nearest resize), `norm_npu` (`L2_NORMALIZATION`), `fc_npu`
(`FULLY_CONNECTED`). `profile` (per-op timing to stderr, incl. the
`breakdown in/conv/out` sub-step split and the resident set), `nchw_resident`
(keep conv→conv intermediates in fp16-NCHW between ops, skipping the per-boundary
transpose + int8 requant/dequant; **default 0**, opt-in — see Quantization). The driver
dependency resolves via
`find_package(rocketnpu)` if installed, else from a sibling `rocket-userspace` checkout
(override with `-DROCKETNPU_DIR=/path/to/rocket-userspace`).

`ROCKET_PROF_POOL=1` (env, off by default, zero cost when unset) tallies the host `parallel_oh`
fan-out — fan-out calls, threads spawned, and total wall in the fan-out regions — and prints one
`[rocket-pool-probe]` line at exit. It exists to weigh a would-be persistent worker pool against the
measured `std::thread` create+join cost before building one: on SSDLite-MobileDet the spawn/join is
~0.83% of inference (below the host-scatter/gather ceiling), so no pool is warranted — re-check with
this probe on a different detector before revisiting.

## Host-side work and remaining gaps

- **Host requant** is the largest remaining host cost, and is fully NEON-vectorized.
  The **M-major 1×1→matmul requant epilogue** (`rocket_out_mn_to_nhwc_q_per_axis*_band`, both the
  signed/symmetric and uint8-recenter `β·Sx` variants) is **NEON-vectorized**: a 1×1's output has
  the channels contiguous on *both* the int32 read and the NHWC write, so it requantizes 8 channels
  per step with no scatter. Bit-identical to the scalar `lrintf` epilogue — every intermediate fits
  int32 and `vcvtnq_s32_f32` is its round-nearest-ties-even — and worth **+5.5 % warm MobileDet /
  +9.3 % EfficientDet-Lite0** (`ROCKET_REQUANT_SCALAR=1` restores the scalar path for A/B or a
  non-NEON target). The per-output-pixel box-sum (`rocket_in_window_sum_i8`, the asymmetric-uint8
  `β·Sx` residual) is likewise **NEON-vectorized** for the contiguous case (skipped when every
  `w_zp==128`, parallelized across the A76s by output-row band), and for `KW>1` uses a **separable**
  decomposition: the `IC·KH·KW` window sum splits into a channel+row reduce
  `T[iw]=Σ_{ic,kh} x[ic,oh·sy+kh·dy,iw]` (16-wide) followed by a horizontal `KW` window-sum
  `Sx[oh,ow]=Σ_{kw} T[ow+kw]` (4-wide), cutting the SIMD column-pass count from `IC·KH·KW` to
  `IC·KH + KW` while keeping both passes vectorized. Pure integer adds (order-independent), so it is
  byte-identical to the per-window and scalar forms (`convert_test max|dq|=0` under
  `ROCKET_BOXSUM_MODE=0/1/2`; the `boxsum_bench` gate cross-checks all three checksums). Box-sum
  microbench **2.6× (3×3) / 3.3× (5×5) / 4.3× (7×7) over the per-window NEON** (~10–16× over scalar);
  `ROCKET_BOXSUM_MODE` selects `0`=separable (default) / `1`=per-window / `2`=scalar for A/B and as the
  non-NEON fallback. Whole-model impact stays sub-1 % (the box-sum is a small slice of the gather-bound
  total; most MobileDet convs route to the 1×1 matmul path). The
  **NCHW direct-conv requant** (`rocket_out_nchw_to_nhwc_q_per_axis_band` signed + `_u8_band` uint8)
  is also **NEON-vectorized**: it is oc-outer (the conv accumulator is NCHW) so the NHWC write
  `dst[pix·OC+oc]` is strided — the per-pixel compute vectorizes over 4 contiguous source pixels and
  the 4 result bytes scatter out individually. Bit-identical to the scalar epilogue (same float order;
  convert_test `max|dq|=0` and both detectors byte-identical NEON vs scalar). Per-function microbench
  **1.6–3.4×**, but whole-model impact is sub-1 % (most MobileDet convs route to the already-vectorized
  1×1 path and EfficientDet has no native-direct convs), consistent with the gather-bound envelope.
  Every requant epilogue is NEON-vectorized; the remaining single-stream lever is resident NCHW intermediates,
  not the requant.
- **Throughput pool for multi-camera** — warm single-stream ~336 ms is not video-rate;
  throughput comes from running several detection contexts concurrently (one context's host
  pack/readback phase overlaps another's NPU phase), not a faster per-conv. **The recipe needs
  no delegate/driver change** — run one PROCESS per stream (exactly how Frigate runs cameras)
  and set `ROCKET_CPU_AFFINITY=<one distinct big core>` (e.g. `4`,`5`,`6`,`7`) in each before
  the delegate loads, so the P contexts spread across the four A76 cores instead of colliding on
  one (the auto-affinity pins every 1-thread context's worker to big-core 0). Measured on the RK1
  (`tools/pool_throughput.py`, a 1×1-conv submit-bound unit, 600 MHz): **P=1→4 = 1.00× / 2.17× /
  3.11× / 3.56×** aggregate inferences/s — end-to-end through the delegate, matching the driver
  gate's pool ceiling (`rocket-userspace/tests/ctx_pool_throughput.c`). The application glue ships in
  `frigate/`: **Frigate's `rocket.py` detector plugin** (one process per camera, `num_threads: 1`, the
  per-process affinity env) plus config and compose. Live-validated on the RK1 with a pool of
  `rocket0..rocket3` detectors, each pinned to a distinct A76, serving four SSD-MobileDet cameras
  (600 MHz, NPU IRQs pinned to the big cores via the driver repo's `npu_set_irq_affinity.sh throughput`):
  aggregate `detection_fps` **P=1→4 = 3.20 / 6.00 / 8.00 / 9.55 = 1.00× / 1.88× / 2.50× / 2.98×**
  [HW sweep], the four A76 cores carrying one detector each (~73 % busy) while the A55s handle video
  decode. The live pool tracks below the submit-bound delegate ceiling because a real MobileDet
  inference is host cube-scatter/gather-bound, not submit-bound: per-inference latency rises 338→424 ms
  as the four contexts contend for DRAM bandwidth, so aggregate throughput tapers rather than reaching
  4×. Under that contended pool the
  `ROCKET_CONV_BATCH=1` knob (coalescing a tiled conv's per-tile submits into one job) cuts each
  process's trips through the shared submit/IOMMU path — **neutral single-stream and on conv-tile-light
  MobileDet/EfficientDet, +7.6 % aggregate only on a conv-tile-heavy unit at P=4**, so leave it off
  unless a model is conv-tile-heavy. Single-stream latency is host cube-scatter/gather-bound, not
  submit-bound, so the single-stream levers are the NEON requant above and resident NCHW
  intermediates (`nchw_resident`), not submit-coalescing.
- **Op-coverage gaps** (measured against the common detection/vision op set). The high-value layout, vision,
  and arithmetic ops are claimed (`RESIZE_*`, `PRELU`, `MEAN`/reduce, `TRANSPOSE`/`PAD`/
  `SLICE`/`STRIDED_SLICE`/`SPLIT`, `MUL`/`SUB`/`DIV`, `TRANSPOSE_CONV`, the activation/
  elementwise batch). The two-tensor elementwise ops (`ADD`/`MUL`/`SUB`/`DIV`/`MAXIMUM`/
  `MINIMUM`) accept **NumPy/TFLite broadcast** (per-channel `[C]`, scalar, and general
  right-aligned shapes — the SE-block scale and bias-add cases), lowered by materializing each
  operand to the output shape (`rocket_broadcast_copy`) before the exact same-shape host kernel;
  and `STRIDED_SLICE` handles any **non-zero stride — positive OR negative** (the signed strided
  gather, with begin/end/shrink_axis masks resolved per TFLite's Start/Stop rules). Remaining:
  ellipsis / new-axis masks (they remap begin/end/stride off the 1:1 input-axis correspondence),
  and moving the host `ADD`/`POOL`/`CONCAT`/layout kernels onto DPU regcmd once inter-op buffers
  are NCHW-resident.
- **Per-channel depthwise int8** (BS_MUL per-OC requant) — the remaining native-int8 gap.
  The mAP gate showed it costs ~0, so pursue it only if a future model shows DW-quant drift.
</content>
