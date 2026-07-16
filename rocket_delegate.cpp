// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
// ===========================================================================
// File overview and includes
// ===========================================================================
/*
 * rocket_delegate.cpp — TFLite external delegate for the RK3588 NPU (rocket).
 *
 * Runs float32 CONV_2D and DEPTHWISE_CONV_2D on the NPU. General KxK / stride / pad
 * (SAME|VALID) / dilation go through the HW-validated rocket_conv2d_fp16; a
 * matmul-aligned 1x1 pointwise conv takes a fast path onto the multicore fp16 matmul
 * (a 1x1 conv IS a matmul: out[m,cout] = sum_cin in[m,cin]*w[cout,cin], over
 * m = H*W positions). Depthwise (depth_multiplier 1 => OC==IC, one KH×KW filter per
 * channel) sets desc.depthwise=1 and the driver runs its native depthwise job — the
 * SAME pointwise(1x1)+depthwise(KxK) decomposition a MobileNet/SSD backbone is built
 * from. Filters are converted + reordered into the driver's NCHW fp16 layout ONCE in
 * Prepare; per inference only the activation is transposed in (with the SAME/VALID
 * padding materialized) and the result transposed out (+ bias + fused activation).
 *
 * int8/uint8 CONV_2D has two paths. By DEFAULT the delegate dequantizes at the
 * partition boundary (filter once in Prepare, activation per inference), reuses the
 * SAME fp16 conv, then requantizes the output — an fp16 approximation of TFLite's
 * int8 kernel. With the `native_int8` option a NATIVE int8/uint8 conv path
 * runs int8 directly on the NPU with no host dequant (direct + 1x1, plus per-tensor
 * symmetric depthwise via the int8-out on-chip-requant runtime); per-channel DW int8
 * and uint8 depthwise fall back to the fp16 approximation. See rocket_convert.h.
 *
 * Beyond conv, the delegate also claims the ops a real detector graph carries AROUND
 * its convolutions so the partitioner can take CONTIGUOUS subgraphs instead of one
 * partition per conv: ADD (residual), AVERAGE/MAX_POOL_2D, CONCATENATION (SSD head
 * joins), and shape-only RESHAPE. These run as thin HOST kernels (float + int8/uint8)
 * inside the delegate — memory-bound elementwise/reduction work for which inventing
 * NPU regcmd is not worth it in a v1; see rocket_ops.h. They do NOT move work to the
 * NPU (and since each conv transposes NHWC<->NCHW itself, they don't save a transpose)
 * — the win is fewer/larger partitions, with NCHW-resident inter-op buffers as a
 * follow-up. They are gated by the `aux_ops` option (default on) for HW A/B.
 *
 * Everything this delegate does not claim falls back to CPU via TFLite's graph
 * partitioner: depthwise with depth_multiplier != 1, a depthwise layer too big for one
 * CBUF pass (the driver has no DW spatial tiling yet), grouped convs, broadcast ADD,
 * and fused activations beyond None/Relu/Relu6/ReluN1To1.
 *
 * Builds to libtflite_rocket.so, loadable by tflite_runtime:
 *   tflite.load_delegate("libtflite_rocket.so",
 *       options={"nthreads": "4", "min_macs": "0", "aux_ops": "1", "profile": "0"})
 */
// TFLite C API only (no C++ TFLite lib). These headers ship with Mesa's Teflon build
// (tensorflow/lite/core/c/...); the delegate is a classic C TfLiteDelegate, so the few
// C symbols it references (TfLiteIntArrayCreate/Free) bind at dlopen from the host
// interpreter — exactly like Mesa's libteflon.so. We deliberately do NOT use the C++
// SimpleDelegate helper (its header/lib aren't shipped without a full TFLite build).
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>
#include <climits>

extern "C" {
#include "rocket_npu.h"       // rocket_open / rocket_close
#include "rocket_matmul.h"    // rocket_ctx / rocket_weights / prepacked matmul (1x1 path)
#include "rocket_conv.h"      // rocket_conv2d_fp16 / _plan / _oh / _ow
#include "rocket_pool.h"      // rocket_pool_fp16 (opt-in on-NPU PPU MaxPool/AvgPool)
#include "rocket_activation.h"// rocket_activation_fp16 (opt-in on-NPU DPU LUT activation)
#include "rocket_reduce.h"    // rocket_global_{avg,max,min}pool_fp16 (opt-in spatial reduce)
#include "rocket_resize.h"    // rocket_upsample_nearest_fp16 (opt-in integer-factor resize)
#include "rocket_normvision.h"// rocket_l2norm_fp16 (opt-in on-NPU L2 normalization)
// Pin a worker thread to a big (A76) core. Exported by the driver lib (rocket_affinity.c)
// but not in an installed header; the host-side box-sum/requant fan-out reuses it so the
// threads land on the fast cores (idx % n_big internally; a no-op if pinning is disabled).
void rocket_pin_worker(int worker_idx);
}
#include "rocket_convert.h"   // NHWC<->NCHW + SAME/VALID pad + bias/act glue
#include "rocket_ops.h"       // host ADD / POOL / CONCAT NHWC kernels (float + quant)

namespace {

// ===========================================================================
// Options and helper utilities (quantization, layout, shape math)
// ===========================================================================

// ---------------------------------------------------------------------------
// options (parsed from the external-delegate key/value pairs)
// ---------------------------------------------------------------------------
struct RocketOptions {
    int  nthreads = 4;   // matmul (1x1) fan-out across the 3 NPU cores (clamped 1..8)
    long min_macs = 0;   // min OC*OH*OW*IC*KH*KW to offload (0 = every supported conv)
    bool aux_ops  = true;// claim ADD/POOL/CONCAT/RESHAPE host ops (contiguous partitions)
    bool profile  = false;
    bool native_int8 = false;   // run signed-int8 DIRECT/1x1 convs as a NATIVE int8
                                // conv (int8 x int8 -> int32 on the NPU, host per-axis
                                // requant) instead of the dequant->fp16->requant approx.
                                // EXACT int8 accumulate (bit-identical to TFLite's int8
                                // CPU kernel up to the <=1 requant rounding), no host
                                // dequant/requant round-trip. OFF by default for an A/B;
                                // uint8 + depthwise use the fp16 path.
    bool mm_int8 = true;        // perf Step 1: route native int8/uint8 1x1 DIRECT convs to the
                                // RESIDENT multicore int8 matmul (rocket_matmul_int8_prepacked)
                                // instead of the single-core conv pool. A 1x1 IS a matmul; the
                                // NHWC in/out need no transpose and the integer accumulate is
                                // exact, so the result is BYTE-IDENTICAL to the conv path (a kill
                                // switch / A-B toggle — only takes effect under native_int8). K/N
                                // are zero-padded to %32; M%4||M==1 or it stays on the conv path.
    bool act_npu = false;       // run the claimed unary activations (HardSwish / Sigmoid) on the
                                // NPU DPU LUT (rocket_activation_fp16) instead of the exact host
                                // kernel. OFF by default: the host kernel is exact + free (a
                                // standalone NPU LUT pass is an extra round-trip), so this is an
                                // A/B / demonstration toggle for the on-NPU nonlinearity. Falls
                                // back to the host kernel with no device or on a LUT failure.
    bool fc_npu = false;        // run a claimed FULLY_CONNECTED on the NPU resident matmul
                                // instead of the host matmul. OFF: the host matmul beats the
                                // dispatch-bound NPU for one-shot TFLite FC (not-mac-bound);
                                // on, for large batched / repeated FC (M%4; M==1 stays host).
    bool ew_npu = false;        // run a claimed MAXIMUM / MINIMUM on the NPU DPU EW ALU
                                // (rocket_ew_max/min_fp16) instead of the exact host kernel.
                                // OFF: like ADD these are memory-bound, so the host kernel is
                                // exact + free and a standalone NPU EW pass is an extra
                                // round-trip; on, for A/B / cube-resident fusion. FLOAT only;
                                // falls back to the host kernel with no device or on failure.
    bool norm_npu = false;      // run a claimed L2_NORMALIZATION on the NPU (rocket_l2norm_fp16)
                                // instead of the exact host kernel. OFF: the host kernel is exact
                                // + free (a standalone NPU norm is an extra round-trip). FLOAT
                                // only; falls back to host with no device or on failure.
    bool resize_npu = false;    // run a claimed RESIZE_NEAREST_NEIGHBOR on the NPU
                                // (rocket_upsample_nearest_fp16) instead of the host kernel.
                                // OFF: resize is a memory-bound gather (host is exact + free).
                                // Only an INTEGER-factor, align_corners=false, half_pixel=false
                                // nearest resize (floor mode = the NPU's block replication) is
                                // routed — anything else (bilinear, non-integer, other modes)
                                // stays on the exact host kernel. Falls back to host on failure.
    bool pool_npu = false;      // run a claimed MAX_POOL_2D / AVERAGE_POOL_2D on the NPU PPU
                                // (rocket_pool_fp16) instead of the exact host kernel. OFF: the
                                // host kernel is exact + free (a standalone PPU pool is a 2nd
                                // round-trip + NHWC<->cube transpose until partitions stay
                                // cube-resident). FLOAT only; AVERAGE only when VALID (the PPU
                                // divides by KH*KW = count-include-pad, vs TFLite's valid count,
                                // so a padded average diverges — MAX with pad is fine). Falls
                                // back to the host kernel with no device or on a PPU failure.
    bool nchw_resident = false; // keep conv->conv intermediates in fp16-NCHW (skips the
                                // per-boundary transpose + int8 requant/dequant). Big win
                                // (single block 1.37x), but OFF by default: the skipped int8
                                // requant makes the activation diverge from the int8 reference,
                                // UNBOUNDEDLY when a resident value reaches an output predictor
                                // undamped (SSD/stack max|delegate-CPU| 2 -> 65). Needs a
                                // re-quantization-barrier rule (or mAP validation) before it can
                                // default on. nchw_resident=1 opts in (validated on chains that
                                // re-quantize before the output, e.g. a MobileNetV2 block).
};

// A 1x1 stride-1 conv we can hand to the matmul: just the matmul's HARD alignment
// (K%32, N%16, M%4||M==1; M = H*W positions, K = IC, N = OC). The old "worth
// offloading" size floor (K,N>=64, M>=4) is gone: with the weights resident (packed
// once in Prepare, only the activation packed per call) even a small aligned 1x1 is
// cheaper on the matmul than re-running the 5-BO conv job, so every aligned 1x1
// offloads. Non-aligned 1x1s (e.g. IC%32!=0) still take the conv path.
static bool dims_offloadable(int M, int K, int Ncout) {
    return (K % 32 == 0) && (Ncout % 16 == 0) && (M % 4 == 0 || M == 1) && M >= 1;
}

static int round_up(int x, int m) { return (x + m - 1) / m * m; }

static int map_activation(TfLiteFusedActivation a) {
    switch (a) {
    case kTfLiteActNone:      return ROCKET_ACT_NONE;
    case kTfLiteActRelu:      return ROCKET_ACT_RELU;
    case kTfLiteActRelu6:     return ROCKET_ACT_RELU6;
    case kTfLiteActReluN1To1: return ROCKET_ACT_RELUN1;
    default:                  return -1;   // tanh/sigmoid/etc. -> leave the node on CPU
    }
}

static bool is_quant_type(TfLiteType t) { return t == kTfLiteInt8 || t == kTfLiteUInt8; }

// A byte-preserving layout op (reshape/transpose/slice/pad) moves quantized bytes
// unchanged, so it is only correct when input and output carry the SAME affine quant
// params. Reject mismatched (or differently-typed) quant metadata so a malformed graph
// falls back to the CPU instead of silently emitting numerically-wrong bytes. Float
// tensors have no quant params -> trivially equal.
static bool same_quant_params(const TfLiteTensor &a, const TfLiteTensor &b) {
    if (a.type != b.type) return false;
    if (!is_quant_type(a.type)) return true;
    return a.params.scale == b.params.scale && a.params.zero_point == b.params.zero_point;
}

// Bytes per element for the data-path types the delegate handles (the layout ops are
// byte-exact over any of them); 0 for an unsupported type (so the op declines to CPU).
// Matches RESHAPE: float32 and int8/uint8 only — int32 etc. stay on the CPU.
static int type_elem_size(TfLiteType t) {
    switch (t) {
    case kTfLiteFloat32:                 return 4;
    case kTfLiteInt8: case kTfLiteUInt8: return 1;
    default:                             return 0;
    }
}

// Read element i of a constant int32/int64 index tensor (TFLite begin/size/axis come
// as either). Returns 0 for an unsupported type — callers validate the type first.
static long read_const_int(const TfLiteTensor &t, int i) {
    if (t.type == kTfLiteInt32) return (long)t.data.i32[i];
    if (t.type == kTfLiteInt64) return (long)t.data.i64[i];
    return 0;
}

// Map a delegate unary-activation kind (ROCKET_UNARY_*) to the driver's on-NPU DPU
// LUT kind (ROCKET_ACTIVATION_*); -1 if no LUT path exists. Used only by the opt-in
// act_npu route.
static int unary_to_rocket_act(int kind) {
    switch (kind) {
    case ROCKET_UNARY_HARDSWISH:   return ROCKET_ACTIVATION_HARDSWISH;
    case ROCKET_UNARY_SIGMOID:     return ROCKET_ACTIVATION_SIGMOID;
    case ROCKET_UNARY_HARDSIGMOID: return ROCKET_ACTIVATION_HARDSIGMOID;
    case ROCKET_UNARY_TANH:        return ROCKET_ACTIVATION_TANH;
    // ELU has a dedicated driver entry point (rocket_elu_fp16, with the x~=0
    // signed-output host repair), NOT the generic rocket_activation_fp16 LUT call;
    // eval_act special-cases it. Return its enum so callers can name it.
    case ROCKET_UNARY_ELU:         return ROCKET_ACTIVATION_ELU;
    // LOG rides the generic LUT call, but the LUT is domain-limited ([~0.25,32]); the
    // host kernel (exact logf) is the default, act_npu is the approximate demo route.
    case ROCKET_UNARY_LOG:         return ROCKET_ACTIVATION_LOG;
    // LEAKY_RELU has a dedicated parametric driver entry point (rocket_leaky_relu_fp16,
    // per-tensor alpha); eval_act special-cases it like ELU.
    case ROCKET_UNARY_LEAKY_RELU:  return ROCKET_ACTIVATION_LEAKY_RELU;
    // The positive-/symmetric-domain LUT kinds. Domain-limited (like LOG), so the exact
    // host kernel is the default and act_npu is the approximate demo route.
    case ROCKET_UNARY_EXP:         return ROCKET_ACTIVATION_EXP;
    case ROCKET_UNARY_SQRT:        return ROCKET_ACTIVATION_SQRT;
    case ROCKET_UNARY_RSQRT:       return ROCKET_ACTIVATION_RSQRT;
    case ROCKET_UNARY_ABS:         return ROCKET_ACTIVATION_ABS;
    // RELU/RELU6/RELU_N1_1/NEG/SQUARE/FLOOR are exact host arithmetic with no LUT
    // (RELU is max(x,0), not a curve); host-only, no NPU route.
    default:                       return -1;
    }
}

// Product of tensor extents with overflow + negative-extent rejection. Returns the
// element count, or -1 if any extent is negative or the running product overflows.
// Model-supplied dims are untrusted: a negative extent or a wrapping product would
// otherwise size an allocation to a near-SIZE_MAX or small-then-overrun buffer.
static long long safe_elem_count(const TfLiteIntArray *d) {
    if (!d) return -1;
    long long n = 1;
    for (int i = 0; i < d->size; i++) {
        const long long e = d->data[i];
        if (e < 0) return -1;
        if (e != 0 && n > LLONG_MAX / e) return -1;   // multiply would overflow
        n *= e;
    }
    return n;
}

// Total element count of a dims array (product of extents; batch included). Returns
// -1 on a negative/overflowing product so downstream sizing/index math fails the
// validation checks instead of wrapping.
static long dims_count(const TfLiteIntArray *d) {
    const long long n = safe_elem_count(d);
    return (n < 0 || n > LONG_MAX) ? -1 : (long)n;
}

// Identical shape (rank + every extent) — the no-broadcast elementwise gate.
static bool same_dims(const TfLiteIntArray *a, const TfLiteIntArray *b) {
    if (!a || !b || a->size != b->size) return false;
    for (int i = 0; i < a->size; i++) if (a->data[i] != b->data[i]) return false;
    return true;
}

// True if `in` broadcasts to `out` (NumPy / TFLite right-aligned: in_rank <= out_rank,
// and aligning the trailing axes, each in dim is 1 or == the out dim). The broadcast
// elementwise gate — analyze_add/binary/arith claim a node when BOTH inputs broadcast to
// the output shape; eval_add then materializes each to the output shape (rocket_broadcast_copy)
// and runs the existing same-shape host kernel, so the result is bit-exact.
static bool broadcast_to(const TfLiteIntArray *in, const TfLiteIntArray *out) {
    if (!in || !out || in->size > out->size) return false;
    int off = out->size - in->size;
    for (int i = 0; i < in->size; i++)
        if (in->data[i] != 1 && in->data[i] != out->data[off + i]) return false;
    return true;
}

// Identical shape ignoring one axis (rank + every non-axis extent) — the concat gate:
// inputs may differ only along the concatenation axis, and the copy applies the
// output-derived outer/inner strides to each input, so the non-axis extents must match.
static bool same_dims_except(const TfLiteIntArray *a, const TfLiteIntArray *b, int axis) {
    if (!a || !b || a->size != b->size) return false;
    for (int i = 0; i < a->size; i++)
        if (i != axis && a->data[i] != b->data[i]) return false;
    return true;
}

// ===========================================================================
// Node support and graph partitioning
//
// The claimed-op structs (ConvNode / FCNode / the aux-op params), the per-op
// analyze_* gates that decide which nodes the delegate claims and capture their
// static parameters, and analyze_node, the single dispatch the partitioner and
// the kernel's Init both call.
// ===========================================================================

// One claimed CONV_2D node. Filter-derived dims (IC/OC/KH/KW) + stride/dilation/act
// are static and captured at Init; the spatial dims and padding are recomputed per
// Eval from the live tensors (so a correct result survives an input resize, even if
// the matmul-vs-conv routing was decided for the Init-time shape).
struct ConvNode {
    int in_idx, w_idx, bias_idx, out_idx;
    int IC, OC, KH, KW;
    int sy, sx, dy, dx;
    int act;                     // ROCKET_ACT_*
    bool depthwise = false;      // DEPTHWISE_CONV_2D (OC==IC, one KH×KW filter/channel)
    std::vector<_Float16> w;     // direct [OC][IC][KH][KW] / depthwise [C][KH][KW] fp16
    bool packed = false;

    // resident matmul (1x1 pointwise) path: the fp16 weight w == the matmul B[OC,IC],
    // packed ONCE into NPU BOs in Prepare and kept resident on the kernel's mm_ctx_, so
    // Eval only packs the activation per inference (vs rocket_matmul_fp16_mt opening fds
    // + re-scattering B every call). mm_w is null when the conv is not matmul-aligned, the
    // device/scratch-cache is unavailable, or the input M changed (resize) -> conv path.
    rocket_weights *mm_w = nullptr;
    int            mm_M  = 0;     // spatial M (=IH*IW) the resident weights were packed for

    // resident int8 matmul (native int8/uint8 1x1 pointwise) path: the recentered/raw int8
    // weight is zero-padded to [Np,Kp] (Np=ceil(OC,32), Kp=ceil(IC,32)) and scattered into
    // resident NPU BOs ONCE on the kernel's mm_i8_ctx_; Eval then runs the int8 matmul
    // (output columns fanned across the 3 cores) + a per-pixel requant. mm_w_i8 is null when
    // the conv is not native int8/uint8, M%4||M==1 fails, mm_int8 is off, or no device.
    rocket_i8_weights *mm_w_i8 = nullptr;
    int                mm_i8_M = 0;     // spatial M the resident int8 weights were packed for

    // quantized (int8/uint8) path — dequant at the boundary, reuse the fp16 conv
    bool is_quant = false;
    int  in_uns = 0, w_uns = 0, out_uns = 0;   // 1 = uint8 storage for that tensor
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
    std::vector<float> w_scale;  // [OC] per-axis filter scale
    std::vector<int>   w_zp;     // [OC] per-axis filter zero point
    std::vector<float> bias_f;   // [OC] dequantized bias (empty when no bias)

    // NATIVE int8 path (native_int8 option). Set for a signed-int8 DIRECT/1x1 conv
    // with symmetric weights: int8 x int8 -> int32 on the NPU + host per-axis requant
    // (EXACT int8, no fp16 approximation). Mutually exclusive with the dequant->fp16
    // path; uint8 / depthwise / w_zp!=0 keep is_quant on this struct but native=false.
    bool native = false;             // direct: int32-raw + per-axis requant
    bool native_dw = false;          // depthwise: int8-out on-chip requant (per-tensor)
    bool native_u8 = false;          // direct UINT8 (Option D): recenter to int8 + box-sum requant
    bool needs_boxsum = false;       // native_u8 with an asymmetric weight zp (some w_zp != 128)
    std::vector<int8_t>  w_i8;       // direct [OC][IC][KH][KW] / dw [C][KH][KW] raw int8
    std::vector<int32_t> eff_bias;   // direct: [OC] = bias_q - in_zp*Σ_kernel w_q
    std::vector<int32_t> bias_q;     // dw: raw TFLite int32 bias [C] (runtime folds the corr)
    bool has_bias_q = false;         // a real TFLite int32 bias is present
};

// One claimed FULLY_CONNECTED node (float). A matmul C[M,N] = A[M,K] * B[N,K]^T + bias,
// fused act: M = flattened batch (input elems / K), K = input_dim (weights.dims[1]),
// N = units (weights.dims[0]). The weight B[N,K] is exactly the 1x1-conv / matmul B layout,
// so it packs ONCE into resident NPU BOs (mm_w on the kernel's mm_ctx_) and Eval runs the
// resident fp16 matmul (rocket_matmul_fp16_prepacked) — the same path the 1x1 pointwise conv
// uses. K is zero-padded to %32 and N to %16 (Kpad/Npad) so ANY FC head offloads (the pad
// zeros are transparent to the dot product); a host matmul (rocket_fc_f) is the fd<0 / no-pack
// fallback. M must be 1 or %4 (the matmul alignment); other M (rare) -> CPU fallback.
struct FCNode {
    int in_idx, w_idx, bias_idx, out_idx;
    int M, K, N;                 // batch / input_dim / units (logical, unpadded)
    int act;                     // ROCKET_ACT_*
    // weight/bias are model constants -> read straight from the live tensors (w.data.f /
    // bias.data.f) in Prepare (pack) + Eval (host fallback); no stored copy.
    rocket_weights *mm_w = nullptr;  // resident padded [Npad][Kpad] weight (null => host path)
    int mm_M = 0;                // the M the resident weight was packed for (resize guard)
};

// Pull per-axis (or per-tensor) filter quant params into ws/wz (length OC).
// Returns false if the params are missing or an unexpected length.
static bool fill_filter_quant(const TfLiteTensor &flt, int OC,
                              std::vector<float> &ws, std::vector<int> &wz) {
    ws.assign(OC, 0.f);
    wz.assign(OC, 0);
    if (flt.quantization.type == kTfLiteAffineQuantization && flt.quantization.params) {
        const auto *aq =
            reinterpret_cast<const TfLiteAffineQuantization *>(flt.quantization.params);
        if (!aq->scale || aq->scale->size < 1) return false;
        const int ss = aq->scale->size;
        if (ss != 1 && ss != OC) return false;                 // not per-tensor or per-OC
        const int zs = aq->zero_point ? aq->zero_point->size : 0;
        if (zs != 0 && zs != 1 && zs != OC) return false;      // reject mismatched zero_point len
        for (int oc = 0; oc < OC; oc++) {
            ws[oc] = aq->scale->data[ss == 1 ? 0 : oc];
            wz[oc] = zs ? aq->zero_point->data[zs == 1 ? 0 : oc] : 0;
        }
        return true;
    }
    // legacy per-tensor params
    if (flt.params.scale == 0.f) return false;
    for (int oc = 0; oc < OC; oc++) { ws[oc] = flt.params.scale; wz[oc] = flt.params.zero_point; }
    return true;
}

// Gate + parameter extraction shared by IsNodeSupportedByDelegate and the kernel's
// Init. Handles CONV_2D and DEPTHWISE_CONV_2D (depth_multiplier==1 => OC==IC, one
// KH×KW filter per channel; the driver runs it as a native depthwise job). Returns
// true iff this is a conv the delegate can run; fills *out (when non-null) with the
// static parameters.
static bool analyze_conv(const TfLiteRegistration *reg, const TfLiteNode *node,
                         TfLiteContext *ctx, const RocketOptions &opts, ConvNode *out)
{
    const bool is_dw = (reg->builtin_code == kTfLiteBuiltinDepthwiseConv2d);
    if (reg->builtin_code != kTfLiteBuiltinConv2d && !is_dw) return false;

    // CONV_2D and DEPTHWISE_CONV_2D are distinct builtins with distinct param structs;
    // both carry the same padding/stride/dilation/activation. Depthwise adds
    // depth_multiplier (only 1 is supported => OC==IC).
    int act, sy, sx, dy, dx;
    if (is_dw) {
        const auto *p = reinterpret_cast<const TfLiteDepthwiseConvParams *>(node->builtin_data);
        if (!p) return false;
        if (p->depth_multiplier != 1) return false;   // OC==IC only; else CPU fallback
        act = map_activation(p->activation);
        sy = p->stride_height; sx = p->stride_width;
        dy = p->dilation_height_factor; dx = p->dilation_width_factor;
    } else {
        const auto *p = reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);
        if (!p) return false;
        act = map_activation(p->activation);
        sy = p->stride_height; sx = p->stride_width;
        dy = p->dilation_height_factor; dx = p->dilation_width_factor;
    }
    if (act < 0) return false;
    if (sy < 1 || sx < 1) return false;
    if (dy < 1 || dx < 1) return false;
    if (node->inputs->size < 2 || node->outputs->size < 1) return false;

    const TfLiteTensor &in   = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &flt  = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &outp = ctx->tensors[node->outputs->data[0]];
    const bool has_bias = node->inputs->size > 2 && node->inputs->data[2] >= 0;

    auto is_q = [](TfLiteType t) { return t == kTfLiteInt8 || t == kTfLiteUInt8; };
    const bool float_path = (in.type == kTfLiteFloat32 && flt.type == kTfLiteFloat32 &&
                             outp.type == kTfLiteFloat32);
    const bool quant_path = is_q(in.type) && is_q(flt.type) && is_q(outp.type);
    if (!float_path && !quant_path) return false;                  // float or int8/uint8
    if (has_bias) {
        const TfLiteType bt = ctx->tensors[node->inputs->data[2]].type;
        if (float_path && bt != kTfLiteFloat32) return false;
        if (quant_path && bt != kTfLiteInt32)   return false;      // int8 conv bias = int32
    }
    if (!in.dims || in.dims->size != 4 || !flt.dims || flt.dims->size != 4 ||
        !outp.dims || outp.dims->size != 4) return false;
    if (in.dims->data[0] != 1) return false;                       // batch 1 only

    const int IH = in.dims->data[1], IW = in.dims->data[2], IC = in.dims->data[3];
    // Filter layout differs: conv [OC][KH][KW][IC]; depthwise [1][KH][KW][C] (OC==IC==C).
    int OC, KH, KW, FIC;
    if (is_dw) {
        if (flt.dims->data[0] != 1) return false;     // depth_multiplier folded into C
        KH = flt.dims->data[1]; KW = flt.dims->data[2]; FIC = flt.dims->data[3];
        OC = IC;
    } else {
        OC = flt.dims->data[0]; KH = flt.dims->data[1]; KW = flt.dims->data[2];
        FIC = flt.dims->data[3];
    }
    const int OH = outp.dims->data[1], OW = outp.dims->data[2];
    if (FIC != IC) return false;                  // grouped conv (filter C != input C)
    if (outp.dims->data[3] != OC) return false;
    // OC%16!=0 is fine: the driver pads OC up to the weight oc group (16) and slices the
    // result. IC%G + single-CBUF-pass fit (incl. DW channel/spatial tiling) are gated by
    // rocket_conv2d_plan on the materialized descriptor below.
    if (OH <= 0 || OW <= 0) return false;

    // worth-offloading floor. Depthwise reduces only KH*KW per output (no IC sum).
    const long macs = is_dw ? (long)OC * OH * OW * KH * KW
                            : (long)OC * OH * OW * IC * KH * KW;
    if (macs < opts.min_macs) return false;

    // matmul-offloadable 1x1 stride-1 pointwise? (direct float only — depthwise does a
    // per-channel reduction, never a matmul; the quant path always runs the conv so it
    // can dequant/requant at the boundary)
    const int M = IH * IW;
    const bool mm_ok = !is_dw && float_path
                    && (KH == 1 && KW == 1 && sy == 1 && sx == 1 && dy == 1 && dx == 1
                        && dims_offloadable(M, IC, OC));

    // general conv: validate the MATERIALIZED descriptor (the SAME/VALID pad is
    // folded into a larger zero-haloed input, so the driver runs pad_top=pad_left=0)
    // through the driver's pure planner, and confirm it reproduces TFLite's dims. For
    // depthwise this is also the IC%G + single-CBUF-pass gate (a DW layer too big for
    // one pass returns <0 -> CPU fallback, since the driver has no DW spatial tiling).
    const int tot_h = rocket_total_pad(IH, KH, sy, dy, OH);
    const int tot_w = rocket_total_pad(IW, KW, sx, dx, OW);
    rocket_conv2d_desc d = {};
    d.ic = IC; d.ih = IH + tot_h; d.iw = IW + tot_w; d.oc = OC;
    d.kh = KH; d.kw = KW; d.stride_y = sy; d.stride_x = sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = dy; d.dil_x = dx; d.depthwise = is_dw ? 1 : 0;
    const bool conv_ok = (rocket_conv2d_plan(&d) == 0)
                      && rocket_conv2d_oh(&d) == OH && rocket_conv2d_ow(&d) == OW;

    if (!mm_ok && !conv_ok) return false;

    // Validate (and, for quant, capture) the quantization params. Done even when
    // out==null so IsNodeSupportedByDelegate never claims a node Init can't run.
    // Depthwise filter quant is per-channel along axis 3 (the C dim); since OC==C and
    // channel c maps to output channel c, the flat per-OC array fill_filter_quant
    // produces (length OC) indexes correctly for either layout.
    std::vector<float> ws;
    std::vector<int>   wz;
    if (quant_path) {
        if (in.params.scale <= 0.f || outp.params.scale <= 0.f) return false;
        if (!fill_filter_quant(flt, OC, ws, wz)) return false;
    }

    if (out) {
        out->in_idx   = node->inputs->data[0];
        out->w_idx    = node->inputs->data[1];
        out->bias_idx = has_bias ? node->inputs->data[2] : -1;
        out->out_idx  = node->outputs->data[0];
        out->IC = IC; out->OC = OC; out->KH = KH; out->KW = KW;
        out->sy = sy; out->sx = sx; out->dy = dy; out->dx = dx;
        out->act = act;
        out->depthwise = is_dw;
        out->is_quant = quant_path;
        if (quant_path) {
            out->in_uns  = (in.type   == kTfLiteUInt8);
            out->w_uns   = (flt.type  == kTfLiteUInt8);
            out->out_uns = (outp.type == kTfLiteUInt8);
            out->in_scale  = in.params.scale;    out->in_zp  = in.params.zero_point;
            out->out_scale = outp.params.scale;  out->out_zp = outp.params.zero_point;
            // NATIVE int8: signed-int8 DIRECT/1x1 with symmetric weights only.
            // A quant conv always takes the conv path (mm_ok is float-only), and conv_ok
            // is already required above, so the int8 runtime can run it (1x1 = degenerate
            // conv). uint8 / depthwise / per-axis w_zp!=0 stay on the dequant->fp16 path.
            bool all_w_sym = true;
            for (int oc = 0; oc < OC; oc++) if (wz[oc] != 0) { all_w_sym = false; break; }
            const bool all_i8 = in.type == kTfLiteInt8 && flt.type == kTfLiteInt8 &&
                                outp.type == kTfLiteInt8;
            const bool all_u8 = in.type == kTfLiteUInt8 && flt.type == kTfLiteUInt8 &&
                                outp.type == kTfLiteUInt8;
            // The int8 input zero-point is materialized into the conv pad halo via an
            // int8 memset; an out-of-int8-range in_zp would truncate there. Likewise the
            // uint8 path recenters in_zp by -128 into the int8 range. Reject anything
            // outside the type's range so it stays on the dequant->fp16 path instead.
            const bool in_zp_i8_ok = in.params.zero_point >= -128 && in.params.zero_point <= 127;
            const bool in_zp_u8_ok = in.params.zero_point >= 0    && in.params.zero_point <= 255;
            // DIRECT native: int32-raw + host per-axis requant (per-axis weights are fine).
            out->native = opts.native_int8 && !is_dw && all_i8 && all_w_sym && in_zp_i8_ok;
            // DIRECT native UINT8 (Option D): the common coral/MediaPipe case (uint8 I/O,
            // asymmetric per-tensor weight zp). Recenter the uint8 operands to int8 on the
            // host (x=in_q-128, y=w_q-128 — always in range), reuse the SAME int32-raw NPU
            // conv, and fold the centering into eff_bias + a per-output-pixel box-sum
            // correction (the price of w_zp != 128). No symmetry requirement; uint8
            // depthwise stays on fp16 (a follow-on). Mutually exclusive with native (i8 vs u8).
            out->native_u8 = opts.native_int8 && !is_dw && all_u8 && in_zp_u8_ok;
            if (out->native_u8) {
                // box-sum is only needed when some weight zp != 128 (else beta == 0); the
                // real MobileDet weights are all asymmetric, so this is normally true.
                bool sym128 = true;
                for (int oc = 0; oc < OC; oc++) if (wz[oc] != 128) { sym128 = false; break; }
                out->needs_boxsum = !sym128;
            }
            // DEPTHWISE native (int8-out on-chip requant): PER-TENSOR quant only (Teflon's
            // constraint; per-channel DW needs BS_MUL, stays on fp16) and a SYMMETRIC pad
            // (the runtime uses the CNA's symmetric HW pad — the validated config; an
            // asymmetric SAME pad falls back to fp16). w_zp==0 (symmetric weights).
            bool w_per_tensor = false;
            if (flt.quantization.type == kTfLiteAffineQuantization && flt.quantization.params) {
                const auto *aq =
                    reinterpret_cast<const TfLiteAffineQuantization *>(flt.quantization.params);
                w_per_tensor = aq->scale && aq->scale->size == 1;
            } else if (flt.params.scale != 0.f) {
                w_per_tensor = true;                       // legacy per-tensor
            }
            out->native_dw = opts.native_int8 && is_dw && all_i8 && all_w_sym &&
                             w_per_tensor && (tot_h % 2 == 0) && (tot_w % 2 == 0) && in_zp_i8_ok;
            out->has_bias_q = has_bias;
            out->w_scale = std::move(ws);
            out->w_zp    = std::move(wz);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Aux host ops: the elementwise / pooling / join ops around the convs. Each is a
// thin NHWC host kernel (rocket_ops.h); claiming them keeps the delegated partition
// contiguous. Quant params here are PER-TENSOR (input/output scale+zp), unlike the
// conv filter's per-axis params. Only static params are captured here; spatial dims
// are re-read from live tensors in Eval (resize-safe, like the conv path).
// ---------------------------------------------------------------------------
enum class NodeKind { Conv, Add, Pool, Concat, Reshape, Act, FC, Prelu, Reduce, Resize, TConv,
                      L2Norm, LogSoftmax, Softmax, Cumsum, Transpose, Pad, Slice, Split };

// A standalone unary activation node (HARD_SWISH / LOGISTIC): pure elementwise,
// in/out the SAME shape. kind is a ROCKET_UNARY_* code (rocket_ops.h). Quant is
// per-tensor (one scale/zp each side), like the other aux ops.
struct ActP {
    int in, out, kind;
    float param = 0.f;        // LEAKY_RELU slope (alpha); ignored by the other kinds
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
};
struct AddP {
    int in0, in1, out, act;
    // op selects the binary kernel: -1 = ADD (rocket_add, fused act), else a
    // ROCKET_BINOP_* (MAXIMUM / MINIMUM, no fused act). Keeps one NodeKind/struct
    // for all same-shape two-tensor elementwise ops.
    int op = -1;
    bool is_quant = false;
    int in0_uns = 0, in1_uns = 0, out_uns = 0;
    float in0_scale = 1.f, in1_scale = 1.f, out_scale = 1.f;
    int   in0_zp = 0, in1_zp = 0, out_zp = 0;
};
struct PoolP {
    int in, out;
    int kh, kw, sy, sx, same, is_avg, act;
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
};
struct ConcatP {
    std::vector<int> ins;
    int out, axis, act;
    bool is_quant = false;
    int out_uns = 0; float out_scale = 1.f; int out_zp = 0;
    std::vector<int>   in_uns;
    std::vector<float> in_scale;
    std::vector<int>   in_zp;
};
struct ReshapeP { int in, out, elem_size; };   // shape-only: byte copy
// LAYOUT OPS (TRANSPOSE / PAD / SLICE / SPLIT): pure byte moves — values pass
// through unchanged and the quant scale/zp are identical in==out, so the dtype is
// just elem_size (1 int8/uint8, 4 float32). perm/paddings/begin/size/axis are model
// constants captured at analyze; the live shapes are re-read + re-validated at eval
// (the external delegate can't un-delegate at Invoke). The NPU has no on-chip
// layout-conversion engine, so all four are exact host kernels (rocket_ops.h); the
// value is keeping a real graph's conv->layout->conv in ONE delegated partition.
struct TransposeP { int in, out, elem_size, rank; int perm[ROCKET_MAX_RANK]; };
// SLICE / STRIDED_SLICE: gather begin[d]..begin[d]+size[d] (stride 1) per INPUT axis.
// size[] is stored explicitly (not read from the output dims) so STRIDED_SLICE's
// shrink_axis_mask — which drops a size-1 axis, making the output rank < input rank —
// still works: the gather runs in input rank, the output holds the same elements in
// the same order, validated by an element-count match.
struct SliceP     { int in, out, elem_size, rank; int begin[ROCKET_MAX_RANK]; int size[ROCKET_MAX_RANK];
                    int stride[ROCKET_MAX_RANK]; bool strided = false; };  // strided=false => stride[] unused (SLICE/SPLIT)
struct PadP {
    int in, out, elem_size, rank;
    int pad_before[ROCKET_MAX_RANK];
    bool is_quant = false;
    float pad_value_f = 0.f;          // float PAD (0) / PADV2 constant
    signed char pad_byte = 0;         // quant fill byte (zero_point, or PADV2's quantized const)
};
// SPLIT / SPLIT_V: N outputs, each a contiguous slice along `axis`. The per-output
// begin[axis] is the running sum of the earlier outputs' axis extents (computed at
// eval from the live output dims). Multi-output (the only such NodeKind).
struct SplitP { int in, elem_size, rank, axis; std::vector<int> outs; };
// PRELU: per-channel parametric ReLU. alpha is a model constant (alpha_idx),
// dequantized to float[C] at eval. x/out per-tensor quant like the other aux ops.
struct PreluP {
    int in, alpha_idx, out, C;
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
    bool  alpha_uns = false;
    float alpha_scale = 1.f;
    int   alpha_zp = 0;
};
// SPATIAL REDUCE (MEAN / REDUCE_MAX / REDUCE_MIN over axes [1,2]): in [1,H,W,C] ->
// out [1,C] or [1,1,1,C]. op = ROCKET_REDUCE_*. x/out per-tensor quant.
struct ReduceP {
    int in, out, H, W, C, op;
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
};
// RESIZE_NEAREST_NEIGHBOR / RESIZE_BILINEAR: in [1,IH,IW,C] -> [1,OH,OW,C]. bilinear
// selects op; align_corners / half_pixel from the op params. Quant params unchanged
// (resize never rescales), so quant carries one (scale,zp) for in==out.
struct ResizeP {
    int in, out, IH, IW, C, OH, OW;
    bool bilinear = false, align_corners = false, half_pixel = false;
    bool is_quant = false;
    int uns = 0, elem_size = 4;
    float scale = 1.f; int zp = 0;
};
// TRANSPOSE_CONV (float v1): learned upsample. weights TFLite [OC,KH,KW,IC] (const),
// optional bias (const). pad_h/pad_w precomputed the TFLite way. host exact kernel.
struct TConvP {
    int in, w_idx, bias_idx, out;
    int IH, IW, IC, OC, KH, KW, sy, sx, pad_h, pad_w, OH, OW, act;
};
// L2_NORMALIZATION: per-spatial-position normalize over the channel (last) axis.
// M = product of all dims except last; C = last dim. x/out per-tensor quant.
struct L2NormP {
    int in, out, M, C;
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
};
// LOG_SOFTMAX / CUMSUM: over the last axis. M = product of all dims except last,
// N = last. Cumsum carries exclusive/reverse. x/out per-tensor quant.
struct SeqP {
    int in, out, M, N;
    int exclusive = 0, reverse = 0;   // cumsum only
    float beta = 1.f;                 // softmax only (logit scale; TFLite default 1)
    bool is_quant = false;
    int in_uns = 0, out_uns = 0;
    float in_scale = 1.f, out_scale = 1.f;
    int   in_zp = 0, out_zp = 0;
};

// A claimed node: exactly one of the embedded params is live per `kind` (the unused
// ones stay default — cheap; partitions are small).
struct Node {
    NodeKind kind = NodeKind::Conv;
    ConvNode conv;
    AddP     add;
    PoolP    pool;
    ConcatP  concat;
    ReshapeP reshape{};
    ActP     unary;
    FCNode   fc;
    PreluP   prelu;
    ReduceP  reduce;
    ResizeP  resize;
    TConvP   tconv;
    L2NormP  l2norm;
    SeqP     seq;
    TransposeP transpose;
    PadP       pad;
    SliceP     slice;
    SplitP     split;
};

// ADD: elementwise residual, SAME shape only (broadcast -> CPU). float | int8 | uint8.
static bool analyze_add(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteAddParams *>(node->builtin_data);
    if (!p) return false;
    const int act = map_activation(p->activation);
    if (act < 0) return false;
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &a = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &b = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];
    if (!broadcast_to(a.dims, o.dims) || !broadcast_to(b.dims, o.dims)) return false;  // a,b broadcast to o
    const bool fl = a.type == kTfLiteFloat32 && b.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(a.type) && a.type == b.type && a.type == o.type;
    if (!fl && !q) return false;
    if (q && (a.params.scale <= 0.f || b.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Add;
        AddP &x = out->add;
        x.in0 = node->inputs->data[0]; x.in1 = node->inputs->data[1];
        x.out = node->outputs->data[0]; x.act = act; x.is_quant = q;
        if (q) {
            x.in0_uns = (a.type == kTfLiteUInt8); x.in1_uns = (b.type == kTfLiteUInt8);
            x.out_uns = (o.type == kTfLiteUInt8);
            x.in0_scale = a.params.scale; x.in0_zp = a.params.zero_point;
            x.in1_scale = b.params.scale; x.in1_zp = b.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// MAXIMUM / MINIMUM: elementwise two-tensor, SAME shape only (broadcast -> CPU).
// No fused-activation param (the op is the whole node). float | int8 | uint8.
// Shares NodeKind::Add / AddP, distinguished by AddP.op (ROCKET_BINOP_MAX/MIN).
static bool analyze_binary(const TfLiteRegistration *reg, const TfLiteNode *node,
                           TfLiteContext *ctx, Node *out) {
    int op;
    switch (reg->builtin_code) {
    case kTfLiteBuiltinMaximum: op = ROCKET_BINOP_MAX; break;
    case kTfLiteBuiltinMinimum: op = ROCKET_BINOP_MIN; break;
    default: return false;
    }
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &a = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &b = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];
    if (!broadcast_to(a.dims, o.dims) || !broadcast_to(b.dims, o.dims)) return false;  // a,b broadcast to o
    const bool fl = a.type == kTfLiteFloat32 && b.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(a.type) && a.type == b.type && a.type == o.type;
    if (!fl && !q) return false;
    if (q && (a.params.scale <= 0.f || b.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Add;
        AddP &x = out->add;
        x.in0 = node->inputs->data[0]; x.in1 = node->inputs->data[1];
        x.out = node->outputs->data[0]; x.act = ROCKET_ACT_NONE; x.op = op; x.is_quant = q;
        if (q) {
            x.in0_uns = (a.type == kTfLiteUInt8); x.in1_uns = (b.type == kTfLiteUInt8);
            x.out_uns = (o.type == kTfLiteUInt8);
            x.in0_scale = a.params.scale; x.in0_zp = a.params.zero_point;
            x.in1_scale = b.params.scale; x.in1_zp = b.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// MUL / SUB / DIV: elementwise two-tensor, SAME shape only (broadcast -> CPU), WITH a
// fused activation (like ADD). float | int8 | uint8. Shares NodeKind::Add / AddP,
// distinguished by AddP.op (ROCKET_ARITH_MUL/SUB/DIV); eval_add routes these to the
// fused-act arithmetic host kernel. SUB/DIV are non-commutative: in0 op in1.
static bool analyze_arith(const TfLiteRegistration *reg, const TfLiteNode *node,
                          TfLiteContext *ctx, Node *out) {
    int op, act;
    switch (reg->builtin_code) {
    case kTfLiteBuiltinMul: {
        const auto *p = reinterpret_cast<const TfLiteMulParams *>(node->builtin_data);
        if (!p) return false; op = ROCKET_ARITH_MUL; act = map_activation(p->activation); break; }
    case kTfLiteBuiltinSub: {
        const auto *p = reinterpret_cast<const TfLiteSubParams *>(node->builtin_data);
        if (!p) return false; op = ROCKET_ARITH_SUB; act = map_activation(p->activation); break; }
    case kTfLiteBuiltinDiv: {
        const auto *p = reinterpret_cast<const TfLiteDivParams *>(node->builtin_data);
        if (!p) return false; op = ROCKET_ARITH_DIV; act = map_activation(p->activation); break; }
    default: return false;
    }
    if (act < 0) return false;
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &a = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &b = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];
    if (!broadcast_to(a.dims, o.dims) || !broadcast_to(b.dims, o.dims)) return false;  // a,b broadcast to o
    const bool fl = a.type == kTfLiteFloat32 && b.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(a.type) && a.type == b.type && a.type == o.type;
    if (!fl && !q) return false;
    if (q && (a.params.scale <= 0.f || b.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Add;
        AddP &x = out->add;
        x.in0 = node->inputs->data[0]; x.in1 = node->inputs->data[1];
        x.out = node->outputs->data[0]; x.act = act; x.op = op; x.is_quant = q;
        if (q) {
            x.in0_uns = (a.type == kTfLiteUInt8); x.in1_uns = (b.type == kTfLiteUInt8);
            x.out_uns = (o.type == kTfLiteUInt8);
            x.in0_scale = a.params.scale; x.in0_zp = a.params.zero_point;
            x.in1_scale = b.params.scale; x.in1_zp = b.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// PRELU: per-channel parametric ReLU, out = x>=0 ? x : alpha[c]*x. inputs[0]=x,
// inputs[1]=alpha (a model CONSTANT, one slope per channel). Claimed when alpha is a
// constant whose element count == the channel (last) dim of x and broadcasts over the
// other axes (the standard Keras shared_axes=[1,2] / detector PReLU). Same shape/type
// in->out. float | int8 | uint8 (per-tensor x/out/alpha quant). The channel is NHWC's
// innermost index, so the host kernel needs only C (= last dim).
static bool analyze_prelu(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &al = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims)) return false;
    if (!in.dims || in.dims->size < 1) return false;
    if (!al.data.data) return false;                       // alpha must be a constant
    const int C = in.dims->data[in.dims->size - 1];
    if (C <= 0 || dims_count(al.dims) != (long)C) return false;   // per-channel, broadcast
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32 && al.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type && is_quant_type(al.type);
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f || al.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Prelu;
        PreluP &x = out->prelu;
        x.in = node->inputs->data[0]; x.alpha_idx = node->inputs->data[1];
        x.out = node->outputs->data[0]; x.C = C; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.alpha_uns = (al.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
            x.alpha_scale = al.params.scale; x.alpha_zp = al.params.zero_point;
        }
    }
    return true;
}

// MEAN / REDUCE_MAX / REDUCE_MIN over the SPATIAL axes [1,2]: in [1,H,W,C] ->
// [1,C] (keepdims=False) or [1,1,1,C] (True). inputs[1] is the axis CONSTANT, which
// must be exactly the set {1,2}. batch 1. float | int8 | uint8 (per-tensor quant).
// Other axis sets / ranks / reduce ops -> CPU (this is the global-pool form only).
static bool analyze_reduce(const TfLiteRegistration *reg, const TfLiteNode *node,
                           TfLiteContext *ctx, Node *out) {
    int op;
    switch (reg->builtin_code) {
    case kTfLiteBuiltinMean:      op = ROCKET_REDUCE_MEAN; break;
    case kTfLiteBuiltinReduceMax: op = ROCKET_REDUCE_MAX;  break;
    case kTfLiteBuiltinReduceMin: op = ROCKET_REDUCE_MIN;  break;
    default: return false;
    }
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &ax = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || in.dims->size != 4 || in.dims->data[0] != 1) return false;   // NHWC, batch 1
    if (!ax.data.data || ax.type != kTfLiteInt32) return false;                   // axis constant
    if (dims_count(ax.dims) != 2) return false;                                   // exactly 2 axes
    // accept {1,2} in either order (normalize negatives: -3->1, -2->2 for rank 4)
    int a0 = ax.data.i32[0], a1 = ax.data.i32[1];
    auto norm = [](int a) { return a < 0 ? a + 4 : a; };
    a0 = norm(a0); a1 = norm(a1);
    if (!((a0 == 1 && a1 == 2) || (a0 == 2 && a1 == 1))) return false;            // spatial only
    const int H = in.dims->data[1], W = in.dims->data[2], C = in.dims->data[3];
    if (H <= 0 || W <= 0 || C <= 0) return false;
    if (dims_count(o.dims) != (long)C) return false;                             // [1,C] or [1,1,1,C]
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Reduce;
        ReduceP &x = out->reduce;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.H = H; x.W = W; x.C = C; x.op = op; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// RESIZE_NEAREST_NEIGHBOR / RESIZE_BILINEAR: in [1,IH,IW,C] -> [1,OH,OW,C].
// inputs[1] is the int32 size CONSTANT {OH,OW}. align_corners/half_pixel come from
// the op params; the host kernel mirrors TFLite's coordinate math exactly. Quant:
// resize never rescales, so in/out must carry the SAME (scale,zp); nearest is a byte
// gather, bilinear a dequant->lerp->requant. float | int8 | uint8, batch 1.
static bool analyze_resize(const TfLiteRegistration *reg, const TfLiteNode *node,
                           TfLiteContext *ctx, Node *out) {
    bool bilinear;
    bool align_corners, half_pixel;
    switch (reg->builtin_code) {
    case kTfLiteBuiltinResizeNearestNeighbor: {
        bilinear = false;
        const auto *p = reinterpret_cast<const TfLiteResizeNearestNeighborParams *>(node->builtin_data);
        if (!p) return false;
        align_corners = p->align_corners; half_pixel = p->half_pixel_centers;
        break;
    }
    case kTfLiteBuiltinResizeBilinear: {
        bilinear = true;
        const auto *p = reinterpret_cast<const TfLiteResizeBilinearParams *>(node->builtin_data);
        if (!p) return false;
        align_corners = p->align_corners; half_pixel = p->half_pixel_centers;
        break;
    }
    default: return false;
    }
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &sz = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || in.dims->size != 4 || in.dims->data[0] != 1) return false;
    if (!o.dims  || o.dims->size  != 4) return false;
    if (!sz.data.data || sz.type != kTfLiteInt32 || dims_count(sz.dims) != 2) return false;
    const int OH = sz.data.i32[0], OW = sz.data.i32[1];
    const int IH = in.dims->data[1], IW = in.dims->data[2], C = in.dims->data[3];
    if (OH <= 0 || OW <= 0 || IH <= 0 || IW <= 0 || C <= 0) return false;
    if (o.dims->data[1] != OH || o.dims->data[2] != OW || o.dims->data[3] != C) return false;
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q) {
        // resize keeps quant params; require in==out so nearest can byte-gather safely.
        if (in.params.scale <= 0.f) return false;
        if (in.params.scale != o.params.scale || in.params.zero_point != o.params.zero_point)
            return false;
    }
    if (out) {
        out->kind = NodeKind::Resize;
        ResizeP &x = out->resize;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.IH = IH; x.IW = IW; x.C = C; x.OH = OH; x.OW = OW;
        x.bilinear = bilinear; x.align_corners = align_corners; x.half_pixel = half_pixel;
        x.is_quant = q;
        if (q) {
            x.uns = (in.type == kTfLiteUInt8); x.elem_size = 1;
            x.scale = in.params.scale; x.zp = in.params.zero_point;
        }
    }
    return true;
}

// TRANSPOSE_CONV (float v1): inputs [output_shape(const), weights(const), input,
// bias(opt const)]. weights TFLite [OC,KH,KW,IC]. pad_h/pad_w computed the TFLite way
// (ComputePadding on the OUTPUT spatial dims). Scatter convention matches TFLite, so
// the host kernel is bit-exact. int8/uint8 (per-axis weight scales) deferred -> CPU.
static bool analyze_tconv(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteTransposeConvParams *>(node->builtin_data);
    if (!p) return false;
    if (node->inputs->size < 3 || node->outputs->size < 1) return false;
    const TfLiteTensor &osh = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &w   = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &in  = ctx->tensors[node->inputs->data[2]];
    const TfLiteTensor &o   = ctx->tensors[node->outputs->data[0]];
    const bool has_bias = node->inputs->size > 3 && node->inputs->data[3] >= 0;
    if (in.type != kTfLiteFloat32 || w.type != kTfLiteFloat32 || o.type != kTfLiteFloat32)
        return false;                                          // float v1
    if (!w.data.data) return false;                            // weight const
    if (has_bias && (ctx->tensors[node->inputs->data[3]].type != kTfLiteFloat32 ||
                     !ctx->tensors[node->inputs->data[3]].data.data)) return false;
    if (!in.dims || in.dims->size != 4 || in.dims->data[0] != 1) return false;
    if (!w.dims  || w.dims->size  != 4) return false;          // [OC,KH,KW,IC]
    if (!o.dims  || o.dims->size  != 4) return false;
    const int IH = in.dims->data[1], IW = in.dims->data[2], IC = in.dims->data[3];
    const int OC = w.dims->data[0], KH = w.dims->data[1], KW = w.dims->data[2];
    if (w.dims->data[3] != IC) return false;                   // weight IC must match input
    const int OH = o.dims->data[1], OW = o.dims->data[2];
    if (o.dims->data[3] != OC) return false;
    if (IH<=0||IW<=0||IC<=0||OC<=0||KH<=0||KW<=0||OH<=0||OW<=0) return false;
    const int sy = p->stride_height, sx = p->stride_width;
    if (sy < 1 || sx < 1) return false;
    int act = ROCKET_ACT_NONE;
    // newer TFLite TRANSPOSE_CONV carries a fused activation; map it (NONE if unsupported).
    act = map_activation(p->activation);
    if (act < 0) return false;
    // TFLite pad: ComputePadding on the OUTPUT spatial dims (transpose = forward grad).
    auto pad_of = [](int padding_same, int out_dim, int k, int stride) -> int {
        int out_size = padding_same ? (out_dim + stride - 1) / stride
                                    : (out_dim - k + stride) / stride;        // ComputeOutSize
        int pad = ((out_size - 1) * stride + k - out_dim) / 2;
        return pad > 0 ? pad : 0;
    };
    const int same = (p->padding == kTfLitePaddingSame);
    const int pad_h = pad_of(same, OH, KH, sy), pad_w = pad_of(same, OW, KW, sx);
    if (out) {
        out->kind = NodeKind::TConv;
        TConvP &x = out->tconv;
        x.in = node->inputs->data[2]; x.w_idx = node->inputs->data[1];
        x.bias_idx = has_bias ? node->inputs->data[3] : -1; x.out = node->outputs->data[0];
        x.IH = IH; x.IW = IW; x.IC = IC; x.OC = OC; x.KH = KH; x.KW = KW;
        x.sy = sy; x.sx = sx; x.pad_h = pad_h; x.pad_w = pad_w; x.OH = OH; x.OW = OW; x.act = act;
        (void)osh;
    }
    return true;
}

// L2_NORMALIZATION: normalize each row over the last (channel) axis. M = product of
// all dims except last, C = last dim. activation must be NONE (the standard op). float
// | int8 | uint8 (per-tensor quant). Same shape in->out.
static bool analyze_l2norm(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteL2NormParams *>(node->builtin_data);
    if (p && map_activation(p->activation) != ROCKET_ACT_NONE) return false;
    if (node->inputs->size != 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims)) return false;
    if (!in.dims || in.dims->size < 1) return false;
    const int C = in.dims->data[in.dims->size - 1];
    const long n = dims_count(in.dims);
    if (C <= 0 || n <= 0 || n % C != 0) return false;
    const int M = (int)(n / C);
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::L2Norm;
        L2NormP &x = out->l2norm;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0]; x.M = M; x.C = C; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// LOG_SOFTMAX: x - logsumexp(x) over the last axis. M = prod(except last), N = last.
// Same shape in->out. float | int8 | uint8 (per-tensor quant).
static bool analyze_logsoftmax(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size != 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims) || !in.dims || in.dims->size < 1) return false;
    const int N = in.dims->data[in.dims->size - 1];
    const long n = dims_count(in.dims);
    if (N <= 0 || n <= 0 || n % N != 0) return false;
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::LogSoftmax;
        SeqP &x = out->seq;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.M = (int)(n / N); x.N = N; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// SOFTMAX: out = softmax(beta*x) over the last axis (TFLite SOFTMAX always reduces the
// innermost dim). Mirrors analyze_logsoftmax: M = product of all dims except last, N =
// last; exact host kernel (the NPU rocket_softmax_fp16 [M][N] route is a follow-on).
// float | int8 | uint8 (per-tensor quant). beta from TfLiteSoftmaxParams (default 1).
static bool analyze_softmax(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size != 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims) || !in.dims || in.dims->size < 1) return false;
    const int N = in.dims->data[in.dims->size - 1];
    const long n = dims_count(in.dims);
    if (N <= 0 || n <= 0 || n % N != 0) return false;
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    const auto *p = reinterpret_cast<const TfLiteSoftmaxParams *>(node->builtin_data);
    if (out) {
        out->kind = NodeKind::Softmax;
        SeqP &x = out->seq;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.M = (int)(n / N); x.N = N; x.beta = p ? p->beta : 1.f; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// CUMSUM: prefix sum along an axis. inputs[1] is the int32 scalar axis CONSTANT, which
// must be the LAST axis (the driver/host kernel form). exclusive/reverse from params.
// M = prod(except last), N = last. float | int8 | uint8 (per-tensor quant).
static bool analyze_cumsum(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteCumsumParams *>(node->builtin_data);
    if (!p) return false;
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &ax = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims) || !in.dims || in.dims->size < 1) return false;
    if (!ax.data.data || ax.type != kTfLiteInt32 || dims_count(ax.dims) != 1) return false;
    const int rank = in.dims->size;
    int axis = ax.data.i32[0]; if (axis < 0) axis += rank;
    if (axis != rank - 1) return false;                       // last axis only
    const int N = in.dims->data[rank - 1];
    const long n = dims_count(in.dims);
    if (N <= 0 || n <= 0 || n % N != 0) return false;
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Cumsum;
        SeqP &x = out->seq;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.M = (int)(n / N); x.N = N; x.exclusive = p->exclusive ? 1 : 0; x.reverse = p->reverse ? 1 : 0;
        x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// AVERAGE / MAX_POOL_2D: NHWC, batch 1, channels preserved. float | int8 | uint8.
static bool analyze_pool(const TfLiteRegistration *reg, const TfLiteNode *node,
                         TfLiteContext *ctx, Node *out) {
    const bool is_avg = reg->builtin_code == kTfLiteBuiltinAveragePool2d;
    const auto *p = reinterpret_cast<const TfLitePoolParams *>(node->builtin_data);
    if (!p) return false;
    const int act = map_activation(p->activation);
    if (act < 0) return false;
    if (p->stride_height < 1 || p->stride_width < 1) return false;
    if (p->filter_height < 1 || p->filter_width < 1) return false;
    if (node->inputs->size < 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || in.dims->size != 4 || !o.dims || o.dims->size != 4) return false;
    if (in.dims->data[0] != 1 || o.dims->data[0] != 1) return false;
    if (in.dims->data[3] != o.dims->data[3]) return false;            // channels preserved
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    const int same = (p->padding == kTfLitePaddingSame);
    const int IH = in.dims->data[1], IW = in.dims->data[2];
    const int OH = o.dims->data[1],  OW = o.dims->data[2];
    if (rocket_out_dim(IH, p->filter_height, p->stride_height, 1, same) != OH) return false;
    if (rocket_out_dim(IW, p->filter_width,  p->stride_width,  1, same) != OW) return false;
    if (out) {
        out->kind = NodeKind::Pool;
        PoolP &x = out->pool;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.kh = p->filter_height; x.kw = p->filter_width;
        x.sy = p->stride_height; x.sx = p->stride_width;
        x.same = same; x.is_avg = is_avg; x.act = act; x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// CONCATENATION: join N inputs along `axis` (negative normalized). All inputs match
// the output in rank, type, and every extent except `axis`; the axis extents sum.
static bool analyze_concat(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteConcatenationParams *>(node->builtin_data);
    if (!p) return false;
    const int act = map_activation(p->activation);
    if (act < 0) return false;
    const int N = node->inputs->size;
    if (N < 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];
    if (!o.dims) return false;
    const int rank = o.dims->size;
    int axis = p->axis; if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;
    const TfLiteType ot = o.type;
    const bool fl = ot == kTfLiteFloat32;
    const bool q  = is_quant_type(ot);
    if (!fl && !q) return false;
    if (q && o.params.scale <= 0.f) return false;
    long axis_sum = 0;
    for (int i = 0; i < N; i++) {
        const TfLiteTensor &in = ctx->tensors[node->inputs->data[i]];
        if (!in.dims || in.dims->size != rank || in.type != ot) return false;
        for (int d = 0; d < rank; d++)
            if (d != axis && in.dims->data[d] != o.dims->data[d]) return false;
        if (q && in.params.scale <= 0.f) return false;
        axis_sum += in.dims->data[axis];
    }
    if (axis_sum != o.dims->data[axis]) return false;
    if (out) {
        out->kind = NodeKind::Concat;
        ConcatP &x = out->concat;
        x.out = node->outputs->data[0]; x.axis = axis; x.act = act; x.is_quant = q;
        x.ins.resize(N);
        if (q) {
            x.in_uns.resize(N); x.in_scale.resize(N); x.in_zp.resize(N);
            x.out_uns = (ot == kTfLiteUInt8);
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
        for (int i = 0; i < N; i++) {
            const TfLiteTensor &in = ctx->tensors[node->inputs->data[i]];
            x.ins[i] = node->inputs->data[i];
            if (q) {
                x.in_uns[i] = (in.type == kTfLiteUInt8);
                x.in_scale[i] = in.params.scale; x.in_zp[i] = in.params.zero_point;
            }
        }
    }
    return true;
}

// RESHAPE: shape-only passthrough (same element count + type => a byte copy; quant
// params are preserved, never rescaled). Input 1 (shape) is ignored — output dims
// come from the live output tensor.
static bool analyze_reshape(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size < 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || !o.dims || in.type != o.type) return false;
    if (!same_quant_params(in, o)) return false;            // byte copy => quant must match
    int esz;
    switch (in.type) {
    case kTfLiteFloat32:               esz = 4; break;
    case kTfLiteInt8: case kTfLiteUInt8: esz = 1; break;
    default: return false;                                  // only data-path types
    }
    if (dims_count(in.dims) != dims_count(o.dims)) return false;
    if (out) {
        out->kind = NodeKind::Reshape;
        out->reshape.in = node->inputs->data[0];
        out->reshape.out = node->outputs->data[0];
        out->reshape.elem_size = esz;
    }
    return true;
}

// TRANSPOSE: reorder axes by a constant int32 perm (input 1). Output rank == input
// rank, output dims = in_dims[perm[d]]. Pure byte gather (rocket_transpose_bytes).
static bool analyze_transpose(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size != 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &pm = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || !o.dims || in.type != o.type) return false;
    if (!same_quant_params(in, o)) return false;            // byte gather => quant must match
    const int esz = type_elem_size(in.type); if (!esz) return false;
    const int rank = in.dims->size;
    if (rank < 1 || rank > ROCKET_MAX_RANK || o.dims->size != rank) return false;
    if (!pm.data.data || pm.type != kTfLiteInt32 || dims_count(pm.dims) != rank) return false;
    int perm[ROCKET_MAX_RANK]; bool seen[ROCKET_MAX_RANK] = { false };
    for (int d = 0; d < rank; d++) {
        const int p = pm.data.i32[d];
        if (p < 0 || p >= rank || seen[p]) return false;            // a valid permutation
        seen[p] = true; perm[d] = p;
        if (o.dims->data[d] != in.dims->data[p]) return false;      // output shape consistent
    }
    if (out) {
        out->kind = NodeKind::Transpose;
        TransposeP &x = out->transpose;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.elem_size = esz; x.rank = rank;
        for (int d = 0; d < rank; d++) x.perm[d] = perm[d];
    }
    return true;
}

// SLICE: crop a contiguous block. begin (input 1) + size (input 2) are constant
// int32/int64, rank elements each; size[d]==-1 means "to the end". Output dims ==
// size. Pure byte gather (rocket_slice_bytes).
static bool analyze_slice(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    if (node->inputs->size != 3 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &bg = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &sz = ctx->tensors[node->inputs->data[2]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || !o.dims || in.type != o.type) return false;
    if (!same_quant_params(in, o)) return false;            // byte gather => quant must match
    const int esz = type_elem_size(in.type); if (!esz) return false;
    const int rank = in.dims->size;
    if (rank < 1 || rank > ROCKET_MAX_RANK || o.dims->size != rank) return false;
    const bool bg_int = bg.type == kTfLiteInt32 || bg.type == kTfLiteInt64;
    const bool sz_int = sz.type == kTfLiteInt32 || sz.type == kTfLiteInt64;
    if (!bg.data.data || !bg_int || dims_count(bg.dims) != rank) return false;
    if (!sz.data.data || !sz_int || dims_count(sz.dims) != rank) return false;
    int begin[ROCKET_MAX_RANK], size[ROCKET_MAX_RANK];
    for (int d = 0; d < rank; d++) {
        const long b = read_const_int(bg, d);
        long s = read_const_int(sz, d);
        if (s < -1) return false;                                   // only -1 is the sentinel
        if (s == -1) s = (long)in.dims->data[d] - b;                // -1 => to the end
        if (b < 0 || s < 0 || b + s > in.dims->data[d]) return false;
        if (o.dims->data[d] != s) return false;
        begin[d] = (int)b; size[d] = (int)s;
    }
    if (out) {
        out->kind = NodeKind::Slice;
        SliceP &x = out->slice;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.elem_size = esz; x.rank = rank;
        for (int d = 0; d < rank; d++) { x.begin[d] = begin[d]; x.size[d] = size[d]; }
    }
    return true;
}

// TFLite StridedSlice start/stop semantics (strided_slice_logic.h StartForAxis /
// StopForAxis), reproduced so a negative- or strided-axis range resolves bit-identically
// to the CPU reference. Inputs are longs to absorb the int-sentinel mask overrides.
static inline long ss_clamp(long v, long lo, long hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline long ss_start_for_axis(long start, long step, bool begin_mask, long dim) {
    if (dim == 0) return 0;
    if (begin_mask) start = (step > 0) ? LONG_MIN : LONG_MAX;   // first / last element
    if (start < 0)  start += dim;
    return (step > 0) ? ss_clamp(start, 0, dim) : ss_clamp(start, 0, dim - 1);
}
static inline long ss_stop_for_axis(long stop, long step, bool end_mask, bool shrink,
                                    long start, long dim) {
    if (dim == 0) return 0;
    if (shrink)    return start + 1;                            // 1-elem slice, end irrelevant
    if (end_mask)  stop = (step > 0) ? LONG_MAX : LONG_MIN;
    if (stop < 0)  stop += dim;
    return (step > 0) ? ss_clamp(stop, 0, dim) : ss_clamp(stop, -1, dim - 1);
}

// STRIDED_SLICE — the strided/masked generalization of SLICE. The begin/end/strides
// arrays form a "spec" list whose length need NOT equal the input rank: ellipsis_mask
// inserts a run of full-range axes, new_axis_mask inserts an output-only size-1 dimension
// that consumes no input axis, and a spec shorter than the rank leaves the trailing input
// axes full-range. This pass EXPANDS that spec into the per-INPUT-axis begin[]/stride[]/
// size[] the gather wants (one entry per input axis, exactly as if every axis had been
// written out), so the gather and eval_slice need no notion of the masks:
//   * new_axis spec -> a size-1 OUTPUT dim, no input axis touched (transparent to the byte
//     gather, which works purely over the input axes; eval_slice validates by element
//     VOLUME not rank, so the extra size-1 output dims pass through).
//   * ellipsis spec -> the remaining input axes not claimed by the other consuming specs,
//     each begin=0/stride=1/size=dim (full range).
//   * a normal spec -> begin[]/stride[]/size[] via the TFLite Start/Stop rules (begin_mask/
//     end_mask take the full extent direction-aware, shrink_axis_mask takes one index and
//     drops the axis). The masks are indexed by SPEC position, not input axis.
// begin[] is the first gathered index (the HIGH index when stride<0); a stride==1 on every
// axis is the SLICE gather (rocket_slice_bytes), else the signed strided gather
// (rocket_strided_slice_bytes, out[k]=in[begin+k*stride]). Output rank may differ from input
// rank (shrink removes, new_axis adds); validated by the element-count match in eval_slice.
static bool analyze_strided_slice(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteStridedSliceParams *>(node->builtin_data);
    if (!p) return false;
    if (node->inputs->size != 4 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &bg = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &en = ctx->tensors[node->inputs->data[2]];
    const TfLiteTensor &st = ctx->tensors[node->inputs->data[3]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || !o.dims || in.type != o.type) return false;
    if (!same_quant_params(in, o)) return false;            // byte gather => quant must match
    const int esz = type_elem_size(in.type); if (!esz) return false;
    const int rank = in.dims->size;
    if (rank < 1 || rank > ROCKET_MAX_RANK) return false;
    const bool bg_int = bg.type == kTfLiteInt32 || bg.type == kTfLiteInt64;
    const bool en_int = en.type == kTfLiteInt32 || en.type == kTfLiteInt64;
    const bool st_int = st.type == kTfLiteInt32 || st.type == kTfLiteInt64;
    const int numspecs = (int)dims_count(bg.dims);
    if (!bg.data.data || !bg_int) return false;
    if (!en.data.data || !en_int || dims_count(en.dims) != numspecs) return false;
    if (!st.data.data || !st_int || dims_count(st.dims) != numspecs) return false;
    if (numspecs < 1 || numspecs > 2 * ROCKET_MAX_RANK) return false;
    // At most one ellipsis (TF rule). The ellipsis fills the input axes the other consuming
    // (non-new-axis) specs leave over: ell_axes = rank - (#consuming specs - #ellipsis).
    const int ellipsis_cnt = __builtin_popcount((unsigned)p->ellipsis_mask);
    if (ellipsis_cnt > 1) return false;
    const int new_axis_cnt = __builtin_popcount((unsigned)p->new_axis_mask);
    const int consuming = numspecs - new_axis_cnt;                  // specs mapping to input axes
    const int ell_axes  = rank - (consuming - ellipsis_cnt);        // axes the ellipsis covers
    if (ell_axes < 0) return false;                                 // more consuming specs than axes
    int begin[ROCKET_MAX_RANK], size[ROCKET_MAX_RANK], stride[ROCKET_MAX_RANK];
    long vol = 1; int out_rank = 0; bool strided = false; int id = 0;   // id = input-axis cursor
    for (int s = 0; s < numspecs; s++) {
        if ((p->new_axis_mask >> s) & 1) { out_rank++; continue; }  // output-only size-1 dim
        if ((p->ellipsis_mask >> s) & 1) {                          // expand to full-range axes
            for (int e = 0; e < ell_axes; e++, id++) {
                const long dim = in.dims->data[id];
                begin[id] = 0; size[id] = (int)dim; stride[id] = 1;
                vol *= dim; out_rank++;
            }
            continue;
        }
        if (id >= rank) return false;
        const long step = read_const_int(st, s);
        if (step == 0) return false;                                // 0 stride -> CPU
        stride[id] = (int)step; if (step != 1) strided = true;
        const long dim = in.dims->data[id];
        const bool shrink = (p->shrink_axis_mask >> s) & 1;
        const long start = ss_start_for_axis(read_const_int(bg, s), step,
                                             (p->begin_mask >> s) & 1, dim);
        const long stop  = ss_stop_for_axis(read_const_int(en, s), step,
                                            (p->end_mask >> s) & 1, shrink, start, dim);
        if (shrink) {                                               // take one index, drop axis
            if (start < 0 || start >= dim) return false;
            begin[id] = (int)start; size[id] = 1;                  // stride irrelevant on a 1-elem axis
        } else {
            const long range = stop - start;
            // loop-step count: stride>0 -> ceil(range/step) for range>0; stride<0 ->
            // ceil(range/step) for range<0 (both terms negative -> positive count).
            const long cnt = step > 0 ? (range <= 0 ? 0 : (range + step - 1) / step)
                                      : (range >= 0 ? 0 : (range + step + 1) / step);
            if (cnt <= 0) return false;                            // empty slice -> CPU
            begin[id] = (int)start; size[id] = (int)cnt;
            vol *= cnt; out_rank++;                                 // shrink axes are size 1, omitted
        }
        id++;
    }
    while (id < rank) {                                             // trailing axes: implicit full range
        const long dim = in.dims->data[id];
        begin[id] = 0; size[id] = (int)dim; stride[id] = 1;
        vol *= dim; out_rank++; id++;
    }
    if (id != rank) return false;                                  // spec consumed != input rank
    if (o.dims->size != out_rank) return false;                    // shrunk removed + new-axis added
    if (dims_count(o.dims) != vol) return false;                   // same element volume
    if (out) {
        out->kind = NodeKind::Slice;
        SliceP &x = out->slice;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.elem_size = esz; x.rank = rank; x.strided = strided;
        for (int d = 0; d < rank; d++) { x.begin[d] = begin[d]; x.size[d] = size[d]; x.stride[d] = stride[d]; }
    }
    return true;
}

// PAD / PADV2: extend the border. paddings (input 1) is a constant int32 [rank,2]
// of {before,after}. Float fills 0 (PAD) or the constant_values scalar (PADV2);
// quant fills the input zero_point (PAD) or the quantized constant byte (PADV2).
// in/out share quant params (a pure copy). Host kernel rocket_pad_bytes.
static bool analyze_pad(const TfLiteRegistration *reg, const TfLiteNode *node,
                        TfLiteContext *ctx, Node *out) {
    const bool padv2 = reg->builtin_code == kTfLiteBuiltinPadv2;
    if (node->inputs->size < (padv2 ? 3 : 2) || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &pd = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!in.dims || !o.dims || in.type != o.type) return false;
    if (!same_quant_params(in, o)) return false;            // border copy => quant must match
    const int esz = type_elem_size(in.type); if (!esz) return false;
    const int rank = in.dims->size;
    if (rank < 1 || rank > ROCKET_MAX_RANK || o.dims->size != rank) return false;
    if (!pd.data.data || pd.type != kTfLiteInt32) return false;
    if (!pd.dims || pd.dims->size != 2 || pd.dims->data[0] != rank || pd.dims->data[1] != 2)
        return false;
    const bool q = is_quant_type(in.type);
    if (q && in.params.scale <= 0.f) return false;
    int pad_before[ROCKET_MAX_RANK];
    for (int d = 0; d < rank; d++) {
        const int b = pd.data.i32[2 * d], a = pd.data.i32[2 * d + 1];
        if (b < 0 || a < 0) return false;
        if (o.dims->data[d] != in.dims->data[d] + b + a) return false;
        pad_before[d] = b;
    }
    float pad_value_f = 0.f; signed char pad_byte = q ? (signed char)in.params.zero_point : 0;
    if (padv2) {                                          // PADV2 supplies a constant value
        const TfLiteTensor &cv = ctx->tensors[node->inputs->data[2]];
        if (!cv.data.data) return false;
        if (cv.type != in.type) return false;             // const must match the padded dtype
        if (q) pad_byte = ((const signed char *)cv.data.data)[0];   // already quantized
        else   pad_value_f = cv.data.f[0];
    }
    if (out) {
        out->kind = NodeKind::Pad;
        PadP &x = out->pad;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0];
        x.elem_size = esz; x.rank = rank; x.is_quant = q;
        x.pad_value_f = pad_value_f; x.pad_byte = pad_byte;
        for (int d = 0; d < rank; d++) x.pad_before[d] = pad_before[d];
    }
    return true;
}

// SPLIT / SPLIT_V: cut into N contiguous outputs along `axis`. SPLIT: inputs
// {axis, data}, equal parts (num_splits from params). SPLIT_V: inputs {data,
// size_splits, axis}, sizes from the live output dims. Every output: same rank +
// type, non-axis extents == input, axis extents sum to the input; quant params
// identical to the input (a pure copy). Multi-output. Host kernel: N slices.
static bool analyze_split(const TfLiteRegistration *reg, const TfLiteNode *node,
                          TfLiteContext *ctx, Node *out) {
    const bool splitv = reg->builtin_code == kTfLiteBuiltinSplitV;
    const int data_idx = splitv ? 0 : 1;
    const int axis_idx = splitv ? 2 : 0;
    if (node->inputs->size != (splitv ? 3 : 2) || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[data_idx]];
    const TfLiteTensor &ax = ctx->tensors[node->inputs->data[axis_idx]];
    if (!in.dims) return false;
    const int esz = type_elem_size(in.type); if (!esz) return false;
    const int rank = in.dims->size;
    if (rank < 1 || rank > ROCKET_MAX_RANK) return false;
    if (!ax.data.data || ax.type != kTfLiteInt32 || dims_count(ax.dims) != 1) return false;
    int axis = ax.data.i32[0]; if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;
    const bool q = is_quant_type(in.type);
    const int nout = node->outputs->size;
    long sum = 0;
    for (int i = 0; i < nout; i++) {
        const TfLiteTensor &od = ctx->tensors[node->outputs->data[i]];
        if (!od.dims || od.dims->size != rank || od.type != in.type) return false;
        for (int d = 0; d < rank; d++)
            if (d != axis && od.dims->data[d] != in.dims->data[d]) return false;
        if (q && (od.params.scale != in.params.scale ||
                  od.params.zero_point != in.params.zero_point)) return false;
        sum += od.dims->data[axis];
    }
    if (sum != in.dims->data[axis]) return false;
    if (out) {
        out->kind = NodeKind::Split;
        SplitP &x = out->split;
        x.in = node->inputs->data[data_idx]; x.elem_size = esz; x.rank = rank; x.axis = axis;
        x.outs.resize(nout);
        for (int i = 0; i < nout; i++) x.outs[i] = node->outputs->data[i];
    }
    return true;
}

// Unary activation (HARD_SWISH / LOGISTIC): a standalone elementwise op, in/out the
// SAME shape, float | int8 | uint8. Captured like the other aux ops; run as an exact
// host kernel (or, under act_npu, on the NPU DPU LUT). No fused-activation param of its
// own (the op IS the activation).
static bool analyze_act(const TfLiteRegistration *reg, const TfLiteNode *node,
                        TfLiteContext *ctx, Node *out) {
    int kind;
    float param = 0.f;
    switch (reg->builtin_code) {
    case kTfLiteBuiltinHardSwish: kind = ROCKET_UNARY_HARDSWISH; break;
    case kTfLiteBuiltinLogistic:  kind = ROCKET_UNARY_SIGMOID;   break;
    case kTfLiteBuiltinTanh:      kind = ROCKET_UNARY_TANH;      break;
    case kTfLiteBuiltinElu:       kind = ROCKET_UNARY_ELU;       break;
    case kTfLiteBuiltinLog:       kind = ROCKET_UNARY_LOG;       break;
    case kTfLiteBuiltinRelu:      kind = ROCKET_UNARY_RELU;      break;
    case kTfLiteBuiltinRelu6:     kind = ROCKET_UNARY_RELU6;     break;
    case kTfLiteBuiltinReluN1To1: kind = ROCKET_UNARY_RELU_N1_1; break;
    case kTfLiteBuiltinLeakyRelu: {
        kind = ROCKET_UNARY_LEAKY_RELU;
        const auto *p = reinterpret_cast<const TfLiteLeakyReluParams *>(node->builtin_data);
        if (!p) return false;
        param = p->alpha;
        break;
    }
    case kTfLiteBuiltinExp:       kind = ROCKET_UNARY_EXP;       break;
    case kTfLiteBuiltinSqrt:      kind = ROCKET_UNARY_SQRT;      break;
    case kTfLiteBuiltinRsqrt:     kind = ROCKET_UNARY_RSQRT;     break;
    case kTfLiteBuiltinAbs:       kind = ROCKET_UNARY_ABS;       break;
    case kTfLiteBuiltinNeg:       kind = ROCKET_UNARY_NEG;       break;
    case kTfLiteBuiltinSquare:    kind = ROCKET_UNARY_SQUARE;    break;
    case kTfLiteBuiltinFloor:     kind = ROCKET_UNARY_FLOOR;     break;
    default: return false;
    }
    if (node->inputs->size < 1 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    if (!same_dims(in.dims, o.dims)) return false;             // pure elementwise, no reshape
    const bool fl = in.type == kTfLiteFloat32 && o.type == kTfLiteFloat32;
    const bool q  = is_quant_type(in.type) && in.type == o.type;   // same int8/uint8 storage
    if (!fl && !q) return false;
    if (q && (in.params.scale <= 0.f || o.params.scale <= 0.f)) return false;
    if (out) {
        out->kind = NodeKind::Act;
        ActP &x = out->unary;
        x.in = node->inputs->data[0]; x.out = node->outputs->data[0]; x.kind = kind;
        x.param = param;
        x.is_quant = q;
        if (q) {
            x.in_uns = (in.type == kTfLiteUInt8); x.out_uns = (o.type == kTfLiteUInt8);
            x.in_scale = in.params.scale; x.in_zp = in.params.zero_point;
            x.out_scale = o.params.scale; x.out_zp = o.params.zero_point;
        }
    }
    return true;
}

// FULLY_CONNECTED (float): a matmul C[M,N]=A[M,K]*B[N,K]^T + bias, fused act. Reuses the
// resident fp16 matmul (the 1x1-conv path). M = input.size/K (flatten), K = weights.dims[1],
// N = weights.dims[0]. Claimed when float, weight is a constant, M is 1 or %4 (matmul
// alignment), and the input flattens evenly to [M,K]. K/N are padded (%32 / %16) in Prepare,
// so N (= num classes / FC units) need not be aligned. int8 FC stays on CPU (a follow-up).
static bool analyze_fc(const TfLiteNode *node, TfLiteContext *ctx, Node *out) {
    const auto *p = reinterpret_cast<const TfLiteFullyConnectedParams *>(node->builtin_data);
    if (!p) return false;
    if (p->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) return false;
    const int act = map_activation(p->activation);
    if (act < 0) return false;
    if (node->inputs->size < 2 || node->outputs->size < 1) return false;
    const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
    const TfLiteTensor &w  = ctx->tensors[node->inputs->data[1]];
    const TfLiteTensor &o  = ctx->tensors[node->outputs->data[0]];
    const bool has_bias = node->inputs->size > 2 && node->inputs->data[2] >= 0;
    if (in.type != kTfLiteFloat32 || w.type != kTfLiteFloat32 || o.type != kTfLiteFloat32)
        return false;                                          // float path only (v1)
    if (has_bias && ctx->tensors[node->inputs->data[2]].type != kTfLiteFloat32) return false;
    if (!w.dims || w.dims->size != 2 || !in.dims || in.dims->size < 1 || !o.dims) return false;
    if (!w.data.data) return false;                            // weight must be a constant
    const int N = w.dims->data[0], K = w.dims->data[1];
    if (K <= 0 || N <= 0) return false;
    const long in_elems = dims_count(in.dims);
    if (in_elems <= 0) return false;                           // reject overflow/negative dims
    if (in_elems % K != 0) return false;                       // input flattens to [M, K]
    if (in_elems / K > INT_MAX) return false;                  // M must fit int (no truncation)
    const int M = (int)(in_elems / K);
    if (!(M % 4 == 0 || M == 1)) return false;                 // matmul M alignment; else CPU
    if (dims_count(o.dims) != (long)M * N) return false;       // output is [M, N]
    if (out) {
        out->kind = NodeKind::FC;
        FCNode &x = out->fc;
        x.in_idx = node->inputs->data[0]; x.w_idx = node->inputs->data[1];
        x.bias_idx = has_bias ? node->inputs->data[2] : -1;
        x.out_idx = node->outputs->data[0];
        x.M = M; x.K = K; x.N = N; x.act = act;
    }
    return true;
}

// Gate + capture for ANY claimed op. Conv is always eligible; the aux host ops are
// behind `aux_ops`. Shared by IsNodeSupportedByDelegate (out==null) and Init.
static bool analyze_node(const TfLiteRegistration *reg, const TfLiteNode *node,
                         TfLiteContext *ctx, const RocketOptions &opts, Node *out) {
    switch (reg->builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d: {
        const bool ok = analyze_conv(reg, node, ctx, opts, out ? &out->conv : nullptr);
        if (ok && out) out->kind = NodeKind::Conv;
        return ok;
    }
    case kTfLiteBuiltinAdd:           return opts.aux_ops && analyze_add(node, ctx, out);
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinMaxPool2d:     return opts.aux_ops && analyze_pool(reg, node, ctx, out);
    case kTfLiteBuiltinConcatenation: return opts.aux_ops && analyze_concat(node, ctx, out);
    case kTfLiteBuiltinReshape:       return opts.aux_ops && analyze_reshape(node, ctx, out);
    case kTfLiteBuiltinTranspose:     return opts.aux_ops && analyze_transpose(node, ctx, out);
    case kTfLiteBuiltinSlice:         return opts.aux_ops && analyze_slice(node, ctx, out);
    case kTfLiteBuiltinStridedSlice:  return opts.aux_ops && analyze_strided_slice(node, ctx, out);
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:         return opts.aux_ops && analyze_pad(reg, node, ctx, out);
    case kTfLiteBuiltinSplit:
    case kTfLiteBuiltinSplitV:        return opts.aux_ops && analyze_split(reg, node, ctx, out);
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinElu:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinSqrt:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinNeg:
    case kTfLiteBuiltinSquare:
    case kTfLiteBuiltinFloor:         return opts.aux_ops && analyze_act(reg, node, ctx, out);
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:       return opts.aux_ops && analyze_binary(reg, node, ctx, out);
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinDiv:           return opts.aux_ops && analyze_arith(reg, node, ctx, out);
    case kTfLiteBuiltinPrelu:         return opts.aux_ops && analyze_prelu(node, ctx, out);
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:     return opts.aux_ops && analyze_reduce(reg, node, ctx, out);
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinResizeBilinear: return opts.aux_ops && analyze_resize(reg, node, ctx, out);
    case kTfLiteBuiltinTransposeConv:  return opts.aux_ops && analyze_tconv(node, ctx, out);
    case kTfLiteBuiltinL2Normalization: return opts.aux_ops && analyze_l2norm(node, ctx, out);
    case kTfLiteBuiltinLogSoftmax:    return opts.aux_ops && analyze_logsoftmax(node, ctx, out);
    case kTfLiteBuiltinSoftmax:       return opts.aux_ops && analyze_softmax(node, ctx, out);
    case kTfLiteBuiltinCumsum:        return opts.aux_ops && analyze_cumsum(node, ctx, out);
    case kTfLiteBuiltinFullyConnected: return analyze_fc(node, ctx, out);
    default:                          return false;
    }
}

// Run fn(oh0,oh1) over the output rows [0,OH), fanned across up to `nthreads` big
// cores (the native-conv requant — the delegate's largest host cost — splits by row).
// `work` is the output element count (OH*OW*OC); below a floor the thread-spawn cost
// would exceed the work, so it runs inline on the caller. Disjoint row bands => no
// races and bit-identical to one pass (the band functions index by absolute oh). Every
// band runs on a PINNED worker (not the caller, which may sit on an A55 and straggle the
// join); the caller just blocks in join, so it never contends with a worker for a core.
// Exposure probe (ROCKET_PROF_POOL): tally fan-out calls + threads spawned so a
// detector run can be weighed against the measured pthread create+join cost, before
// committing to a persistent pool. Zero cost when the env is unset (one getenv, cached).
static std::atomic<uint64_t> g_po_fanout{0}, g_po_threads{0}, g_po_wall_ns{0};
static int g_po_probe = -1;
static void po_probe_dump() {
    if (g_po_probe <= 0) return;
    fprintf(stderr, "[rocket-pool-probe] fanout_calls=%llu threads_spawned=%llu "
            "fanout_wall=%.3f ms\n",
            (unsigned long long)g_po_fanout.load(),
            (unsigned long long)g_po_threads.load(),
            g_po_wall_ns.load() / 1e6);
}
static bool po_probe_on() {
    if (g_po_probe < 0) {
        const char *e = getenv("ROCKET_PROF_POOL");
        const int on = (e && *e && strcmp(e, "0")) ? 1 : 0;
        // Register the exit dump EXACTLY once. Under a multi-interpreter pool (Frigate P=1->4)
        // two threads can reach this check-then-set concurrently: the g_po_probe write is benign
        // (both compute the same `on`), but a double atexit(po_probe_dump) would print the probe
        // line twice. An atomic test-and-set gates the registration to the first arrival.
        static std::atomic<bool> registered{false};
        if (on && !registered.exchange(true)) atexit(po_probe_dump);
        g_po_probe = on;
    }
    return g_po_probe;
}

static void parallel_oh(int OH, size_t work, int nthreads,
                        const std::function<void(int, int)> &fn)
{
    int T = nthreads < 1 ? 1 : nthreads;
    if (T > OH) T = OH;
    if (T <= 1 || work < (size_t)64 * 1024) { fn(0, OH); return; }   // too small to fan out
    const bool probe = po_probe_on();
    const auto pt0 = probe ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    const int per = (OH + T - 1) / T;
    std::vector<std::thread> th;
    th.reserve(T);
    for (int t = 0; t < T; t++) {
        const int oh0 = t * per, oh1 = std::min(OH, oh0 + per);
        if (oh0 >= oh1) break;
        th.emplace_back([t, oh0, oh1, &fn] { rocket_pin_worker(t); fn(oh0, oh1); });
    }
    for (auto &x : th) x.join();
    if (probe) {
        g_po_fanout.fetch_add(1);
        g_po_threads.fetch_add(th.size());
        g_po_wall_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - pt0).count());
    }
}

// ===========================================================================
// RocketKernel — per-partition execution
//
// One instance per delegated partition; the C TfLiteRegistration below routes
// init/free/prepare/invoke to it (init builds it from the TfLiteDelegateParams).
// Runs the partition's claimed ops (conv on the NPU; aux ops on the host).
// ===========================================================================

// ---------------------------------------------------------------------------
// kernel: runs the partition's claimed ops (conv on the NPU; aux ops on the host).
// One instance per delegated partition; the C TfLiteRegistration below routes
// init/free/prepare/invoke to it (init builds it from the TfLiteDelegateParams).
// ---------------------------------------------------------------------------
class RocketKernel {
public:
    explicit RocketKernel(const RocketOptions &opts) : opts_(opts) {}
    ~RocketKernel() {
        // Free resident matmul weights (they hold BOs on mm_ctx_'s fds) BEFORE the ctx
        // — which frees the shared scratch and closes those fds — then the conv fd.
        for (Node &n : nodes_) {
            if (n.kind == NodeKind::Conv && n.conv.mm_w)
                rocket_weights_free(mm_ctx_, n.conv.mm_w);
            if (n.kind == NodeKind::FC && n.fc.mm_w)        // FC shares mm_ctx_ with conv 1x1
                rocket_weights_free(mm_ctx_, n.fc.mm_w);
        }
        if (mm_ctx_) rocket_ctx_free(mm_ctx_);
        // Same ordering for the resident int8 matmul weights (1x1 native int8/uint8 path).
        for (Node &n : nodes_)
            if (n.kind == NodeKind::Conv && n.conv.mm_w_i8)
                rocket_i8_weights_free(mm_i8_ctx_, n.conv.mm_w_i8);
        if (mm_i8_ctx_) rocket_i8_ctx_free(mm_i8_ctx_);
        if (conv_ctx_) rocket_conv_ctx_free(conv_ctx_);   // frees BOs on fd_ — before close
        if (conv_pool_) rocket_conv_pool_free(conv_pool_); // owns its own worker fds
        if (fd_ >= 0) rocket_close(fd_);
    }

    // ----- Init / Prepare / Eval (lifecycle entry points) -----

    TfLiteStatus Init(TfLiteContext *context, const TfLiteDelegateParams *params) {
        std::vector<std::vector<int>> node_ins(params->nodes_to_replace->size);
        std::vector<std::vector<int>> node_outs(params->nodes_to_replace->size);
        std::vector<int> produced;   // every tensor any partition node produces
        for (int i = 0; i < params->nodes_to_replace->size; i++) {
            const int idx = params->nodes_to_replace->data[i];
            TfLiteNode *node = nullptr;
            TfLiteRegistration *reg = nullptr;
            if (context->GetNodeAndRegistration(context, idx, &node, &reg) != kTfLiteOk)
                return kTfLiteError;
            for (int k = 0; k < node->inputs->size; k++)
                node_ins[i].push_back(node->inputs->data[k]);
            for (int k = 0; k < node->outputs->size; k++) {
                const int t = node->outputs->data[k];
                node_outs[i].push_back(t);
                if (t >= 0) produced.push_back(t);
            }
            Node n;
            if (!analyze_node(reg, node, context, opts_, &n)) return kTfLiteError;
            nodes_.push_back(std::move(n));
        }
        // nodes_to_replace is in execution (topological) order; the kernel runs the
        // ops in that order so an op reading an earlier op's output is correct.
        // Defensive: verify no node reads an intra-partition tensor before its producer.
        // A forward reference would mean the claimed subgraph isn't a contiguous,
        // topologically-ordered partition (e.g. a future op claim reading across
        // partitions), which the Eval order silently relies on. Runtime guard (not an
        // assert, so -DNDEBUG can't strip it); cheap — partitions are tens of ops.
        {
            std::vector<int> seen;
            for (int i = 0; i < params->nodes_to_replace->size; i++) {
                for (int t : node_ins[i]) {
                    if (t < 0) continue;
                    const bool intra = std::find(produced.begin(), produced.end(), t) != produced.end();
                    if (!intra) continue;   // external/model input — fine
                    const bool earlier = std::find(seen.begin(), seen.end(), t) != seen.end();
                    // An in-place op reading its own output (input idx == output idx) is
                    // legal and not a forward reference; exempt the current node's outputs.
                    const bool self = std::find(node_outs[i].begin(), node_outs[i].end(), t)
                                      != node_outs[i].end();
                    if (!earlier && !self) {
                        fprintf(stderr, "[rocket] partition node %d reads intra-partition tensor %d "
                                "before its producer (non-topological partition)\n",
                                params->nodes_to_replace->data[i], t);
                        return kTfLiteError;
                    }
                }
                for (int t : node_outs[i]) if (t >= 0) seen.push_back(t);
            }
        }

        // INTERNAL tensors: a tensor that is the OUTPUT of some op in this partition but
        // NOT a partition output (params->output_tensors). TFLite does not allocate arena
        // storage for tensors whose only producer and consumer are the same delegate node
        // — the delegate owns them — so a multi-op partition (e.g. conv->dw->conv->add)
        // would otherwise read/write a NULL .data.data for every intermediate. We allocate
        // our own buffer per internal tensor and bind it onto context->tensors in Eval.
        for (int i = 0; i < params->output_tensors->size; i++)
            part_out_.push_back(params->output_tensors->data[i]);
        for (const Node &n : nodes_) {
            std::vector<int> outs;            // every output (SPLIT has N; all others 1)
            node_outputs(n, outs);
            for (int oidx : outs) {
                if (oidx < 0) continue;
                bool is_part_out = false;
                for (int t : part_out_) if (t == oidx) { is_part_out = true; break; }
                bool dup = false;
                for (int t : internal_idx_) if (t == oidx) { dup = true; break; }
                if (!is_part_out && !dup) internal_idx_.push_back(oidx);
            }
        }
        return kTfLiteOk;
    }

    // Pack each filter ONCE (weights are model constants): TFLite [OC,KH,KW,IC] ->
    // driver [OC,IC,KH,KW] fp16. For a 1x1 that layout is exactly the matmul's
    // [N=OC, K=IC] weight, so the same packed buffer feeds both Eval paths.
    TfLiteStatus Prepare(TfLiteContext *context) {
        if (!opened_) {
            fd_ = rocket_open();          // may be <0: the conv path then computes on
            opened_ = true;               // CPU (rocket_conv2d_fp16 fd<0 oracle)
            if (fd_ < 0) {
                // An explicitly-requested on-NPU path (native_int8 / act_npu / fc_npu /
                // pool_npu) silently computing on the CPU oracle is surprising, so warn
                // once even without profile; the bare default-conv case stays profile-gated.
                const bool npu_requested = opts_.native_int8 || opts_.act_npu ||
                                           opts_.fc_npu || opts_.pool_npu || opts_.ew_npu ||
                                           opts_.resize_npu || opts_.norm_npu;
                if (npu_requested)
                    fprintf(stderr, "[rocket] an on-NPU path was requested but no NPU device "
                            "opened (%d); running on the CPU oracle instead\n", fd_);
                else if (opts_.profile)
                    fprintf(stderr, "[rocket] no NPU device (%d); conv runs on the CPU oracle\n", fd_);
            }
            // Resident-BO pool for the general/depthwise conv path (borrows fd_, frees in
            // the dtor). fd_<0 is fine — the ctx threads through to the CPU oracle. One ctx
            // serves every conv node; its pool grows to the largest tile across the partition.
            conv_ctx_ = rocket_conv_ctx_create(fd_);
            // Multicore worker pool for the NATIVE int8/uint8 DIRECT conv: opens its own
            // nthreads fds (one NPU core each) so a conv's independent OC/OH/OW tiles run
            // in parallel instead of serializing on fd_. Only used by the native_int8
            // direct/1x1 path; the fp16/dw paths keep conv_ctx_. Skipped without a device.
            if (fd_ >= 0 && opts_.native_int8)
                conv_pool_ = rocket_conv_pool_create(opts_.nthreads > 0 ? opts_.nthreads : 4);
        }
        for (Node &n : nodes_)
            if (n.kind == NodeKind::Conv && !n.conv.packed &&
                pack_weights(context, n.conv) != kTfLiteOk) return kTfLiteError;

        // Resident matmul weights for the 1x1 pointwise convs: pack each eligible filter
        // into NPU BOs ONCE here (weights are model constants) and keep the worker fds +
        // shared scratch alive on mm_ctx_, so Eval packs only the activation per inference
        // instead of opening fds + re-scattering B every call. Best-effort: when there is
        // no device, the per-shape scratch cache is full, or a previously-packed conv was
        // resized out of eligibility, mm_w is left/cleared null and Eval falls back to the
        // conv path (which carries the fd<0 CPU oracle). Re-runnable: Prepare may fire
        // again after a resize, so a stale (different-M) resident weight is dropped first.
        for (Node &n : nodes_) {
            if (n.kind != NodeKind::Conv) continue;
            ConvNode &c = n.conv;
            const TfLiteTensor &in = context->tensors[c.in_idx];
            const int M = (in.dims && in.dims->size == 4)
                        ? in.dims->data[1] * in.dims->data[2] : 0;
            const bool want = (M > 0) && conv_is_matmul(c, M);
            if (c.mm_w && (!want || c.mm_M != M)) {     // stale (resize) / no longer eligible
                rocket_weights_free(mm_ctx_, c.mm_w);
                c.mm_w = nullptr;
            }
            if (want && !c.mm_w) {
                if (!mm_ctx_) mm_ctx_ = rocket_ctx_create(opts_.nthreads);
                if (mm_ctx_) {              // c.w is the 1x1 [OC][IC] fp16 == matmul B[N=OC,K=IC]
                    c.mm_w = rocket_weights_pack(mm_ctx_, M, c.IC, c.OC, c.w.data());
                    c.mm_M = M;
                    if (!c.mm_w && opts_.profile)
                        fprintf(stderr, "[rocket] op (conv 1x1 IC=%d OC=%d M=%d): resident pack "
                                "unavailable -> conv path\n", c.IC, c.OC, M);
                }
            }
        }

        // Resident matmul weights for FULLY_CONNECTED: pack the const weight B[N,K] ONCE
        // here, ZERO-PADDED to [Npad,Kpad] (Kpad=ceil(K,32), Npad=ceil(N,16) — the fp16
        // matmul's K%32/N%16; the pad rows/cols are zero, transparent to the dot product),
        // onto mm_ctx_. Eval then runs the resident fp16 matmul. Best-effort + stale-drop
        // like the conv mm_w loop above; mm_w==null (no device / pack full) => Eval's host
        // matmul fallback (rocket_fc_f) keeps it correct off-HW. The pad makes ANY N (num
        // classes / FC units) offloadable without a per-N alignment gate.
        for (Node &n : nodes_) {
            if (n.kind != NodeKind::FC) continue;
            FCNode &f = n.fc;
            // FC defaults to the HOST matmul (rocket_fc_f): exact, and for a ONE-SHOT TFLite
            // FC it beats the NPU, which is DMA/dispatch-bound (pack+readback ~67% of wall,
            // the not-mac-bound finding) — even a 512x1024x512 FC is ~85 ms on the NPU vs
            // ~27 ms host; the resident matmul only wins at LLM-prefill scale / repeated calls.
            // So the NPU path is opt-in (fc_npu=1), and only for a GEMM-shaped M%4 FC (M==1 is
            // a GEMV the NPU runs ~80x slower). mm_w==null => host. Claiming FC regardless keeps
            // the partition contiguous + covers the op (like the aux host kernels).
            const bool want = opts_.fc_npu && fd_ >= 0 && (f.M % 4 == 0);
            if (f.mm_w && (!want || f.mm_M != f.M)) {
                rocket_weights_free(mm_ctx_, f.mm_w);
                f.mm_w = nullptr;
            }
            if (want && !f.mm_w) {
                const TfLiteTensor &w = context->tensors[f.w_idx];
                if (!w.data.f) return kTfLiteError;
                if (!mm_ctx_) mm_ctx_ = rocket_ctx_create(opts_.nthreads);
                if (mm_ctx_) {
                    const int Kpad = round_up(f.K, 32), Npad = round_up(f.N, 16);
                    std::vector<_Float16> Bpad((size_t)Npad * Kpad, (_Float16)0);
                    for (int nn = 0; nn < f.N; nn++)
                        for (int kk = 0; kk < f.K; kk++)
                            Bpad[(size_t)nn * Kpad + kk] = (_Float16)w.data.f[(size_t)nn * f.K + kk];
                    f.mm_w = rocket_weights_pack(mm_ctx_, f.M, Kpad, Npad, Bpad.data());
                    f.mm_M = f.M;
                    if (!f.mm_w && opts_.profile)
                        fprintf(stderr, "[rocket] op (fc K=%d N=%d M=%d): resident pack "
                                "unavailable -> host matmul\n", f.K, f.N, f.M);
                }
            }
        }

        // Resident int8 matmul weights for the native int8/uint8 1x1 pointwise convs (perf
        // Step 1): zero-pad the recentered/raw int8 filter c.w_i8 [OC][IC] to [Np][Kp]
        // (Np=ceil(OC,32), Kp=ceil(IC,32) — the matmul's N%32/K%32) and scatter it into
        // resident BOs ONCE on mm_i8_ctx_. Eval then runs the int8 matmul (output columns
        // fanned across the 3 cores) so these single-tile 1x1s leave the single-core conv
        // path. Same best-effort/stale-drop discipline as the fp16 mm_w loop above; gated on
        // mm_int8 + a real device (the matmul has no fd<0 CPU oracle, so off-HW it stays on
        // the conv path, which does). The padded operands carry zeros => byte-identical to
        // the conv path (proven in convert_test run_mm1x1 + the HW matmul-on/off A/B).
        if (opts_.mm_int8 && fd_ >= 0)
        for (Node &n : nodes_) {
            if (n.kind != NodeKind::Conv) continue;
            ConvNode &c = n.conv;
            const TfLiteTensor &in = context->tensors[c.in_idx];
            const int M = (in.dims && in.dims->size == 4)
                        ? in.dims->data[1] * in.dims->data[2] : 0;
            const bool want = (M > 0) && conv_is_matmul_i8(c, M);
            if (c.mm_w_i8 && (!want || c.mm_i8_M != M)) {     // stale (resize) / no longer eligible
                rocket_i8_weights_free(mm_i8_ctx_, c.mm_w_i8);
                c.mm_w_i8 = nullptr;
            }
            if (want && !c.mm_w_i8) {
                if (!mm_i8_ctx_) mm_i8_ctx_ = rocket_i8_ctx_create(opts_.nthreads);
                if (mm_i8_ctx_) {
                    const int Kp = i8_round32(c.IC), Np = i8_round32(c.OC);
                    // padded B[Np][Kp] from c.w_i8 [OC][IC] (1x1 driver layout == [N,K]):
                    // rows OC..Np-1 and cols IC..Kp-1 are zero (transparent to the int sum).
                    std::vector<int8_t> Bpad((size_t)Np * Kp, 0);
                    for (int oc = 0; oc < c.OC; oc++)
                        memcpy(&Bpad[(size_t)oc * Kp], &c.w_i8[(size_t)oc * c.IC], (size_t)c.IC);
                    c.mm_w_i8 = rocket_i8_weights_pack(mm_i8_ctx_, M, Kp, Np, Bpad.data());
                    c.mm_i8_M = M;
                    if (!c.mm_w_i8 && opts_.profile)
                        fprintf(stderr, "[rocket] op (native 1x1 IC=%d OC=%d M=%d): int8 resident "
                                "pack unavailable -> conv path\n", c.IC, c.OC, M);
                }
            }
        }

        // split the internal tensors into conv-resident (kept fp16-NCHW between two
        // conv-path ops — no transpose/requant/dequant round-trip) and the rest (NHWC). A
        // tensor is conv-resident iff its producer is a GENERAL-conv-path op (mm_w==null:
        // quant / non-aligned 1x1 / depthwise — NOT the prepacked matmul, which is
        // NHWC-native) AND every consumer is a general-conv-path conv reading it as the conv
        // input AND that consumer is NOT an output predictor (re-quantization barrier rule
        // below). Anything touched by a host aux op (ADD/POOL/CONCAT/RESHAPE), a
        // prepacked-matmul conv, or a partition boundary stays NHWC. Decided here (after the
        // mm_w decision). nchw_resident=0 => resident set empty => every internal stays NHWC.
        resident_idx_.clear();
        nhwc_idx_.clear();
        for (int t : internal_idx_) {
            bool resident = opts_.nchw_resident;
            if (resident) {
                const Node *prod = nullptr;
                for (Node &n : nodes_) if (node_output_idx(n) == t) { prod = &n; break; }
                // native-int8/uint8 convs write requantized int8/uint8 NHWC, not fp16-NCHW, so
                // they are never resident producers (the fp16-resident path is a separate lever).
                resident = prod && prod->kind == NodeKind::Conv &&
                           prod->conv.mm_w == nullptr &&
                           !prod->conv.native && !prod->conv.native_dw && !prod->conv.native_u8;
            }
            if (resident)
                for (Node &n : nodes_) {
                    if (!node_reads(n, t)) continue;
                    if (!(n.kind == NodeKind::Conv && n.conv.mm_w == nullptr &&
                          !n.conv.native && !n.conv.native_dw && !n.conv.native_u8 &&
                          n.conv.in_idx == t)) {
                        resident = false; break;   // host op / prepacked / native-int8 conv consumes it
                    }
                    // RE-QUANTIZATION BARRIER (the default-safety rule). Keeping t in fp16-NCHW
                    // skips its producer's int8 requant, so t carries sub-int8 drift from the
                    // int8 reference. That drift is harmless while it stays bounded — but if a
                    // consuming conv's RESULT reaches a PARTITION OUTPUT with no intervening
                    // requant barrier (another conv / ADD / POOL — only RESHAPE/CONCAT are
                    // transparent shape/join ops), the drift lands on the result undamped and
                    // grows by that conv's gain (the SSD head's box/class predictors: a resident
                    // shared feature -> predictor -> reshape -> concat -> output drove
                    // max|delegate-CPU| 2 -> 65). Forcing the last conv before such an output to
                    // read REQUANTIZED (NHWC) input bounds the output drift to one conv's gain on
                    // a <=1-step input — the same regime as the int8 reference's own rounding,
                    // which is the safe "block" case (project -> ADD damps -> max 2). This keeps
                    // damped chains resident (the MobileNetV2 block) and even keeps the SSD head's
                    // upstream `feat` resident (its consumer tap conv is itself a barrier before
                    // the predictors); only the predictors' direct inputs drop back to NHWC.
                    if (conv_reaches_output_undamped(n)) { resident = false; break; }
                }
            (resident ? resident_idx_ : nhwc_idx_).push_back(t);
        }
        if (opts_.profile) {
            fprintf(stderr, "[rocket] conv->conv NCHW-resident intermediates: %zu/%zu internal:",
                    resident_idx_.size(), internal_idx_.size());
            for (int t : resident_idx_) {
                const TfLiteTensor &tt = context->tensors[t];
                int cc = (tt.dims && tt.dims->size == 4) ? tt.dims->data[3] : -1;
                fprintf(stderr, " t%d(C%d)", t, cc);
            }
            fprintf(stderr, "\n");
        }

        // Size the storage now that shapes are resolved (allocate_tensors has run). Bound in
        // Eval (NHWC ones onto context->tensors; resident ones used directly), not here —
        // TFLite may reset the arena data pointers between Prepare and Invoke.
        internal_buf_.assign(nhwc_idx_.size(), {});
        for (size_t k = 0; k < nhwc_idx_.size(); k++) {
            const TfLiteTensor &t = context->tensors[nhwc_idx_[k]];
            size_t bytes = tensor_byte_size(t);
            if (bytes == 0) return kTfLiteError;
            internal_buf_[k].resize(bytes);
        }
        resident_buf_.assign(resident_idx_.size(), {});
        for (size_t k = 0; k < resident_idx_.size(); k++) {
            const TfLiteTensor &t = context->tensors[resident_idx_[k]];
            if (!t.dims || t.dims->size != 4) return kTfLiteError;
            const size_t H = t.dims->data[1], W = t.dims->data[2], C = t.dims->data[3];
            if (C == 0 || H == 0 || W == 0) return kTfLiteError;
            resident_buf_[k].assign(C * H * W, (_Float16)0);   // fp16-NCHW [C][H][W]
        }
        return kTfLiteOk;
    }

    TfLiteStatus Eval(TfLiteContext *context) {
        // Bind our NHWC buffers onto the partition-internal tensors TFLite leaves
        // unallocated (see Init). Done every Eval because TFLite owns the tensor structs and
        // may reset .data between invokes. The conv-resident (fp16-NCHW) internals are NOT
        // bound here — the producer/consumer convs read/write resident_buf_ directly.
        for (size_t k = 0; k < nhwc_idx_.size(); k++)
            context->tensors[nhwc_idx_[k]].data.data = internal_buf_[k].data();

        int op_i = 0;
        for (Node &n : nodes_) {
            // Print the op as it STARTS (not just after it succeeds) so a failing op is
            // visible — the success-only timing print below can't name the op that died.
            if (opts_.profile) {
                if (n.kind == NodeKind::Conv)
                    fprintf(stderr, "[rocket] op %d: %s K=%dx%d IC=%d OC=%d s=%dx%d %s ...\n",
                            op_i, n.conv.depthwise ? "dwconv" : "conv",
                            n.conv.KH, n.conv.KW, n.conv.IC, n.conv.OC, n.conv.sy, n.conv.sx,
                            n.conv.is_quant ? "quant" : "float");
                else
                    fprintf(stderr, "[rocket] op %d: %s %s ...\n", op_i, node_op_name(n),
                            node_loc_tag(n));
            }
            const auto t0 = std::chrono::steady_clock::now();
            TfLiteStatus st = kTfLiteOk;
            switch (n.kind) {
            case NodeKind::Conv:
                if (!n.conv.packed && pack_weights(context, n.conv) != kTfLiteOk) {
                    fprintf(stderr, "[rocket] op %d: pack_weights FAILED\n", op_i);
                    return kTfLiteError;
                }
                st = eval_node(context, n.conv);
                break;
            case NodeKind::Add:     st = eval_add(context, n.add);       break;
            case NodeKind::Pool:    st = eval_pool(context, n.pool);     break;
            case NodeKind::Concat:  st = eval_concat(context, n.concat); break;
            case NodeKind::Reshape: st = eval_reshape(context, n.reshape); break;
            case NodeKind::Act:     st = eval_act(context, n.unary);      break;
            case NodeKind::FC:      st = eval_fc(context, n.fc);          break;
            case NodeKind::Prelu:   st = eval_prelu(context, n.prelu);    break;
            case NodeKind::Reduce:  st = eval_reduce(context, n.reduce);  break;
            case NodeKind::Resize:  st = eval_resize(context, n.resize);  break;
            case NodeKind::TConv:   st = eval_tconv(context, n.tconv);    break;
            case NodeKind::L2Norm:  st = eval_l2norm(context, n.l2norm);  break;
            case NodeKind::LogSoftmax: st = eval_logsoftmax(context, n.seq); break;
            case NodeKind::Softmax: st = eval_softmax(context, n.seq); break;
            case NodeKind::Cumsum:  st = eval_cumsum(context, n.seq);     break;
            case NodeKind::Transpose: st = eval_transpose(context, n.transpose); break;
            case NodeKind::Pad:     st = eval_pad(context, n.pad);        break;
            case NodeKind::Slice:   st = eval_slice(context, n.slice);    break;
            case NodeKind::Split:   st = eval_split(context, n.split);    break;
            }
            if (st != kTfLiteOk) {
                fprintf(stderr, "[rocket] op %d (%s) FAILED to eval\n", op_i,
                        n.kind == NodeKind::Conv
                            ? (n.conv.depthwise ? "dwconv" : "conv") : node_op_name(n));
                return st;
            }
            op_i++;
            if (opts_.profile) {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (n.kind == NodeKind::Conv)
                    fprintf(stderr, "[rocket] %s K=%dx%d IC=%d OC=%d s=%dx%d  %.3f ms\n",
                            n.conv.depthwise ? "dwconv" : "conv",
                            n.conv.KH, n.conv.KW, n.conv.IC, n.conv.OC, n.conv.sy, n.conv.sx, ms);
                else
                    fprintf(stderr, "[rocket] %s %s  %.3f ms\n", node_op_name(n),
                            node_loc_tag(n), ms);
            }
        }
        return kTfLiteOk;
    }

private:
    // ----- Routing predicates and partition-graph helpers -----

    // Will this (float) conv run on the resident prepacked matmul at spatial M=IH*IW?
    // A 1x1 stride-1 (no dilation) pointwise whose (M,IC,OC) meets the matmul alignment.
    // Decided in Prepare (where the weights are packed); Eval routes on mm_w, not this.
    static bool conv_is_matmul(const ConvNode &c, int M) {
        return !c.depthwise && !c.is_quant &&
               c.KH == 1 && c.KW == 1 && c.sy == 1 && c.sx == 1 &&
               c.dy == 1 && c.dx == 1 && dims_offloadable(M, c.IC, c.OC);
    }

    static int i8_round32(int x) { return (x + 31) / 32 * 32; }

    // Will this native int8/uint8 1x1 conv run on the resident int8 matmul at M=IH*IW?
    // A 1x1 stride-1 (no dilation) native int8 (c.native) or uint8 (c.native_u8) DIRECT
    // conv whose M meets the matmul's M%4||M==1 alignment. K (IC) and N (OC) are zero-padded
    // to %32, so only M gates here (a 5x5/3x3 SSD-head feature with M=25/9 stays on the
    // conv path — tiny, ~14M of the 588M 1x1 MACs). Decided in Prepare; Eval routes on mm_w_i8.
    static bool conv_is_matmul_i8(const ConvNode &c, int M) {
        return (c.native || c.native_u8) &&
               c.KH == 1 && c.KW == 1 && c.sy == 1 && c.sx == 1 &&
               c.dy == 1 && c.dx == 1 && (M % 4 == 0 || M == 1) && M >= 1;
    }

    static const char *node_kind_name(NodeKind k) {
        switch (k) {
        case NodeKind::Conv:    return "conv";
        case NodeKind::Add:     return "add";
        case NodeKind::Pool:    return "pool";
        case NodeKind::Concat:  return "concat";
        case NodeKind::Reshape: return "reshape";
        case NodeKind::Act:     return "act";
        case NodeKind::FC:      return "fc";
        case NodeKind::Prelu:   return "prelu";
        case NodeKind::Reduce:  return "reduce";
        case NodeKind::Resize:  return "resize";
        case NodeKind::TConv:   return "tconv";
        case NodeKind::L2Norm:  return "l2norm";
        case NodeKind::LogSoftmax: return "logsoftmax";
        case NodeKind::Softmax: return "softmax";
        case NodeKind::Cumsum:  return "cumsum";
        case NodeKind::Transpose: return "transpose";
        case NodeKind::Pad:     return "pad";
        case NodeKind::Slice:   return "slice";
        case NodeKind::Split:   return "split";
        }
        return "?";
    }

    // Profile label that distinguishes the binary ops sharing NodeKind::Add.
    static const char *node_op_name(const Node &n) {
        if (n.kind == NodeKind::Add && n.add.op == ROCKET_BINOP_MAX) return "max";
        if (n.kind == NodeKind::Add && n.add.op == ROCKET_BINOP_MIN) return "min";
        if (n.kind == NodeKind::Add && n.add.op == ROCKET_ARITH_MUL) return "mul";
        if (n.kind == NodeKind::Add && n.add.op == ROCKET_ARITH_SUB) return "sub";
        if (n.kind == NodeKind::Add && n.add.op == ROCKET_ARITH_DIV) return "div";
        if (n.kind == NodeKind::Reduce)
            return n.reduce.op == ROCKET_REDUCE_MEAN ? "reduce_mean"
                 : n.reduce.op == ROCKET_REDUCE_MAX  ? "reduce_max" : "reduce_min";
        if (n.kind == NodeKind::Resize)
            return n.resize.bilinear ? "resize_bilinear" : "resize_nearest";
        return node_kind_name(n.kind);
    }

    // Where a (non-conv) aux op runs, for the profile line. Everything is a host kernel
    // except a unary activation under act_npu with a device (the on-NPU DPU LUT path).
    const char *node_loc_tag(const Node &n) const {
        if (n.kind == NodeKind::Act && opts_.act_npu && fd_ >= 0) return "(npu-lut)";
        if (n.kind == NodeKind::Pool && opts_.pool_npu && fd_ >= 0) return "(npu-ppu)";
        if (n.kind == NodeKind::Add && n.add.op != -1 && opts_.ew_npu && fd_ >= 0 &&
            !n.add.is_quant) return "(npu-ew)";
        if (n.kind == NodeKind::Prelu && opts_.act_npu && fd_ >= 0 &&
            !n.prelu.is_quant) return "(npu-ew)";
        if (n.kind == NodeKind::Reduce && opts_.pool_npu && fd_ >= 0 &&
            !n.reduce.is_quant) return "(npu-ppu)";
        if (n.kind == NodeKind::Resize && opts_.resize_npu && fd_ >= 0 &&
            !n.resize.is_quant && !n.resize.bilinear) return "(npu)";
        if (n.kind == NodeKind::L2Norm && opts_.norm_npu && fd_ >= 0 &&
            !n.l2norm.is_quant) return "(npu)";
        if (n.kind == NodeKind::FC) return n.fc.mm_w ? "(npu-matmul)" : "(host)";
        return "(host)";
    }

    // The single output tensor index of a claimed node (every claimed op has one output).
    static int node_output_idx(const Node &n) {
        switch (n.kind) {
        case NodeKind::Conv:    return n.conv.out_idx;
        case NodeKind::Add:     return n.add.out;
        case NodeKind::Pool:    return n.pool.out;
        case NodeKind::Concat:  return n.concat.out;
        case NodeKind::Reshape: return n.reshape.out;
        case NodeKind::Act:     return n.unary.out;
        case NodeKind::FC:      return n.fc.out_idx;
        case NodeKind::Prelu:   return n.prelu.out;
        case NodeKind::Reduce:  return n.reduce.out;
        case NodeKind::Resize:  return n.resize.out;
        case NodeKind::TConv:   return n.tconv.out;
        case NodeKind::L2Norm:  return n.l2norm.out;
        case NodeKind::LogSoftmax:
        case NodeKind::Softmax:
        case NodeKind::Cumsum:  return n.seq.out;
        case NodeKind::Transpose: return n.transpose.out;
        case NodeKind::Pad:     return n.pad.out;
        case NodeKind::Slice:   return n.slice.out;
        case NodeKind::Split:   return n.split.outs.empty() ? -1 : n.split.outs[0];
        }
        return -1;
    }

    // ALL output tensor indices of a node. One for every kind except SPLIT, the only
    // multi-output op — used by the partition-internal-tensor enumeration so a SPLIT's
    // 2nd+ outputs that are consumed inside the partition get a buffer bound, too.
    static void node_outputs(const Node &n, std::vector<int> &v) {
        if (n.kind == NodeKind::Split) { for (int o : n.split.outs) if (o >= 0) v.push_back(o); return; }
        const int o = node_output_idx(n); if (o >= 0) v.push_back(o);
    }

    // Does node n read tensor index t as an activation input? (Conv weights/bias are
    // model constants, never partition-internal, so only the conv's in_idx counts here.)
    static bool node_reads(const Node &n, int t) {
        switch (n.kind) {
        case NodeKind::Conv:    return n.conv.in_idx == t;
        case NodeKind::Add:     return n.add.in0 == t || n.add.in1 == t;
        case NodeKind::Pool:    return n.pool.in == t;
        case NodeKind::Concat:  for (int i : n.concat.ins) if (i == t) return true; return false;
        case NodeKind::Reshape: return n.reshape.in == t;
        case NodeKind::Act:     return n.unary.in == t;
        case NodeKind::FC:      return n.fc.in_idx == t;
        case NodeKind::Prelu:   return n.prelu.in == t;   // alpha is a model constant
        case NodeKind::Reduce:  return n.reduce.in == t;  // axis is a model constant
        case NodeKind::Resize:  return n.resize.in == t;  // size is a model constant
        case NodeKind::TConv:   return n.tconv.in == t;   // weights/bias/shape are constants
        case NodeKind::L2Norm:  return n.l2norm.in == t;
        case NodeKind::LogSoftmax:
        case NodeKind::Softmax:
        case NodeKind::Cumsum:  return n.seq.in == t;     // axis is a model constant
        case NodeKind::Transpose: return n.transpose.in == t;  // perm is a model constant
        case NodeKind::Pad:     return n.pad.in == t;          // paddings/value are constants
        case NodeKind::Slice:   return n.slice.in == t;        // begin/size are constants
        case NodeKind::Split:   return n.split.in == t;        // axis/size_splits are constants
        }
        return false;
    }

    // Does tensor index t reach a PARTITION OUTPUT through only "transparent" ops —
    // RESHAPE / CONCAT, the shape/join ops that carry values straight through to the
    // predictor outputs — with NO requant barrier (another conv / ADD / POOL) in between?
    // Used by the re-quantization-barrier rule to find the convs whose result lands
    // on a partition output undamped. The partition graph is a DAG (nodes_ is topological),
    // so the recursion terminates; partitions are tens of ops so the walk is cheap.
    bool tensor_reaches_output_undamped(int t) const {
        for (int o : part_out_) if (o == t) return true;
        for (const Node &n : nodes_) {
            if (!node_reads(n, t)) continue;
            // RESHAPE / CONCAT and the single-output layout ops (TRANSPOSE / PAD / SLICE)
            // carry values through unchanged (no requant), so drift on t reaches the output
            // undamped through them. SPLIT (multi-output) is intentionally NOT transparent
            // here — treating it as a barrier only forces a consuming conv back to NHWC,
            // which is the safe direction.
            if ((n.kind == NodeKind::Reshape || n.kind == NodeKind::Concat ||
                 n.kind == NodeKind::Transpose || n.kind == NodeKind::Pad ||
                 n.kind == NodeKind::Slice) &&
                tensor_reaches_output_undamped(node_output_idx(n)))
                return true;
            // Conv/Add/Pool consumers re-quantize their output -> they damp t's drift; not
            // transparent, so this branch does not propagate "undamped".
        }
        return false;
    }
    // A conv node is an "output predictor": its result reaches a partition output undamped.
    bool conv_reaches_output_undamped(const Node &n) const {
        return n.kind == NodeKind::Conv && tensor_reaches_output_undamped(n.conv.out_idx);
    }

    // The resident fp16-NCHW buffer for tensor index t, or null if t is not conv-resident.
    // Linear scan — partitions are small (tens of ops).
    _Float16 *resident_ptr(int t) {
        for (size_t k = 0; k < resident_idx_.size(); k++)
            if (resident_idx_[k] == t) return resident_buf_[k].data();
        return nullptr;
    }

    // Byte size of a tensor from its (resolved) dims + element type.
    static size_t tensor_byte_size(const TfLiteTensor &t) {
        if (!t.dims) return 0;
        const long long nc = safe_elem_count(t.dims);   // rejects negative/overflowing dims
        if (nc < 0) return 0;
        const size_t n = (size_t)nc;
        switch (t.type) {
        case kTfLiteFloat32: return n * 4;
        case kTfLiteInt32:   return n * 4;
        case kTfLiteInt8:
        case kTfLiteUInt8:   return n * 1;
        default:             return 0;     // unsupported intermediate type
        }
    }

    // ----- Weight packing (filter -> driver layout, once in Prepare) -----

    TfLiteStatus pack_weights(TfLiteContext *context, ConvNode &c) {
        const TfLiteTensor &flt = context->tensors[c.w_idx];
        if (!flt.data.data) return kTfLiteError;     // non-constant filter (unsupported)

        // NATIVE int8: keep the raw int8 filter in driver layout [OC][IC][KH][KW]
        // and fold the TFLite int32 bias + the input zero-point correction into eff_bias.
        // No fp16 dequant — rocket_conv2d_int8 multiplies the raw int8 weights directly.
        if (c.native) {
            c.w_i8.resize((size_t)c.OC * c.IC * c.KH * c.KW);
            rocket_filter_i8_to_oihw((const signed char *)flt.data.data, c.w_i8.data(),
                                     c.OC, c.KH, c.KW, c.IC);
            const int32_t *bq = nullptr;
            if (c.bias_idx >= 0) {
                bq = context->tensors[c.bias_idx].data.i32;
                if (!bq) return kTfLiteError;
            }
            c.eff_bias.resize(c.OC);
            rocket_eff_bias_per_axis(c.w_i8.data(), bq, c.in_zp, c.eff_bias.data(),
                                     c.OC, c.IC, c.KH, c.KW);
            c.packed = true;
            return kTfLiteOk;
        }

        // NATIVE uint8 DIRECT (Option D): recenter the uint8 filter to int8 (y=w_q-128) in
        // driver layout, and fold TFLite's int32 bias + the centering constants
        // (alpha*Wy + N*alpha*beta) into eff_bias. The per-pixel box-sum correction is added
        // per inference in Eval (it depends on the activations). w_zp is kept for the box-sum.
        if (c.native_u8) {
            c.w_i8.resize((size_t)c.OC * c.IC * c.KH * c.KW);
            rocket_filter_u8_to_oihw((const unsigned char *)flt.data.data, c.w_i8.data(),
                                     c.OC, c.KH, c.KW, c.IC);
            const int32_t *bq = nullptr;
            if (c.bias_idx >= 0) {
                bq = context->tensors[c.bias_idx].data.i32;
                if (!bq) return kTfLiteError;
            }
            c.eff_bias.resize(c.OC);
            rocket_eff_bias_u8_per_axis(c.w_i8.data(), bq, c.in_zp, c.w_zp.data(),
                                        c.eff_bias.data(), c.OC, c.IC, c.KH, c.KW);
            c.packed = true;
            return kTfLiteOk;
        }

        // NATIVE int8 DEPTHWISE (int8-out on-chip requant): raw int8 filter
        // [C][KH][KW] + the raw TFLite int32 bias (the driver folds Mesa's zero-point
        // correction itself). Per-tensor quant; OC==IC==C.
        if (c.native_dw) {
            c.w_i8.resize((size_t)c.OC * c.KH * c.KW);
            rocket_dw_filter_i8_to_chw((const signed char *)flt.data.data, c.w_i8.data(),
                                       c.OC, c.KH, c.KW);
            c.bias_q.assign(c.OC, 0);
            if (c.bias_idx >= 0) {
                const int32_t *bq = context->tensors[c.bias_idx].data.i32;
                if (!bq) return kTfLiteError;
                for (int i = 0; i < c.OC; i++) c.bias_q[i] = bq[i];
            }
            c.packed = true;
            return kTfLiteOk;
        }

        // direct: [OC][IC][KH][KW]; depthwise: [C][KH][KW] (one filter per channel,
        // OC==IC==C). The driver does its own (C/G,KH,KW,G) cube scatter from there.
        c.w.resize(c.depthwise ? (size_t)c.OC * c.KH * c.KW
                               : (size_t)c.OC * c.IC * c.KH * c.KW);
        if (c.is_quant) {
            // dequant the int8/uint8 filter per channel into the driver's fp16 layout
            if (c.depthwise)
                rocket_dw_filter_q_to_chw(flt.data.data, c.w_uns, c.w_scale.data(),
                                          c.w_zp.data(), c.w.data(), c.OC, c.KH, c.KW);
            else
                rocket_filter_q_to_oihw(flt.data.data, c.w_uns, c.w_scale.data(),
                                        c.w_zp.data(), c.w.data(), c.OC, c.KH, c.KW, c.IC);
            if (c.bias_idx >= 0) {
                const int32_t *bq = context->tensors[c.bias_idx].data.i32;
                if (!bq) return kTfLiteError;
                c.bias_f.resize(c.OC);
                rocket_dequant_bias(bq, c.in_scale, c.w_scale.data(), c.bias_f.data(), c.OC);
            }
        } else {
            if (c.depthwise)
                rocket_dw_filter_hwc_to_chw(flt.data.f, c.w.data(), c.OC, c.KH, c.KW);
            else
                rocket_filter_ohwi_to_oihw(flt.data.f, c.w.data(), c.OC, c.KH, c.KW, c.IC);
        }
        c.packed = true;
        return kTfLiteOk;
    }

    // ----- Eval: conv / depthwise / pointwise-matmul (NPU) -----

    TfLiteStatus eval_node(TfLiteContext *context, ConvNode &c) {
        const TfLiteTensor &in   = context->tensors[c.in_idx];
        const TfLiteTensor &outp = context->tensors[c.out_idx];

        const int IH = in.dims->data[1],   IW = in.dims->data[2],  IC = in.dims->data[3];
        const int OH = outp.dims->data[1], OW = outp.dims->data[2], OC = outp.dims->data[3];
        if (IC != c.IC || OC != c.OC) return kTfLiteError;   // filter/shape mismatch

        // --- NATIVE int8/uint8 1x1 pointwise via the RESIDENT multicore int8 MATMUL (perf
        //     Step 1). A 1x1 IS a matmul C[M,OC] = A[M,IC] * B[OC,IC]^T over the M=H*W
        //     positions; the NHWC input is already A row-major and C is already NHWC, so no
        //     transpose on either side, and the matmul fans the OUTPUT columns across the 3
        //     cores (the single-tile conv path pins them to ONE). mm_w_i8 is non-null only when
        //     Prepare packed this conv's int8 weights for the current M (1x1, M%4||M==1, device
        //     present, scratch free). The int8 accumulate is exact + order-independent: the matmul
        //     and the conv path are both bit-exact to the int64 reference == CPU TFLite and
        //     deterministic across worker counts (tests/mm_vs_conv_acc.c, tests/mm_nt_det.c), so
        //     routing here is the throughput choice above (3-core fan-out), not a correctness one.
        //     K/N are zero-padded to %32 (Kp/Np), so C's row stride is Np with only OC cols real. ---
        if ((c.native || c.native_u8) && c.mm_w_i8 && IH * IW == c.mm_i8_M) {
            unsigned char *out_q = (unsigned char *)outp.data.data;
            if (!in.data.data || !out_q) return kTfLiteError;
            const int M = IH * IW, Kp = i8_round32(IC), Np = i8_round32(OC);
            std::vector<int8_t>  A;             // recentered (u8) / K-padded (i8) activation
            std::vector<int32_t> C32((size_t)M * Np);
            std::vector<int32_t> Sx;            // u8 asymmetric-weight per-pixel box-sum
            int32_t *Sxp = nullptr;
            if (c.native_u8 && c.needs_boxsum) { Sx.resize(M); Sxp = Sx.data(); }
            const int8_t  *Aptr;
            const auto ts0 = std::chrono::steady_clock::now();
            if (c.native_u8) {                  // recenter x=in_q-128 into [M][Kp] (+box-sum)
                A.resize((size_t)M * Kp);
                parallel_oh(M, (size_t)M * IC, opts_.nthreads, [&](int m0, int m1) {
                    rocket_recenter_u8_mk_band((const unsigned char *)in.data.data, A.data(),
                            Sxp, M, IC, Kp, m0, m1);
                });
                Aptr = A.data();
            } else if (Kp != IC) {              // signed int8, K needs zero-padding to %32
                A.resize((size_t)M * Kp);
                parallel_oh(M, (size_t)M * IC, opts_.nthreads, [&](int m0, int m1) {
                    rocket_pad_i8_mk_band((const signed char *)in.data.data, A.data(),
                            M, IC, Kp, m0, m1);
                });
                Aptr = A.data();
            } else {                            // signed int8, IC%32==0 -> feed the input directly
                Aptr = (const int8_t *)in.data.data;
            }
            const auto ts1 = std::chrono::steady_clock::now();
            if (rocket_matmul_int8_prepacked(mm_i8_ctx_, M, Kp, Np, Aptr, C32.data(),
                                             c.mm_w_i8) != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   rocket_matmul_int8_prepacked FAILED "
                            "(M=%d Kp=%d Np=%d IC=%d OC=%d)\n", M, Kp, Np, IC, OC);
                return kTfLiteError;
            }
            const auto ts2 = std::chrono::steady_clock::now();
            if (c.native_u8)
                parallel_oh(M, (size_t)M * OC, opts_.nthreads, [&](int m0, int m1) {
                    rocket_out_mn_to_nhwc_q_per_axis_u8_band(C32.data(), out_q, M, OC, Np,
                            c.eff_bias.empty() ? nullptr : c.eff_bias.data(),
                            c.w_zp.data(), Sxp, c.act, c.in_scale, c.w_scale.data(),
                            c.out_scale, c.out_zp, m0, m1);
                });
            else
                parallel_oh(M, (size_t)M * OC, opts_.nthreads, [&](int m0, int m1) {
                    rocket_out_mn_to_nhwc_q_per_axis_band(C32.data(), out_q, 0, M, OC, Np,
                            c.eff_bias.empty() ? nullptr : c.eff_bias.data(),
                            c.act, c.in_scale, c.w_scale.data(), c.out_scale, c.out_zp, m0, m1);
                });
            if (opts_.profile) {
                using msd = std::chrono::duration<double, std::milli>;
                const auto ts3 = std::chrono::steady_clock::now();
                fprintf(stderr, "[rocket]   breakdown in=%.3f mm=%.3f out=%.3f ms [mm i8%s]\n",
                        msd(ts1 - ts0).count(), msd(ts2 - ts1).count(), msd(ts3 - ts2).count(),
                        c.native_u8 ? (c.needs_boxsum ? "+box(u8)" : "(u8sym)") : "");
            }
            return kTfLiteOk;
        }

        // --- NATIVE int8: int8 x int8 -> int32 on the NPU + host per-axis requant.
        //     Materialize the int8 input NCHW with pad = input zero-point (the eff_bias
        //     correction cancels it -> exact TFLite boundary), run rocket_conv2d_int8
        //     (degenerate-conv handles 1x1), requant the int32 accumulator. EXACT int8,
        //     no fp16 approximation, no host dequant/requant of the operands. ---
        if (c.native) {
            const signed char *in_q  = (const signed char *)in.data.data;
            signed char       *out_q = (signed char *)outp.data.data;
            if (!in_q || !out_q) return kTfLiteError;
            const int tot_h = rocket_total_pad(IH, c.KH, c.sy, c.dy, OH);
            const int tot_w = rocket_total_pad(IW, c.KW, c.sx, c.dx, OW);
            const int IHp = IH + tot_h, IWp = IW + tot_w;
            rocket_conv2d_desc d = {};
            d.ic = IC; d.ih = IHp; d.iw = IWp; d.oc = OC;
            d.kh = c.KH; d.kw = c.KW; d.stride_y = c.sy; d.stride_x = c.sx;
            d.pad_top = 0; d.pad_left = 0; d.dil_y = c.dy; d.dil_x = c.dx; d.depthwise = 0;
            if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) return kTfLiteError;
            std::vector<int8_t>  in_nchw((size_t)IC * IHp * IWp);
            std::vector<int32_t> out_nchw((size_t)OC * OH * OW);
            const auto ts0 = std::chrono::steady_clock::now();
            rocket_in_i8_to_nchw_pad(in_q, in_nchw.data(), IC, IH, IW, c.in_zp,
                                     tot_h / 2, tot_w / 2, IHp, IWp);
            const auto ts1 = std::chrono::steady_clock::now();
            int rc = conv_pool_
                ? rocket_conv2d_int8_mt(conv_pool_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data())
                : conv_ctx_
                ? rocket_conv2d_int8_ctx(conv_ctx_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data())
                : rocket_conv2d_int8(fd_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data());
            if (rc != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   rocket_conv2d_int8 ret=%d (IC=%d OC=%d IHp=%d "
                            "IWp=%d K=%dx%d s=%dx%d)\n", rc, IC, OC, IHp, IWp,
                            c.KH, c.KW, c.sy, c.sx);
                return kTfLiteError;
            }
            const auto ts2 = std::chrono::steady_clock::now();
            parallel_oh(OH, (size_t)OH * OW * OC, opts_.nthreads, [&](int oh0, int oh1) {
                rocket_out_nchw_to_nhwc_q_per_axis_band(out_nchw.data(), out_q, 0, OC, OH, OW,
                        c.eff_bias.empty() ? nullptr : c.eff_bias.data(),
                        c.act, c.in_scale, c.w_scale.data(), c.out_scale, c.out_zp, oh0, oh1);
            });
            if (opts_.profile) {
                using msd = std::chrono::duration<double, std::milli>;
                const auto ts3 = std::chrono::steady_clock::now();
                fprintf(stderr, "[rocket]   breakdown in=%.3f conv=%.3f out=%.3f ms [native i8]\n",
                        msd(ts1 - ts0).count(), msd(ts2 - ts1).count(), msd(ts3 - ts2).count());
            }
            return kTfLiteOk;
        }

        // --- NATIVE uint8 (uint8 recenter path): recenter the uint8 operands to int8 (x=in_q-128,
        //     y=w_q-128), run the SAME int32-raw NPU conv, then on the host add the box-sum
        //     correction (only when w_zp != 128) and requant to uint8. eff_bias already folds
        //     the per-OC centering constants (Prepare). EXACT uint8, no fp16 approximation. ---
        if (c.native_u8) {
            const unsigned char *in_q  = (const unsigned char *)in.data.data;
            unsigned char       *out_q = (unsigned char *)outp.data.data;
            if (!in_q || !out_q) return kTfLiteError;
            const int tot_h = rocket_total_pad(IH, c.KH, c.sy, c.dy, OH);
            const int tot_w = rocket_total_pad(IW, c.KW, c.sx, c.dx, OW);
            const int IHp = IH + tot_h, IWp = IW + tot_w;
            rocket_conv2d_desc d = {};
            d.ic = IC; d.ih = IHp; d.iw = IWp; d.oc = OC;
            d.kh = c.KH; d.kw = c.KW; d.stride_y = c.sy; d.stride_x = c.sx;
            d.pad_top = 0; d.pad_left = 0; d.dil_y = c.dy; d.dil_x = c.dx; d.depthwise = 0;
            if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) return kTfLiteError;
            std::vector<int8_t>  in_nchw((size_t)IC * IHp * IWp);
            std::vector<int32_t> out_nchw((size_t)OC * OH * OW);
            const auto ts0 = std::chrono::steady_clock::now();
            rocket_in_u8_to_nchw_pad(in_q, in_nchw.data(), IC, IH, IW, c.in_zp,
                                     tot_h / 2, tot_w / 2, IHp, IWp);
            std::vector<int32_t> boxsum;
            const int32_t *Sx = nullptr;
            const auto ts1 = std::chrono::steady_clock::now();
            int rc = conv_pool_
                ? rocket_conv2d_int8_mt(conv_pool_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data())
                : conv_ctx_
                ? rocket_conv2d_int8_ctx(conv_ctx_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data())
                : rocket_conv2d_int8(fd_, &d, in_nchw.data(), c.w_i8.data(), out_nchw.data());
            const auto ts2 = std::chrono::steady_clock::now();
            // box-sum (the asymmetric-weight-zp correction, reads only the input): fan across
            // the big cores by row band. NOT overlapped with the conv — the conv's tile workers
            // already use all 4 big cores, so an overlapping box-sum thread just contends with
            // them (it inflated conv ~16%); running it clean after the conv is faster.
            if (c.needs_boxsum) {
                boxsum.resize((size_t)OH * OW);
                Sx = boxsum.data();
                parallel_oh(OH, (size_t)OH * OW * IC * c.KH * c.KW, opts_.nthreads,
                            [&](int oh0, int oh1) {
                    rocket_in_window_sum_i8_band(in_nchw.data(), boxsum.data(), IC, IHp, IWp,
                            OH, OW, c.KH, c.KW, c.sy, c.sx, c.dy, c.dx, oh0, oh1);
                });
            }
            const auto tsb = std::chrono::steady_clock::now();
            if (rc != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   rocket_conv2d_int8(u8) ret=%d (IC=%d OC=%d IHp=%d "
                            "IWp=%d K=%dx%d s=%dx%d)\n", rc, IC, OC, IHp, IWp,
                            c.KH, c.KW, c.sy, c.sx);
                return kTfLiteError;
            }
            // requant is the single largest host cost -> fan across the big cores by row band.
            parallel_oh(OH, (size_t)OH * OW * OC, opts_.nthreads, [&](int oh0, int oh1) {
                rocket_out_nchw_to_nhwc_q_per_axis_u8_band(out_nchw.data(), out_q, OC, OH, OW,
                        c.eff_bias.empty() ? nullptr : c.eff_bias.data(),
                        c.w_zp.data(), Sx, c.act, c.in_scale,
                        c.w_scale.data(), c.out_scale, c.out_zp, oh0, oh1);
            });
            if (opts_.profile) {
                using msd = std::chrono::duration<double, std::milli>;
                const auto ts3 = std::chrono::steady_clock::now();
                fprintf(stderr, "[rocket]   breakdown in=%.3f conv=%.3f box=%.3f req=%.3f ms [native u8%s]\n",
                        msd(ts1 - ts0).count(), msd(ts2 - ts1).count(), msd(tsb - ts2).count(),
                        msd(ts3 - tsb).count(), c.needs_boxsum ? "+box" : "");
            }
            return kTfLiteOk;
        }

        // --- NATIVE int8 DEPTHWISE (int8-out on-chip requant). Per-tensor quant,
        //     symmetric pad (validated config: the CNA's symmetric HW pad). Plain int8
        //     transpose in/out; the runtime does the uint8-domain centering + on-chip
        //     requant + the zero-point bias fold. Bit-exact to Teflon ground truth. ---
        if (c.native_dw) {
            const signed char *in_q  = (const signed char *)in.data.data;
            signed char       *out_q = (signed char *)outp.data.data;
            if (!in_q || !out_q) return kTfLiteError;
            const int tot_h = rocket_total_pad(IH, c.KH, c.sy, c.dy, OH);
            const int tot_w = rocket_total_pad(IW, c.KW, c.sx, c.dx, OW);
            // Re-check the symmetric-pad gate on LIVE shapes. analyze_conv gated
            // native_dw on even tot_h/tot_w at Init, but a resize can make them odd;
            // d.pad_top/left = tot/2 (floor) would then feed an ASYMMETRIC SAME pad to
            // the symmetric-only HW config -> silently wrong output. The external
            // delegate can't un-delegate at Eval, so fail the Invoke (matches the
            // OH/OW re-check just below).
            if ((tot_h % 2) != 0 || (tot_w % 2) != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   native_dw: asymmetric pad after resize "
                            "(tot_h=%d tot_w=%d) -> kTfLiteError\n", tot_h, tot_w);
                return kTfLiteError;
            }
            rocket_conv2d_desc d = {};
            d.ic = IC; d.ih = IH; d.iw = IW; d.oc = OC;
            d.kh = c.KH; d.kw = c.KW; d.stride_y = c.sy; d.stride_x = c.sx;
            d.pad_top = tot_h / 2; d.pad_left = tot_w / 2;   // symmetric (re-checked above)
            d.dil_y = c.dy; d.dil_x = c.dx; d.depthwise = 1;
            if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) return kTfLiteError;
            std::vector<int8_t> in_nchw((size_t)IC * IH * IW);
            std::vector<int8_t> out_nchw((size_t)OC * OH * OW);
            rocket_in_i8_to_nchw_pad(in_q, in_nchw.data(), IC, IH, IW, c.in_zp, 0, 0, IH, IW);
            const int32_t *bq = c.bias_q.empty() ? nullptr : c.bias_q.data();
            int rc = conv_ctx_
                ? rocket_conv2d_dw_int8_ctx(conv_ctx_, &d, in_nchw.data(), c.w_i8.data(), bq,
                      c.in_scale, c.w_scale[0], c.out_scale, c.in_zp, c.w_zp[0], c.out_zp,
                      out_nchw.data())
                : rocket_conv2d_dw_int8(fd_, &d, in_nchw.data(), c.w_i8.data(), bq,
                      c.in_scale, c.w_scale[0], c.out_scale, c.in_zp, c.w_zp[0], c.out_zp,
                      out_nchw.data());
            if (rc != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   rocket_conv2d_dw_int8 ret=%d (C=%d IH=%d IW=%d "
                            "K=%dx%d s=%dx%d)\n", rc, IC, IH, IW, c.KH, c.KW, c.sy, c.sx);
                return kTfLiteError;
            }
            rocket_out_nchw_to_nhwc_i8(out_nchw.data(), out_q, OC, OH, OW);
            return kTfLiteOk;
        }

        // --- float fast path: 1x1 stride-1 pointwise via the RESIDENT prepacked matmul.
        //     mm_w is non-null only when Prepare packed this conv's weights for the
        //     current M (matmul-aligned, device present, scratch available); the only
        //     per-call NPU cost is then the A-pack inside rocket_matmul_fp16_prepacked.
        //     Anything else (non-aligned 1x1, resized M, no device) falls through to the
        //     general conv path below, which runs the 1x1 as a degenerate conv. ---
        if (!c.is_quant && c.mm_w && IH * IW == c.mm_M) {
            const float *A    = in.data.f;
            float       *C    = outp.data.f;
            const float *bias = (c.bias_idx >= 0) ? context->tensors[c.bias_idx].data.f : nullptr;
            if (!A || !C) return kTfLiteError;
            const int M = IH * IW;
            std::vector<_Float16> A16((size_t)M * IC), C16((size_t)M * OC);
            for (size_t i = 0; i < A16.size(); i++) A16[i] = (_Float16)A[i];
            if (rocket_matmul_fp16_prepacked(mm_ctx_, M, IC, OC, A16.data(), C16.data(),
                                             c.mm_w) != 0)
                return kTfLiteError;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < OC; n++) {
                    float v = (float)C16[(size_t)m * OC + n];
                    if (bias) v += bias[n];
                    C[(size_t)m * OC + n] = rocket_apply_act(v, c.act);
                }
            return kTfLiteOk;
        }

        // --- general conv: materialize pad -> NCHW fp16, run the driver, transpose
        //     out. The float path dequant is a no-op transpose; the quant path
        //     dequantizes int8/uint8 in and requantizes out around the same conv. ---
        const int tot_h = rocket_total_pad(IH, c.KH, c.sy, c.dy, OH);
        const int tot_w = rocket_total_pad(IW, c.KW, c.sx, c.dx, OW);
        const int IHp = IH + tot_h, IWp = IW + tot_w;
        rocket_conv2d_desc d = {};
        d.ic = IC; d.ih = IHp; d.iw = IWp; d.oc = OC;
        d.kh = c.KH; d.kw = c.KW; d.stride_y = c.sy; d.stride_x = c.sx;
        d.pad_top = 0; d.pad_left = 0; d.dil_y = c.dy; d.dil_x = c.dx;
        d.depthwise = c.depthwise ? 1 : 0;
        if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) {
            if (opts_.profile)
                fprintf(stderr, "[rocket]   OH/OW mismatch: driver(%d,%d) tflite(%d,%d) "
                        "IHp=%d IWp=%d K=%dx%d s=%dx%d\n", rocket_conv2d_oh(&d),
                        rocket_conv2d_ow(&d), OH, OW, IHp, IWp, c.KH, c.KW, c.sy, c.sx);
            return kTfLiteError;
        }

        // resident fp16-NCHW intermediates: in_res non-null => this conv's input is a
        // producer conv's resident NCHW output (read + pad it, skip dequant/transpose);
        // out_res non-null => this conv's output is consumed only by conv-path convs (write
        // fp16-NCHW + bias/act, skip transpose/requant). Both null => the usual NHWC path.
        _Float16 *in_res  = resident_ptr(c.in_idx);
        _Float16 *out_res = resident_ptr(c.out_idx);

        std::vector<_Float16> in_nchw((size_t)IC * IHp * IWp);
        std::vector<_Float16> out_local;                // only when the output is NOT resident
        _Float16 *out_nchw = out_res;
        if (!out_res) { out_local.resize((size_t)OC * OH * OW); out_nchw = out_local.data(); }

        // Sub-step timing (profile-gated): in = host input materialize, conv = the driver
        // call (cube scatter + NPU submit/wait + descatter), out = host output. A resident
        // in/out replaces the dequant+transpose / transpose+requant with a cheap NCHW
        // pad / in-place bias+act (the [res ...] tag shows which boundaries were elided).
        const auto ts0 = std::chrono::steady_clock::now();
        if (in_res) {
            rocket_nchw_pad(in_res, in_nchw.data(), IC, IH, IW, tot_h / 2, tot_w / 2, IHp, IWp);
        } else if (c.is_quant) {
            if (!in.data.data) return kTfLiteError;
            rocket_in_q_to_nchw_pad(in.data.data, c.in_uns, c.in_scale, c.in_zp,
                                    in_nchw.data(), IC, IH, IW, tot_h / 2, tot_w / 2, IHp, IWp);
        } else {
            if (!in.data.f) return kTfLiteError;
            rocket_in_nhwc_to_nchw_pad(in.data.f, in_nchw.data(), IC, IH, IW,
                                       tot_h / 2, tot_w / 2, IHp, IWp);
        }
        if (!out_res && (c.is_quant ? outp.data.data == nullptr : outp.data.f == nullptr))
            return kTfLiteError;                        // boundary output needs TFLite storage
        const auto ts1 = std::chrono::steady_clock::now();
        // fd_<0 makes the conv compute on its CPU oracle and return 0, so a nonzero return is
        // a genuine device error. conv_ctx_ reuses resident BOs across calls/tiles; it falls
        // back to the per-call path only if the pool couldn't be created (OOM in Prepare).
        {
            int rc = conv_ctx_
                ? rocket_conv2d_fp16_ctx(conv_ctx_, &d, in_nchw.data(), c.w.data(), out_nchw)
                : rocket_conv2d_fp16(fd_, &d, in_nchw.data(), c.w.data(), out_nchw);
            if (rc != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   rocket_conv2d_fp16 ret=%d (IC=%d OC=%d IHp=%d "
                            "IWp=%d K=%dx%d s=%dx%d dw=%d)\n", rc, IC, OC, IHp, IWp,
                            c.KH, c.KW, c.sy, c.sx, c.depthwise);
                return kTfLiteError;
            }
        }
        const auto ts2 = std::chrono::steady_clock::now();

        if (out_res) {
            // keep fp16-NCHW for the consumer conv(s): bias+act in place, no transpose/requant.
            const float *bias = c.is_quant
                ? (c.bias_f.empty() ? nullptr : c.bias_f.data())
                : (c.bias_idx >= 0 ? context->tensors[c.bias_idx].data.f : nullptr);
            rocket_nchw_bias_act(out_res, OC, OH, OW, bias, c.act);
        } else if (c.is_quant) {
            // the fp16-path requant (the uint8 depthwise lands here) -> fan across big cores.
            parallel_oh(OH, (size_t)OH * OW * OC, opts_.nthreads, [&](int oh0, int oh1) {
                rocket_out_nchw_to_nhwc_q_band(out_nchw, outp.data.data, c.out_uns, OC, OH, OW,
                        c.bias_f.empty() ? nullptr : c.bias_f.data(),
                        c.act, c.out_scale, c.out_zp, oh0, oh1);
            });
        } else {
            const float *bias = (c.bias_idx >= 0) ? context->tensors[c.bias_idx].data.f : nullptr;
            rocket_out_nchw_to_nhwc_bias_act(out_nchw, outp.data.f, OC, OH, OW, bias, c.act);
        }
        if (opts_.profile) {
            using msd = std::chrono::duration<double, std::milli>;
            const auto ts3 = std::chrono::steady_clock::now();
            const char *tag = (in_res && out_res) ? " [res in+out]"
                            : in_res ? " [res in]" : out_res ? " [res out]" : "";
            fprintf(stderr, "[rocket]   breakdown in=%.3f conv=%.3f out=%.3f ms%s\n",
                    msd(ts1 - ts0).count(), msd(ts2 - ts1).count(), msd(ts3 - ts2).count(), tag);
        }
        return kTfLiteOk;
    }

    // ----- Eval: aux host ops (elementwise / pooling / layout, NHWC on the CPU; see rocket_ops.h) -----

    TfLiteStatus eval_add(TfLiteContext *context, AddP &x) {
        const TfLiteTensor &a = context->tensors[x.in0];
        const TfLiteTensor &b = context->tensors[x.in1];
        const TfLiteTensor &o = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (n < 0) return kTfLiteError;                  // reject the dims_count(-1) overflow sentinel
        // Re-validate LIVE shapes: analyze_* gated this node on broadcast_to (each input
        // broadcasts to the output), but a resize could change a live shape to something
        // non-broadcastable, which the flat element pass would silently mis-compute.
        // Mirror the Init gate on the live tensors.
        if (!broadcast_to(a.dims, o.dims) || !broadcast_to(b.dims, o.dims)) return kTfLiteError;
        if (!a.data.data || !b.data.data || !o.data.data) return kTfLiteError;

        // Materialize a broadcasting operand to the output shape so the SAME-shape host
        // kernels below run unchanged (bit-exact); a same-shape operand passes through with
        // no copy. Byte-wise: int8/uint8 quant = 1 byte, float = 4.
        const int esz = x.is_quant ? 1 : (int)sizeof(float);
        std::vector<char> abuf, bbuf;
        const void *ap = a.data.data, *bp = b.data.data;
        if (!same_dims(a.dims, o.dims)) {
            abuf.resize((size_t)n * esz);
            rocket_broadcast_copy(a.data.data, abuf.data(), a.dims->data, a.dims->size,
                                  o.dims->data, o.dims->size, esz);
            ap = abuf.data();
        }
        if (!same_dims(b.dims, o.dims)) {
            bbuf.resize((size_t)n * esz);
            rocket_broadcast_copy(b.data.data, bbuf.data(), b.dims->data, b.dims->size,
                                  o.dims->data, o.dims->size, esz);
            bp = bbuf.data();
        }
        const float *apf = (const float *)ap, *bpf = (const float *)bp;

        // Two-tensor ops sharing AddP (op != -1): MAXIMUM/MINIMUM (no act) and
        // MUL/SUB/DIV (fused act). ADD (op == -1) falls through to the residual-add below.
        if (x.op != -1) {
            const bool minmax = (x.op == ROCKET_BINOP_MAX || x.op == ROCKET_BINOP_MIN);
            // MAXIMUM/MINIMUM opt-in (ew_npu + a device, FLOAT): NPU DPU EW ALU
            // (rocket_ew_max/min_fp16); any failure falls back to the exact host kernel.
            // Same-shape only — the EW ALU takes flat operands, so a broadcast (a temp was
            // materialized) routes to the host kernel below.
            if (minmax && opts_.ew_npu && fd_ >= 0 && !x.is_quant && n <= INT_MAX &&
                abuf.empty() && bbuf.empty()) {
                std::vector<_Float16> fa((size_t)n), fb((size_t)n), fo((size_t)n);
                for (long i = 0; i < n; i++) { fa[i] = (_Float16)apf[i]; fb[i] = (_Float16)bpf[i]; }
                int rc = (x.op == ROCKET_BINOP_MIN)
                    ? rocket_ew_min_fp16(fd_, fa.data(), fb.data(), fo.data(), (int)n)
                    : rocket_ew_max_fp16(fd_, fa.data(), fb.data(), fo.data(), (int)n);
                if (rc == 0) {
                    for (long i = 0; i < n; i++) o.data.f[i] = (float)fo[i];
                    return kTfLiteOk;
                }
                if (opts_.profile) fprintf(stderr, "[rocket]   ew_npu failed -> host kernel\n");
            }
            if (minmax) {
                if (!x.is_quant)
                    rocket_binary_f(apf, bpf, o.data.f, (size_t)n, x.op);
                else
                    rocket_binary_q(ap, bp, o.data.data, (size_t)n, x.op,
                                    x.in0_uns, x.in1_uns, x.out_uns,
                                    x.in0_scale, x.in0_zp, x.in1_scale, x.in1_zp,
                                    x.out_scale, x.out_zp);
            } else {              // MUL / SUB / DIV — fused-act arithmetic
                if (!x.is_quant)
                    rocket_arith_f(apf, bpf, o.data.f, (size_t)n, x.op, x.act);
                else
                    rocket_arith_q(ap, bp, o.data.data, (size_t)n, x.op,
                                   x.in0_uns, x.in1_uns, x.out_uns,
                                   x.in0_scale, x.in0_zp, x.in1_scale, x.in1_zp,
                                   x.out_scale, x.out_zp, x.act);
            }
            return kTfLiteOk;
        }

        if (!x.is_quant)
            rocket_add_f(apf, bpf, o.data.f, (size_t)n, x.act);
        else
            rocket_add_q(ap, bp, o.data.data, (size_t)n,
                         x.in0_uns, x.in1_uns, x.out_uns,
                         x.in0_scale, x.in0_zp, x.in1_scale, x.in1_zp,
                         x.out_scale, x.out_zp, x.act);
        return kTfLiteOk;
    }

    TfLiteStatus eval_pool(TfLiteContext *context, PoolP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const int IH = in.dims->data[1], IW = in.dims->data[2], C = in.dims->data[3];
        const int OH = o.dims->data[1],  OW = o.dims->data[2];
        if (o.dims->data[3] != C) return kTfLiteError;
        if (rocket_out_dim(IH, x.kh, x.sy, 1, x.same) != OH ||
            rocket_out_dim(IW, x.kw, x.sx, 1, x.same) != OW) return kTfLiteError;
        const int tot_h = rocket_total_pad(IH, x.kh, x.sy, 1, OH);
        const int tot_w = rocket_total_pad(IW, x.kw, x.sx, 1, OW);
        if (!in.data.data || !o.data.data) return kTfLiteError;

        // Opt-in (pool_npu + a device): run the pool on the NPU PPU (rocket_pool_fp16).
        // FLOAT only; AVERAGE only when VALID (no pad) — the PPU divides by KH*KW
        // (count-include-pad) while the host / TFLite divide by the valid count, so a padded
        // average diverges (MAX with pad is fine: the PPU -inf pad fill never wins ==
        // clamp-to-image). rocket_pool_fp16_plan rejects pad>7 (3-bit field). Any failure
        // falls back to the exact host kernel below. NHWC<->CHW transpose makes this a
        // demonstration/coverage path until partitions stay cube-resident.
        if (opts_.pool_npu && fd_ >= 0 && !x.is_quant) {
            const int pt = tot_h / 2, pb = tot_h - tot_h / 2;
            const int pl = tot_w / 2, pr = tot_w - tot_w / 2;
            const bool avg_ok = !x.is_avg || (tot_h == 0 && tot_w == 0);
            rocket_pool_desc d;
            d.c = C; d.ih = IH; d.iw = IW; d.kh = x.kh; d.kw = x.kw;
            d.stride_y = x.sy; d.stride_x = x.sx;
            d.pad_top = pt; d.pad_left = pl; d.pad_bottom = pb; d.pad_right = pr;
            d.method = x.is_avg ? POOL_METHOD_AVG : POOL_METHOD_MAX;
            if (avg_ok && rocket_pool_fp16_plan(&d) == 0) {
                std::vector<_Float16> a((size_t)C * IH * IW), b((size_t)C * OH * OW);
                const float *src = in.data.f;                 // NHWC f32 -> CHW f16
                for (int h = 0; h < IH; h++)
                    for (int w = 0; w < IW; w++)
                        for (int c = 0; c < C; c++)
                            a[((size_t)c * IH + h) * IW + w] = (_Float16)src[((size_t)h * IW + w) * C + c];
                if (rocket_pool_fp16(fd_, &d, a.data(), b.data()) == 0) {
                    float *dst = o.data.f;                    // CHW f16 -> NHWC f32 + fused act
                    for (int oh = 0; oh < OH; oh++)
                        for (int ow = 0; ow < OW; ow++)
                            for (int c = 0; c < C; c++)
                                dst[((size_t)oh * OW + ow) * C + c] =
                                    rocket_apply_act((float)b[((size_t)c * OH + oh) * OW + ow], x.act);
                    return kTfLiteOk;
                }
            }
            if (opts_.profile) fprintf(stderr, "[rocket]   pool_npu failed -> host kernel\n");
        }

        if (!x.is_quant)
            rocket_pool_f(in.data.f, o.data.f, IH, IW, C, OH, OW, x.kh, x.kw, x.sy, x.sx,
                          tot_h / 2, tot_w / 2, x.is_avg, x.act);
        else
            rocket_pool_q(in.data.data, o.data.data, IH, IW, C, OH, OW, x.kh, x.kw, x.sy, x.sx,
                          tot_h / 2, tot_w / 2, x.is_avg, x.act, x.in_uns, x.out_uns,
                          x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    TfLiteStatus eval_concat(TfLiteContext *context, ConcatP &x) {
        const TfLiteTensor &o = context->tensors[x.out];
        const int rank = o.dims->size, axis = x.axis;
        int outer = 1; for (int d = 0; d < axis; d++)        outer *= o.dims->data[d];
        int inner = 1; for (int d = axis + 1; d < rank; d++) inner *= o.dims->data[d];
        const int out_axis = o.dims->data[axis];
        if (!o.data.data) return kTfLiteError;
        int off = 0;
        for (size_t i = 0; i < x.ins.size(); i++) {
            const TfLiteTensor &in = context->tensors[x.ins[i]];
            if (!in.data.data) return kTfLiteError;
            // Re-validate this input's LIVE shape against the output before copying:
            // same rank and identical non-axis extents. The copy applies the
            // output-derived outer/inner strides to EVERY input, so a resize that
            // changed a non-axis extent (only the axis SUM is re-checked, at the end)
            // would smear across rows or read/write out of bounds. The external delegate
            // can't un-delegate at Eval, so fail the Invoke (cf. the conv resize re-checks).
            if (!same_dims_except(in.dims, o.dims, axis)) return kTfLiteError;
            const int in_axis = in.dims->data[axis];
            if (!x.is_quant)
                rocket_concat_in_f(in.data.f, o.data.f, outer, in_axis, out_axis,
                                   inner, off, x.act);
            else
                rocket_concat_in_q(in.data.data, o.data.data, outer, in_axis, out_axis,
                                   inner, off, x.act, x.in_uns[i], x.out_uns,
                                   x.in_scale[i], x.in_zp[i], x.out_scale, x.out_zp);
            off += in_axis;
        }
        return off == out_axis ? kTfLiteOk : kTfLiteError;
    }

    TfLiteStatus eval_reshape(TfLiteContext *context, ReshapeP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n) return kTfLiteError;
        if (n <= 0) return kTfLiteError;                 // reject the dims_count(-1) sentinel
        if (!in.data.data || !o.data.data) return kTfLiteError;
        if (in.data.data != o.data.data)                 // TFLite may alias reshape buffers
            rocket_reshape_copy(o.data.data, in.data.data, (size_t)n, x.elem_size);
        return kTfLiteOk;
    }

    // LAYOUT OPS — byte-exact host kernels. Each re-reads + re-validates the LIVE shapes
    // against the captured params (the external delegate can't un-delegate at Invoke, so a
    // shape that drifted under a dynamic upstream op must fail the Invoke, not miscompute).
    TfLiteStatus eval_transpose(TfLiteContext *context, TransposeP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        if (!in.dims || !o.dims || in.dims->size != x.rank || o.dims->size != x.rank) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        int in_dims[ROCKET_MAX_RANK];
        for (int d = 0; d < x.rank; d++) {
            in_dims[d] = in.dims->data[d];
            if (o.dims->data[d] != in.dims->data[x.perm[d]]) return kTfLiteError;
        }
        rocket_transpose_bytes(o.data.data, in.data.data, x.rank, in_dims, x.perm, x.elem_size);
        return kTfLiteOk;
    }

    TfLiteStatus eval_slice(TfLiteContext *context, SliceP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        // Gather runs in INPUT rank with the stored begin[]/size[]; the output may have a
        // smaller rank (STRIDED_SLICE shrink_axis), so validate by element VOLUME, not rank.
        if (!in.dims || !o.dims || in.dims->size != x.rank) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        int in_dims[ROCKET_MAX_RANK]; long vol = 1;
        for (int d = 0; d < x.rank; d++) {
            in_dims[d] = in.dims->data[d];
            const int step = x.strided ? x.stride[d] : 1;
            // first/last gathered index: with a negative stride begin[] is the HIGH index
            // and the walk descends, so bound BOTH endpoints (not just the larger one).
            const long first = x.begin[d];
            const long last  = (long)x.begin[d] + (long)(x.size[d] - 1) * step;
            const long lo = first < last ? first : last;
            const long hi = first < last ? last : first;
            if (x.size[d] < 0 || step == 0 || lo < 0 || hi >= in_dims[d]) return kTfLiteError;
            vol *= x.size[d];
        }
        if (dims_count(o.dims) != vol) return kTfLiteError;
        if (x.strided)
            rocket_strided_slice_bytes(o.data.data, in.data.data, x.rank, in_dims,
                                       x.begin, x.stride, x.size, x.elem_size);
        else
            rocket_slice_bytes(o.data.data, in.data.data, x.rank, in_dims, x.begin, x.size, x.elem_size);
        return kTfLiteOk;
    }

    TfLiteStatus eval_pad(TfLiteContext *context, PadP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        if (!in.dims || !o.dims || in.dims->size != x.rank || o.dims->size != x.rank) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        int in_dims[ROCKET_MAX_RANK], out_dims[ROCKET_MAX_RANK];
        for (int d = 0; d < x.rank; d++) {
            in_dims[d] = in.dims->data[d]; out_dims[d] = o.dims->data[d];
            if (x.pad_before[d] < 0 || (long)x.pad_before[d] + in_dims[d] > out_dims[d]) return kTfLiteError;
        }
        if (x.is_quant) {
            signed char pe = x.pad_byte;
            rocket_pad_bytes(o.data.data, in.data.data, x.rank, in_dims, x.pad_before, out_dims, x.elem_size, &pe);
        } else {
            float pe = x.pad_value_f;
            rocket_pad_bytes(o.data.data, in.data.data, x.rank, in_dims, x.pad_before, out_dims, x.elem_size, &pe);
        }
        return kTfLiteOk;
    }

    TfLiteStatus eval_split(TfLiteContext *context, SplitP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        if (!in.dims || in.dims->size != x.rank || !in.data.data) return kTfLiteError;
        int in_dims[ROCKET_MAX_RANK];
        for (int d = 0; d < x.rank; d++) in_dims[d] = in.dims->data[d];
        long off = 0;
        for (size_t i = 0; i < x.outs.size(); i++) {
            const TfLiteTensor &o = context->tensors[x.outs[i]];
            if (!o.dims || o.dims->size != x.rank || !o.data.data) return kTfLiteError;
            int out_dims[ROCKET_MAX_RANK], begin[ROCKET_MAX_RANK];
            for (int d = 0; d < x.rank; d++) {
                out_dims[d] = o.dims->data[d]; begin[d] = 0;
                if (d != x.axis && out_dims[d] != in_dims[d]) return kTfLiteError;
            }
            if (off + out_dims[x.axis] > in_dims[x.axis]) return kTfLiteError;
            begin[x.axis] = (int)off;
            rocket_slice_bytes(o.data.data, in.data.data, x.rank, in_dims, begin, out_dims, x.elem_size);
            off += out_dims[x.axis];
        }
        return off == in_dims[x.axis] ? kTfLiteOk : kTfLiteError;
    }

    // Unary activation (HARD_SWISH / LOGISTIC). Default: the exact host kernel
    // (rocket_ops.h). Opt-in (act_npu + a device): run the nonlinearity on the NPU DPU
    // LUT (rocket_activation_fp16) — float in/out converts via fp16; quant dequant->fp16
    // ->LUT->requant. The NPU LUT is an fp16 approximation (gate it with tolerance, not
    // equality), so it stays opt-in; any LUT failure falls back to the host kernel.
    TfLiteStatus eval_act(TfLiteContext *context, ActP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n) return kTfLiteError;
        // n<=0 catches the dims_count(-1) overflow/negative sentinel (both inputs
        // could be -1 and compare equal); n>INT_MAX would truncate in the (int)n
        // cast to rocket_activation_fp16 below, processing fewer elements than a/b hold.
        if (n <= 0 || n > INT_MAX) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;

        if (opts_.act_npu && fd_ >= 0) {
            const int rk = unary_to_rocket_act(x.kind);
            // ELU has its own driver entry point (the x~=0 signed-output host repair);
            // every other claimed kind rides the generic DPU LUT call.
            const int uk = x.kind;
            auto run_npu = [&](const _Float16 *src, _Float16 *dst) -> int {
                if (uk == ROCKET_UNARY_ELU)
                    return rocket_elu_fp16(fd_, 1.0f, src, dst, (int)n);
                if (uk == ROCKET_UNARY_LEAKY_RELU)
                    return rocket_leaky_relu_fp16(fd_, x.param, src, dst, (int)n);
                return rocket_activation_fp16(fd_, rk, src, dst, (int)n);
            };
            if (rk >= 0) {
                std::vector<_Float16> a((size_t)n), b((size_t)n);
                if (!x.is_quant) {
                    const float *src = in.data.f;
                    for (long i = 0; i < n; i++) a[i] = (_Float16)src[i];
                    if (run_npu(a.data(), b.data()) == 0) {
                        float *dst = o.data.f;
                        for (long i = 0; i < n; i++) dst[i] = (float)b[i];
                        return kTfLiteOk;
                    }
                } else {
                    for (long i = 0; i < n; i++)
                        a[i] = (_Float16)rocket_dq1(rocket_qread(in.data.data, i, x.in_uns),
                                                    x.in_scale, x.in_zp);
                    if (run_npu(a.data(), b.data()) == 0) {
                        int qmin, qmax; rocket_qrange(x.out_uns, &qmin, &qmax);
                        const float inv = 1.0f / x.out_scale;
                        for (long i = 0; i < n; i++)
                            rocket_qwrite(o.data.data, i,
                                rocket_rq1((float)b[i], inv, x.out_zp, qmin, qmax), x.out_uns);
                        return kTfLiteOk;
                    }
                }
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   act_npu LUT failed -> host kernel\n");
            }
        }

        if (!x.is_quant)
            rocket_unary_f(in.data.f, o.data.f, (size_t)n, x.kind, x.param);
        else
            rocket_unary_q(in.data.data, o.data.data, (size_t)n, x.kind,
                           x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp,
                           x.param);
        return kTfLiteOk;
    }

    // LOG_SOFTMAX: exact host kernel (stable: subtract row max). The NPU route
    // (rocket_logsoftmax_fp16, [M][N]) is a documented follow-on.
    TfLiteStatus eval_logsoftmax(TfLiteContext *context, SeqP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n || (long)x.M * x.N != n) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        if (!x.is_quant)
            rocket_logsoftmax_f(in.data.f, o.data.f, x.M, x.N);
        else
            rocket_logsoftmax_q(in.data.data, o.data.data, x.M, x.N,
                                x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // SOFTMAX: exact host kernel (stable: subtract row max) over the last axis. The NPU
    // route (rocket_softmax_fp16, [M][N]) is a documented follow-on.
    TfLiteStatus eval_softmax(TfLiteContext *context, SeqP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n || (long)x.M * x.N != n) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        if (!x.is_quant)
            rocket_softmax_f(in.data.f, o.data.f, x.M, x.N, x.beta);
        else
            rocket_softmax_q(in.data.data, o.data.data, x.M, x.N, x.beta,
                             x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // CUMSUM: exact host prefix-sum along the last axis. The NPU route
    // (rocket_cumsum_fp16, cumsum-as-triangular-matmul) is a documented follow-on.
    TfLiteStatus eval_cumsum(TfLiteContext *context, SeqP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n || (long)x.M * x.N != n) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        if (!x.is_quant)
            rocket_cumsum_f(in.data.f, o.data.f, x.M, x.N, x.exclusive, x.reverse);
        else
            rocket_cumsum_q(in.data.data, o.data.data, x.M, x.N, x.exclusive, x.reverse,
                            x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // L2_NORMALIZATION. Default: the exact host kernel (fp32 accumulate). Opt-in
    // (norm_npu + a device, FLOAT): run on the NPU (rocket_l2norm_fp16, [M][C]).
    TfLiteStatus eval_l2norm(TfLiteContext *context, L2NormP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n || n <= 0 || n > INT_MAX) return kTfLiteError;
        if ((long)x.M * x.C != n) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;

        if (opts_.norm_npu && fd_ >= 0 && !x.is_quant) {
            std::vector<_Float16> a((size_t)n), b((size_t)n);
            for (long i = 0; i < n; i++) a[i] = (_Float16)in.data.f[i];
            if (rocket_l2norm_fp16(fd_, x.M, x.C, a.data(), 1e-12f, b.data()) == 0) {
                for (long i = 0; i < n; i++) o.data.f[i] = (float)b[i];
                return kTfLiteOk;
            }
            if (opts_.profile) fprintf(stderr, "[rocket]   norm_npu failed -> host kernel\n");
        }

        if (!x.is_quant)
            rocket_l2norm_f(in.data.f, o.data.f, x.M, x.C);
        else
            rocket_l2norm_q(in.data.data, o.data.data, x.M, x.C,
                            x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // TRANSPOSE_CONV (float v1): exact host scatter kernel (the convention matches
    // TFLite, so bit-exact). The NPU route (rocket_conv_transpose2d_fp16, same scatter)
    // is a follow-on: it needs the weight repack [OC,KH,KW,IC] -> [IC,OC,KH,KW] + the
    // pad_h -> (pad_top, opad) split + the NHWC<->NCHW transpose.
    TfLiteStatus eval_tconv(TfLiteContext *context, TConvP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &w  = context->tensors[x.w_idx];
        const TfLiteTensor &o  = context->tensors[x.out];
        if (!in.data.f || !w.data.f || !o.data.f) return kTfLiteError;
        if (!in.dims || in.dims->size != 4 ||
            in.dims->data[1] != x.IH || in.dims->data[2] != x.IW || in.dims->data[3] != x.IC)
            return kTfLiteError;
        if (!o.dims || o.dims->data[1] != x.OH || o.dims->data[2] != x.OW || o.dims->data[3] != x.OC)
            return kTfLiteError;
        const float *bias = (x.bias_idx >= 0) ? context->tensors[x.bias_idx].data.f : nullptr;
        rocket_transpose_conv_f(in.data.f, w.data.f, bias, o.data.f,
                                x.IH, x.IW, x.IC, x.OC, x.KH, x.KW, x.sy, x.sx,
                                x.pad_h, x.pad_w, x.OH, x.OW, x.act);
        return kTfLiteOk;
    }

    // RESIZE_NEAREST_NEIGHBOR / RESIZE_BILINEAR. Default: the exact host kernel
    // (TFLite coordinate math). Opt-in (resize_npu + a device): an INTEGER-factor,
    // align_corners=false, half_pixel=false NEAREST resize routes onto the NPU
    // (rocket_upsample_nearest_fp16, block replication = floor mode) via an NHWC<->
    // [C][H][W] transpose; everything else stays on the host kernel.
    TfLiteStatus eval_resize(TfLiteContext *context, ResizeP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        if (!in.dims || in.dims->size != 4) return kTfLiteError;
        if (in.dims->data[1] != x.IH || in.dims->data[2] != x.IW || in.dims->data[3] != x.C)
            return kTfLiteError;
        if (!o.dims || o.dims->data[1] != x.OH || o.dims->data[2] != x.OW || o.dims->data[3] != x.C)
            return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        const int IH = x.IH, IW = x.IW, C = x.C, OH = x.OH, OW = x.OW;

        if (opts_.resize_npu && fd_ >= 0 && !x.is_quant && !x.bilinear &&
            !x.align_corners && !x.half_pixel &&
            OH % IH == 0 && OW % IW == 0) {
            const int sy = OH / IH, sx = OW / IW;
            if (rocket_upsample_nearest_plan(C, IH, IW, sy, sx) == 0) {
                std::vector<_Float16> a((size_t)C * IH * IW), b((size_t)C * OH * OW);
                const float *src = in.data.f;
                for (int p = 0; p < IH * IW; p++)
                    for (int c = 0; c < C; c++)
                        a[(size_t)c * IH * IW + p] = (_Float16)src[(size_t)p * C + c];
                if (rocket_upsample_nearest_fp16(fd_, a.data(), b.data(), C, IH, IW, sy, sx) == 0) {
                    float *dst = o.data.f;
                    for (int p = 0; p < OH * OW; p++)
                        for (int c = 0; c < C; c++)
                            dst[(size_t)p * C + c] = (float)b[(size_t)c * OH * OW + p];
                    return kTfLiteOk;
                }
            }
            if (opts_.profile) fprintf(stderr, "[rocket]   resize_npu failed -> host kernel\n");
        }

        if (!x.is_quant) {
            if (x.bilinear)
                rocket_resize_bilinear_f(in.data.f, o.data.f, IH, IW, C, OH, OW,
                                         x.align_corners, x.half_pixel);
            else
                rocket_resize_nearest_f(in.data.f, o.data.f, IH, IW, C, OH, OW,
                                        x.align_corners, x.half_pixel);
        } else {
            if (x.bilinear)
                rocket_resize_bilinear_q(in.data.data, o.data.data, IH, IW, C, OH, OW,
                                         x.align_corners, x.half_pixel, x.uns, x.scale, x.zp);
            else
                rocket_resize_nearest_q(in.data.data, o.data.data, IH, IW, C, OH, OW,
                                        x.align_corners, x.half_pixel, x.elem_size);
        }
        return kTfLiteOk;
    }

    // SPATIAL REDUCE (MEAN / MAX / MIN over [1,2]). Default: the exact host kernel
    // (fp32 accumulate). Opt-in (pool_npu + a device, FLOAT): run on the NPU PPU
    // (rocket_global_{avg,max,min}pool_fp16) when the shape is PPU-decomposable
    // (rocket_global_avgpool_plan); needs an NHWC<->[C][H][W] transpose. quant -> host.
    TfLiteStatus eval_reduce(TfLiteContext *context, ReduceP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &o  = context->tensors[x.out];
        // Re-validate live shapes (a resize could change H/W/C under us).
        if (!in.dims || in.dims->size != 4) return kTfLiteError;
        if (in.dims->data[1] != x.H || in.dims->data[2] != x.W || in.dims->data[3] != x.C)
            return kTfLiteError;
        if (dims_count(o.dims) != (long)x.C) return kTfLiteError;
        if (!in.data.data || !o.data.data) return kTfLiteError;
        const int H = x.H, W = x.W, C = x.C;

        if (opts_.pool_npu && fd_ >= 0 && !x.is_quant && rocket_global_avgpool_plan(C, H, W) == 0) {
            std::vector<_Float16> a((size_t)C * H * W), b((size_t)C);
            const float *src = in.data.f;                       // NHWC (p*C+c) -> [C][H*W]
            for (int p = 0; p < H * W; p++)
                for (int c = 0; c < C; c++)
                    a[(size_t)c * H * W + p] = (_Float16)src[(size_t)p * C + c];
            int rc = x.op == ROCKET_REDUCE_MEAN ? rocket_global_avgpool_fp16(fd_, C, H, W, a.data(), b.data())
                   : x.op == ROCKET_REDUCE_MAX  ? rocket_global_maxpool_fp16(fd_, C, H, W, a.data(), b.data())
                   :                              rocket_global_minpool_fp16(fd_, C, H, W, a.data(), b.data());
            if (rc == 0) {
                for (int c = 0; c < C; c++) o.data.f[c] = (float)b[c];
                return kTfLiteOk;
            }
            if (opts_.profile) fprintf(stderr, "[rocket]   reduce_npu failed -> host kernel\n");
        }

        if (!x.is_quant)
            rocket_reduce_spatial_f(in.data.f, o.data.f, H, W, C, x.op);
        else
            rocket_reduce_spatial_q(in.data.data, o.data.data, H, W, C, x.op,
                                    x.in_uns, x.out_uns, x.in_scale, x.in_zp,
                                    x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // PRELU: per-channel parametric ReLU. Default: the exact host kernel. The alpha
    // model constant is dequantized to float[C] here (NHWC -> channel is the innermost
    // index). Opt-in (act_npu + a device, FLOAT): run on the NPU via rocket_prelu_fp16
    // (ew_max/mul, no LUT) — needs an NHWC<->[C][S] transpose; quant stays host.
    TfLiteStatus eval_prelu(TfLiteContext *context, PreluP &x) {
        const TfLiteTensor &in = context->tensors[x.in];
        const TfLiteTensor &al = context->tensors[x.alpha_idx];
        const TfLiteTensor &o  = context->tensors[x.out];
        const long n = dims_count(o.dims);
        if (dims_count(in.dims) != n) return kTfLiteError;
        if (n <= 0 || n > INT_MAX) return kTfLiteError;
        if (!in.data.data || !al.data.data || !o.data.data) return kTfLiteError;
        const int C = x.C;
        if (C <= 0 || n % C != 0) return kTfLiteError;

        // Dequantize the per-channel slope to float[C] (float: a straight copy).
        std::vector<float> alpha((size_t)C);
        if (!x.is_quant)
            for (int c = 0; c < C; c++) alpha[c] = al.data.f[c];
        else
            for (int c = 0; c < C; c++)
                alpha[c] = rocket_dq1(rocket_qread(al.data.data, c, x.alpha_uns),
                                      x.alpha_scale, x.alpha_zp);

        if (opts_.act_npu && fd_ >= 0 && !x.is_quant) {
            const int S = (int)(n / C);
            std::vector<_Float16> xt((size_t)n), yt((size_t)n);
            const float *src = in.data.f;                       // NHWC (s*C+c) -> [C][S] (c*S+s)
            for (int s = 0; s < S; s++)
                for (int c = 0; c < C; c++)
                    xt[(size_t)c * S + s] = (_Float16)src[(size_t)s * C + c];
            if (rocket_prelu_fp16(fd_, C, S, xt.data(), alpha.data(), yt.data()) == 0) {
                float *dst = o.data.f;
                for (int s = 0; s < S; s++)
                    for (int c = 0; c < C; c++)
                        dst[(size_t)s * C + c] = (float)yt[(size_t)c * S + s];
                return kTfLiteOk;
            }
            if (opts_.profile) fprintf(stderr, "[rocket]   prelu_npu failed -> host kernel\n");
        }

        if (!x.is_quant)
            rocket_prelu_f(in.data.f, o.data.f, (size_t)n, C, alpha.data());
        else
            rocket_prelu_q(in.data.data, o.data.data, (size_t)n, C, alpha.data(),
                           x.in_uns, x.out_uns, x.in_scale, x.in_zp, x.out_scale, x.out_zp);
        return kTfLiteOk;
    }

    // FULLY_CONNECTED (float): C[M,N] = A[M,K]*B[N,K]^T + bias, fused act. NPU path (mm_w):
    // pad A's K to %32, run the resident fp16 matmul into a [M,Npad] buffer, then unpad +
    // bias + act into the [M,N] output. Host fallback (no device / pack failed): rocket_fc_f
    // straight off the const weight. The matmul integer/fp accumulate is HW-validated; the
    // pad + bias/act epilogue here is the only new glue (exercised off-HW by convert_test).
    TfLiteStatus eval_fc(TfLiteContext *context, FCNode &f) {
        const TfLiteTensor &in = context->tensors[f.in_idx];
        const TfLiteTensor &w  = context->tensors[f.w_idx];
        const TfLiteTensor &o  = context->tensors[f.out_idx];
        if (!in.data.f || !w.data.f || !o.data.f) return kTfLiteError;
        const float *bias = (f.bias_idx >= 0) ? context->tensors[f.bias_idx].data.f : nullptr;
        const int M = f.M, K = f.K, N = f.N;

        if (f.mm_w && f.mm_M == M) {              // resident NPU matmul (padded K/N)
            const int Kpad = round_up(K, 32), Npad = round_up(N, 16);
            std::vector<_Float16> A((size_t)M * Kpad, (_Float16)0), C((size_t)M * Npad);
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++)
                    A[(size_t)m * Kpad + k] = (_Float16)in.data.f[(size_t)m * K + k];
            if (rocket_matmul_fp16_prepacked(mm_ctx_, M, Kpad, Npad, A.data(), C.data(),
                                             f.mm_w) != 0) {
                if (opts_.profile)
                    fprintf(stderr, "[rocket]   fc matmul FAILED (M=%d Kpad=%d Npad=%d) -> host\n",
                            M, Kpad, Npad);
                rocket_fc_f(in.data.f, w.data.f, bias, o.data.f, M, K, N, f.act);  // safety net
                return kTfLiteOk;
            }
            for (int m = 0; m < M; m++)
                for (int nn = 0; nn < N; nn++) {
                    float v = (float)C[(size_t)m * Npad + nn] + (bias ? bias[nn] : 0.f);
                    o.data.f[(size_t)m * N + nn] = rocket_apply_act(v, f.act);
                }
            return kTfLiteOk;
        }

        rocket_fc_f(in.data.f, w.data.f, bias, o.data.f, M, K, N, f.act);  // host fallback
        return kTfLiteOk;
    }

    RocketOptions opts_;
    std::vector<Node> nodes_;
    std::vector<int> part_out_;                       // partition output tensor indices
    std::vector<int> internal_idx_;                   // ALL partition-internal tensors (from Init)
    std::vector<int> nhwc_idx_;                        // internal tensors kept in NHWC (Prepare)
    std::vector<std::vector<uint8_t>> internal_buf_;  // NHWC storage, parallel to nhwc_idx_
    std::vector<int> resident_idx_;                    // conv->conv internals kept fp16-NCHW
    std::vector<std::vector<_Float16>> resident_buf_;  // fp16-NCHW storage, parallel to resident_idx_
    int  fd_ = -1;
    bool opened_ = false;
    rocket_ctx *mm_ctx_ = nullptr;        // persistent fp16 matmul worker fds + scratch (1x1 path)
    rocket_i8_ctx *mm_i8_ctx_ = nullptr;  // persistent int8 matmul worker fds (native 1x1 int8/uint8)
    rocket_conv_ctx *conv_ctx_ = nullptr; // resident conv BOs on fd_ (general/dw conv path)
    rocket_conv_pool *conv_pool_ = nullptr; // multicore worker pool (native int8/uint8 direct)
};

// ===========================================================================
// TfLiteDelegate registration and partition prepare
// ===========================================================================

// ---------------------------------------------------------------------------
// delegate (classic C TfLiteDelegate). The C++ SimpleDelegate helper would do this
// for us, but its header/lib aren't shipped without a full TFLite build — so we drive
// the C API directly (the same path Mesa's Teflon delegate takes): Prepare walks the
// execution plan, asks analyze_node which nodes we claim, and hands the contiguous
// supported set to ReplaceNodeSubsetsWithDelegateKernels with a kernel registration
// whose init/free/prepare/invoke route to RocketKernel.
// ---------------------------------------------------------------------------

// The delegate object: a TfLiteDelegate whose data_ points back to this struct so the
// callbacks can recover the parsed options. Heap-owned; freed in destroy.
struct RocketDelegate {
    TfLiteDelegate base;     // handed to TFLite (base.data_ == this)
    RocketOptions  opts;
};

// kernel registration callbacks ------------------------------------------------------

// init: `buffer` is a TfLiteDelegateParams* (the partition's nodes + IO). Build the
// per-partition RocketKernel; the returned pointer is stored in node->user_data.
static void *rocket_kernel_init(TfLiteContext *context, const char *buffer, size_t /*len*/) {
    const auto *params = reinterpret_cast<const TfLiteDelegateParams *>(buffer);
    if (!params || !params->delegate) return nullptr;
    const auto *self = reinterpret_cast<const RocketDelegate *>(params->delegate->data_);
    // Own the kernel until Init succeeds, so an exception from Init (its std::vector
    // growth can throw std::bad_alloc) cannot leak it. Released to the caller (stored
    // in node->user_data, freed by rocket_kernel_free) only on the success path.
    std::unique_ptr<RocketKernel> kernel(new (std::nothrow) RocketKernel(self->opts));
    if (!kernel) return nullptr;
    if (kernel->Init(context, params) != kTfLiteOk) return nullptr;
    return kernel.release();
}

static void rocket_kernel_free(TfLiteContext * /*context*/, void *buffer) {
    delete reinterpret_cast<RocketKernel *>(buffer);
}

static TfLiteStatus rocket_kernel_prepare(TfLiteContext *context, TfLiteNode *node) {
    auto *kernel = reinterpret_cast<RocketKernel *>(node->user_data);
    return kernel ? kernel->Prepare(context) : kTfLiteError;
}

static TfLiteStatus rocket_kernel_invoke(TfLiteContext *context, TfLiteNode *node) {
    auto *kernel = reinterpret_cast<RocketKernel *>(node->user_data);
    return kernel ? kernel->Eval(context) : kTfLiteError;
}

static TfLiteRegistration rocket_kernel_registration() {
    TfLiteRegistration r{};                       // zero-inits version/external/async etc.
    r.init = rocket_kernel_init;
    r.free = rocket_kernel_free;
    r.prepare = rocket_kernel_prepare;
    r.invoke = rocket_kernel_invoke;
    r.builtin_code = kTfLiteBuiltinDelegate;
    r.custom_name = "RocketDelegate";
    return r;
}

// delegate Prepare: claim every supported node and replace the subset with our kernel.
static TfLiteStatus rocket_delegate_prepare(TfLiteContext *context, TfLiteDelegate *delegate) {
    const auto *self = reinterpret_cast<const RocketDelegate *>(delegate->data_);
    TfLiteIntArray *plan = nullptr;
    if (context->GetExecutionPlan(context, &plan) != kTfLiteOk) return kTfLiteError;

    std::vector<int> supported;
    supported.reserve(plan->size);
    for (int i = 0; i < plan->size; i++) {
        const int node_index = plan->data[i];
        TfLiteNode *node = nullptr;
        TfLiteRegistration *reg = nullptr;
        if (context->GetNodeAndRegistration(context, node_index, &node, &reg) != kTfLiteOk)
            return kTfLiteError;
        if (analyze_node(reg, node, context, self->opts, nullptr))
            supported.push_back(node_index);
    }

    TfLiteIntArray *nodes = TfLiteIntArrayCreate((int)supported.size());
    if (!nodes) return kTfLiteError;
    for (size_t i = 0; i < supported.size(); i++) nodes->data[i] = supported[i];
    const TfLiteRegistration kreg = rocket_kernel_registration();
    const TfLiteStatus st =
        context->ReplaceNodeSubsetsWithDelegateKernels(context, kreg, nodes, delegate);
    TfLiteIntArrayFree(nodes);
    return st;
}

}  // namespace

// Parse a delegate option value as a long. atoi/atol fail silently to 0 on garbage
// ("abc" -> 0, "" -> 0, "12x" -> 12), so validate with strtol and warn, falling back
// to `def`, on an empty / non-numeric / trailing-garbage value.
static long rocket_opt_long(const char *k, const char *v, long def) {
    if (!v || !*v) { fprintf(stderr, "[rocket] option '%s': empty value, using %ld\n", k, def); return def; }
    char *end = nullptr;
    const long r = strtol(v, &end, 10);
    if (end == v || *end != '\0') {
        fprintf(stderr, "[rocket] option '%s': cannot parse '%s' as integer, using %ld\n", k, v, def);
        return def;
    }
    return r;
}

// ===========================================================================
// External-delegate entry points (Create / Destroy)
// ===========================================================================

// ---------------------------------------------------------------------------
// external delegate entry points (the two symbols tflite_runtime dlsym's)
// ---------------------------------------------------------------------------
extern "C" {

TfLiteDelegate *tflite_plugin_create_delegate(
        char **options_keys, char **options_values, size_t num_options,
        void (* /*report_error*/)(const char *)) {
    RocketOptions opts;
    for (size_t i = 0; i < num_options; i++) {
        if (!options_keys || !options_values || !options_keys[i] || !options_values[i])
            continue;
        const std::string k = options_keys[i];
        const char *v = options_values[i];
        if      (k == "nthreads") opts.nthreads = (int)rocket_opt_long("nthreads", v, opts.nthreads);
        else if (k == "min_macs") opts.min_macs =      rocket_opt_long("min_macs", v, opts.min_macs);
        else if (k == "aux_ops")  opts.aux_ops  = (rocket_opt_long("aux_ops", v, 1) != 0);
        else if (k == "profile")  opts.profile  = (rocket_opt_long("profile", v, 0) != 0);
        else if (k == "nchw_resident") opts.nchw_resident = (rocket_opt_long("nchw_resident", v, 0) != 0);
        else if (k == "native_int8")   opts.native_int8   = (rocket_opt_long("native_int8", v, 0) != 0);
        else if (k == "mm_int8")       opts.mm_int8       = (rocket_opt_long("mm_int8", v, 1) != 0);
        else if (k == "act_npu")       opts.act_npu       = (rocket_opt_long("act_npu", v, 0) != 0);
        else if (k == "fc_npu")        opts.fc_npu        = (rocket_opt_long("fc_npu", v, 0) != 0);
        else if (k == "pool_npu")      opts.pool_npu      = (rocket_opt_long("pool_npu", v, 0) != 0);
        else if (k == "ew_npu")        opts.ew_npu        = (rocket_opt_long("ew_npu", v, 0) != 0);
        else if (k == "resize_npu")    opts.resize_npu    = (rocket_opt_long("resize_npu", v, 0) != 0);
        else if (k == "norm_npu")      opts.norm_npu      = (rocket_opt_long("norm_npu", v, 0) != 0);
    }
    if (opts.nthreads < 1) opts.nthreads = 1;
    if (opts.nthreads > 8) opts.nthreads = 8;

    auto *d = new RocketDelegate{};
    d->opts = opts;
    d->base = TfLiteDelegate{};                    // zero all callbacks/flags
    d->base.data_ = d;                             // recovered in the callbacks
    d->base.Prepare = &rocket_delegate_prepare;
    d->base.flags = kTfLiteDelegateFlagsNone;
    return &d->base;
}

void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate) {
    if (!delegate) return;
    delete reinterpret_cast<RocketDelegate *>(delegate->data_);
}

}  // extern "C"
