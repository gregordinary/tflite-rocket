// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
#ifndef ROCKET_CONVERT_H
#define ROCKET_CONVERT_H

#include <stddef.h>
#include <stdlib.h>            /* getenv (ROCKET_REQUANT_SCALAR A/B knob) */
#include <string.h>
#include <math.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>            /* NEON box-sum (rocket_in_window_sum_i8_band) */
#endif

/*
 * rocket_convert.h — pure layout/padding/activation glue between TFLite's
 * NHWC float CONV_2D tensors and the rocket-userspace driver's NCHW fp16 conv.
 *
 * These helpers carry NO TFLite (or hardware) dependency on purpose: the layout
 * transposes and the SAME/VALID padding arithmetic are the only genuinely new
 * logic the delegate adds on top of the already-HW-validated rocket_conv2d_fp16,
 * so they are factored out here to be unit-tested on x86 against an independent
 * NHWC oracle (tests/convert_test.cpp) — the same off-hardware discipline the
 * driver's cube-self-check uses. The delegate is the only other includer.
 *
 * Layout conventions:
 *   TFLite (NHWC, batch 1):
 *     input   [1][IH][IW][IC]       elem(h,w,c)      = (h*IW + w)*IC + c
 *     filter  [OC][KH][KW][IC]      elem(oc,kh,kw,ic)= ((oc*KH+kh)*KW+kw)*IC + ic
 *     dwfilt  [1][KH][KW][C]        elem(kh,kw,c)    = (kh*KW + kw)*C + c   (depthwise)
 *     output  [1][OH][OW][OC]       elem(oh,ow,oc)   = (oh*OW + ow)*OC + oc
 *   Driver (NCHW, batch 1):
 *     in      [IC][IH][IW]          elem(ic,ih,iw)   = (ic*IH+ih)*IW + iw
 *     W       [OC][IC][KH][KW]      elem(oc,ic,kh,kw)= ((oc*IC+ic)*KH+kh)*KW + kw
 *     Wdw     [C][KH][KW]           elem(c,kh,kw)    = (c*KH+kh)*KW + kw     (depthwise)
 *     out     [OC][OH][OW]          elem(oc,oh,ow)   = (oc*OH+oh)*OW + ow
 *
 * Depthwise (DEPTHWISE_CONV_2D, depth_multiplier 1 => OC==IC==C): TFLite's filter is
 * channel-last with a leading 1, [1][KH][KW][C]; the driver wants one KH×KW filter per
 * channel, [C][KH][KW] (it does its own (C/G,KH,KW,G) cube scatter from there). The
 * input/output transposes and the SAME/VALID pad arithmetic are IDENTICAL to the direct
 * conv — only the filter reorder (and the quant axis) differ.
 */

/* Fused-activation codes (a TFLite-free mirror of the subset we implement; the
 * delegate maps TfLiteFusedActivation onto these). */
enum {
    ROCKET_ACT_NONE   = 0,
    ROCKET_ACT_RELU   = 1,
    ROCKET_ACT_RELU6  = 2,
    ROCKET_ACT_RELUN1 = 3,   /* ReluN1To1: clamp to [-1, 1] */
};

static inline float rocket_apply_act(float v, int act)
{
    switch (act) {
    case ROCKET_ACT_RELU:   return v > 0.f ? v : 0.f;
    case ROCKET_ACT_RELU6:  return v < 0.f ? 0.f : (v > 6.f ? 6.f : v);
    case ROCKET_ACT_RELUN1: return v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
    default:                return v;
    }
}

#if defined(__ARM_NEON)
/* NEON sibling of rocket_apply_act: the four fused acts are min/max clamps, which
 * vmax/vmin reproduce bit-for-bit (same result as the scalar ?: on every finite v). */
static inline float32x4_t rocket_apply_act_f32x4(float32x4_t v, int act)
{
    switch (act) {
    case ROCKET_ACT_RELU:   return vmaxq_f32(v, vdupq_n_f32(0.f));
    case ROCKET_ACT_RELU6:  return vminq_f32(vmaxq_f32(v, vdupq_n_f32(0.f)), vdupq_n_f32(6.f));
    case ROCKET_ACT_RELUN1: return vminq_f32(vmaxq_f32(v, vdupq_n_f32(-1.f)), vdupq_n_f32(1.f));
    default:                return v;
    }
}
#endif

/* Effective kernel extent on one axis once dilation is applied. */
static inline int rocket_eff_k(int k, int dil) { return (k - 1) * dil + 1; }

/* Total zero-padding TFLite needs on one axis to map `in` -> `out` for the given
 * stride/dilation/kernel. Derived from the output size (TFLite's ground truth):
 * pad = (out-1)*stride + eff_k - in, clamped at 0. For VALID this evaluates to 0;
 * for SAME it is the symmetric-or-off-by-one total (pad_before = pad/2, the rest
 * after). Materializing the full pad and running the driver with pad_top=pad_left=0
 * reproduces TFLite SAME/VALID exactly (including asymmetric SAME, e.g. even input
 * with stride 2), so the delegate never relies on the driver's symmetric pad. */
static inline int rocket_total_pad(int in, int k, int stride, int dil, int out)
{
    int p = (out - 1) * stride + rocket_eff_k(k, dil) - in;
    return p > 0 ? p : 0;
}

/* TFLite output extent on one axis. same=1 -> ceil(in/stride); same=0 (VALID) ->
 * floor((in - eff_k)/stride) + 1. (The delegate instead reads the dims TFLite
 * already computed into the output tensor; this mirrors them for the unit test.) */
static inline int rocket_out_dim(int in, int k, int stride, int dil, int same)
{
    if (same) return (in + stride - 1) / stride;
    int e = rocket_eff_k(k, dil);
    return in >= e ? (in - e) / stride + 1 : 0;
}

/*
 * input NHWC float -> NCHW fp16 with the convolution's padding materialized.
 * `dst` is [IC][IHp][IWp] with IHp=IH+tot_h, IWp=IW+tot_w; the real input lands at
 * offset (pad_top, pad_left) and the surrounding halo is zero. The driver then
 * runs a pad_top=pad_left=0 conv over it (see rocket_total_pad). dst is fully
 * written (the pad region is zeroed here), so the caller need not preclear it.
 */
static inline void rocket_in_nhwc_to_nchw_pad(
        const float *src, _Float16 *dst,
        int IC, int IH, int IW, int pad_top, int pad_left, int IHp, int IWp)
{
    memset(dst, 0, (size_t)IC * IHp * IWp * sizeof(_Float16));
    for (int ih = 0; ih < IH; ih++)
        for (int iw = 0; iw < IW; iw++) {
            const float *s = src + ((size_t)ih * IW + iw) * IC;
            size_t base = ((size_t)(ih + pad_top)) * IWp + (iw + pad_left);
            for (int ic = 0; ic < IC; ic++)
                dst[(size_t)ic * IHp * IWp + base] = (_Float16)s[ic];
        }
}

/*
 * Conv filter TFLite [OC][KH][KW][IC] -> driver [OC][IC][KH][KW], fp16. Packed
 * ONCE in the delegate's Prepare (weights are model constants), not per inference.
 */
static inline void rocket_filter_ohwi_to_oihw(
        const float *src, _Float16 *dst, int OC, int KH, int KW, int IC)
{
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    dst[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                        (_Float16)src[(((size_t)oc * KH + kh) * KW + kw) * IC + ic];
}

/*
 * Depthwise filter TFLite [1][KH][KW][C] -> driver [C][KH][KW], fp16. The driver's
 * depthwise weight is one KH×KW filter per channel (OC==IC==C); it does its own
 * (C/G,KH,KW,G) cube scatter from this row-major [C][KH][KW]. Packed ONCE in Prepare.
 */
static inline void rocket_dw_filter_hwc_to_chw(
        const float *src, _Float16 *dst, int C, int KH, int KW)
{
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                dst[((size_t)c * KH + kh) * KW + kw] =
                    (_Float16)src[((size_t)kh * KW + kw) * C + c];
}

/*
 * output driver NCHW fp16 -> TFLite NHWC float, adding bias (per output channel,
 * may be NULL) and applying the fused activation. This is the inverse transpose
 * of the input pack plus the conv's epilogue.
 */
static inline void rocket_out_nchw_to_nhwc_bias_act(
        const _Float16 *src, float *dst,
        int OC, int OH, int OW, const float *bias, int act)
{
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            float *d = dst + ((size_t)oh * OW + ow) * OC;
            for (int oc = 0; oc < OC; oc++) {
                float v = (float)src[((size_t)oc * OH + oh) * OW + ow];
                if (bias) v += bias[oc];
                d[oc] = rocket_apply_act(v, act);
            }
        }
}

/* ==========================================================================
 * NCHW-resident inter-op helpers. When a partition-internal tensor is
 * produced by an NPU conv and consumed only by other NPU convs, the delegate keeps
 * it in fp16-NCHW (the driver's layout, post bias+activation) BETWEEN the ops instead
 * of transposing+requantizing it back to NHWC (int8/float) and dequantizing+transposing
 * it again at the consumer. These two helpers are that path: the producer applies
 * bias+act in NCHW in place (rocket_nchw_bias_act), and the consumer materializes its
 * padding from the resident NCHW buffer (rocket_nchw_pad) — no transpose, no requant,
 * no dequant. Pure/TFLite-free, unit-tested in convert_test.
 * ========================================================================== */

/* fp16 NCHW [C][IH][IW] (unpadded) -> fp16 NCHW [C][IHp][IWp] with the conv's padding
 * materialized: the real data lands at (pad_top,pad_left) and the halo is zero. The
 * NCHW analogue of rocket_in_nhwc_to_nchw_pad with NO transpose/dequant (the source is
 * already the producer conv's fp16-NCHW output) — just a row-wise memcpy into the
 * zero-haloed destination. dst is fully written. */
static inline void rocket_nchw_pad(
        const _Float16 *src, _Float16 *dst,
        int C, int IH, int IW, int pad_top, int pad_left, int IHp, int IWp)
{
    memset(dst, 0, (size_t)C * IHp * IWp * sizeof(_Float16));
    for (int c = 0; c < C; c++)
        for (int ih = 0; ih < IH; ih++)
            memcpy(&dst[((size_t)c * IHp + (ih + pad_top)) * IWp + pad_left],
                   &src[((size_t)c * IH + ih) * IW],
                   (size_t)IW * sizeof(_Float16));
}

/* Apply per-output-channel bias (may be NULL) + fused activation to a conv's fp16-NCHW
 * output [OC][OH][OW] IN PLACE. This is the epilogue rocket_out_nchw_to_nhwc_{q,bias_act}
 * fold into their transpose; for a resident-NCHW output we apply it here and keep fp16
 * (the value is then the real post-activation activation the next conv consumes directly). */
static inline void rocket_nchw_bias_act(
        _Float16 *buf, int OC, int OH, int OW, const float *bias, int act)
{
    for (int oc = 0; oc < OC; oc++) {
        const float b = bias ? bias[oc] : 0.f;
        for (size_t i = 0; i < (size_t)OH * OW; i++) {
            size_t k = (size_t)oc * OH * OW + i;
            buf[k] = (_Float16)rocket_apply_act((float)buf[k] + b, act);
        }
    }
}

/* ==========================================================================
 * FULLY_CONNECTED (float) — a matmul C[M,N] = A[M,K] · B[N,K]^T + bias, fused act.
 * M = flattened batch (input.size / K), K = input_dim (= weights.dims[1]), N = units
 * (= weights.dims[0]). The NPU path packs B once and runs the resident fp16 matmul
 * (a 1x1 conv IS this matmul); this host kernel is the off-device / fd<0 fallback AND
 * the convert_test oracle (fp32 accumulate -> fp32, no NPU). bias may be NULL. */
static inline void rocket_fc_f(const float *A, const float *B, const float *bias,
                               float *C, int M, int K, int N, int act)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = bias ? bias[n] : 0.f;
            const float *a = A + (size_t)m * K;
            const float *b = B + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            C[(size_t)m * N + n] = rocket_apply_act(acc, act);
        }
}

/* ==========================================================================
 * Quantized (int8 / uint8) CONV_2D boundary.
 *
 * Real Frigate detectors are int8/uint8 TFLite. By default the delegate DEQUANTIZES
 * at the partition boundary and reuses the HW-validated fp16 conv (the opt-in
 * native_int8 path instead runs int8/uint8 convs directly on the NPU): weights are
 * dequantized ONCE in
 * Prepare, activations per inference, the fp16 conv runs on the NPU, and the
 * output is requantized back to int8/uint8. This is an arithmetic approximation
 * of TFLite's int8 kernel (fp16 vs int32 accumulate), so the off-hardware proof
 * compares the glue against an independent oracle that performs the IDENTICAL
 * dequant->fp16-conv->requant numeric path (not against TFLite's int8 reference);
 * a PASS is then a clean statement about the layout + quant plumbing.
 *
 * TFLite affine quantization: real = scale * (q - zero_point).
 *   input/output  per-tensor   (single scale, single zero_point)
 *   filter        per-axis      (scale[oc], zero_point[oc]; symmetric => zp 0)
 *   bias          int32, implicit scale = in_scale * w_scale[oc], zero_point 0
 * Zero-padding a quantized conv pads with the input zero_point, i.e. real 0 — so
 * the dequant pad region is a literal fp16 0 (memset), exactly as the float path.
 * ========================================================================== */

/* Read one stored quantized element as a signed int (sign-correct for both
 * int8 and uint8 storage). */
static inline int rocket_qread(const void *src, size_t k, int is_unsigned)
{
    return is_unsigned ? (int)((const unsigned char *)src)[k]
                       : (int)((const signed char  *)src)[k];
}

/*
 * input NHWC int8/uint8 -> NCHW fp16 (dequantized), padding materialized. Mirror
 * of rocket_in_nhwc_to_nchw_pad: the real input lands at (pad_top,pad_left) in an
 * [IC][IHp][IWp] buffer and the halo is real 0 (= padding with the input
 * zero_point). dst is fully written.
 */
static inline void rocket_in_q_to_nchw_pad(
        const void *src, int is_unsigned, float in_scale, int in_zp,
        _Float16 *dst, int IC, int IH, int IW,
        int pad_top, int pad_left, int IHp, int IWp)
{
    /* A stored byte's dequant depends only on its value (256 of them) and the per-tensor
     * in_scale/in_zp, so precompute the fp16 result for every byte once and index the
     * table by the raw byte -- a single load per element instead of a sign-extend,
     * subtract, multiply and fp16 convert. lut[b] folds in the signed/unsigned read
     * (rocket_qread) and is bit-identical to the inline arithmetic; the 256-entry build
     * amortizes over the IC*IH*IW body. */
    _Float16 lut[256];
    for (int b = 0; b < 256; b++) {
        int q = is_unsigned ? b : (int)(signed char)b;
        lut[b] = (_Float16)(in_scale * (float)(q - in_zp));
    }
    const unsigned char *s = (const unsigned char *)src;
    memset(dst, 0, (size_t)IC * IHp * IWp * sizeof(_Float16));
    for (int ih = 0; ih < IH; ih++)
        for (int iw = 0; iw < IW; iw++) {
            size_t spix = ((size_t)ih * IW + iw) * IC;
            size_t base = ((size_t)(ih + pad_top)) * IWp + (iw + pad_left);
            for (int ic = 0; ic < IC; ic++)
                dst[(size_t)ic * IHp * IWp + base] = lut[s[spix + ic]];
        }
}

/*
 * Conv filter TFLite [OC][KH][KW][IC] int8/uint8 -> driver [OC][IC][KH][KW] fp16,
 * dequantized per output channel (w_scale/w_zp length OC; w_zp may be NULL for
 * symmetric weights). Packed ONCE in Prepare (weights are model constants).
 */
static inline void rocket_filter_q_to_oihw(
        const void *src, int is_unsigned,
        const float *w_scale, const int *w_zp,
        _Float16 *dst, int OC, int KH, int KW, int IC)
{
    for (int oc = 0; oc < OC; oc++) {
        const float sc = w_scale[oc];
        const int   zp = w_zp ? w_zp[oc] : 0;
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) {
                    int q = rocket_qread(
                        src, (((size_t)oc * KH + kh) * KW + kw) * IC + ic, is_unsigned);
                    dst[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                        (_Float16)(sc * (float)(q - zp));
                }
    }
}

/*
 * Depthwise filter TFLite [1][KH][KW][C] int8/uint8 -> driver [C][KH][KW] fp16,
 * dequantized per channel. TFLite quantizes the depthwise filter PER-CHANNEL along
 * axis 3 (the C dim), so w_scale/w_zp have length C and entry c is the channel that
 * maps to output channel c (depth_multiplier 1 => OC==C) — i.e. the same flat
 * per-output-channel array the direct path uses, just sourced from a different axis.
 * w_zp may be NULL for symmetric weights. Packed ONCE in Prepare.
 */
static inline void rocket_dw_filter_q_to_chw(
        const void *src, int is_unsigned,
        const float *w_scale, const int *w_zp,
        _Float16 *dst, int C, int KH, int KW)
{
    for (int c = 0; c < C; c++) {
        const float sc = w_scale[c];
        const int   zp = w_zp ? w_zp[c] : 0;
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int q = rocket_qread(src, ((size_t)kh * KW + kw) * C + c, is_unsigned);
                dst[((size_t)c * KH + kh) * KW + kw] =
                    (_Float16)(sc * (float)(q - zp));
            }
    }
}

/*
 * Dequantize the int32 bias to float (real_bias[oc] = bias_q[oc]*in_scale*w_scale[oc]).
 * Computed once in Prepare; the requant epilogue then adds it in float exactly like
 * the float conv path adds its float bias.
 */
static inline void rocket_dequant_bias(
        const int32_t *bias_q, float in_scale, const float *w_scale,
        float *out, int OC)
{
    for (int oc = 0; oc < OC; oc++)
        out[oc] = (float)bias_q[oc] * in_scale * w_scale[oc];
}

/*
 * output driver NCHW fp16 -> TFLite NHWC int8/uint8, adding the (already
 * dequantized) bias, applying the fused activation in real space, then
 * requantizing: q = clamp(round(v/out_scale) + out_zp). Inverse of the input
 * dequant pack plus the conv epilogue. lrintf is round-to-nearest (ties to even);
 * the oracle uses the same, so the proof is exact.
 */
/* Output-row band [oh0,oh1) of the per-tensor fp16->int8/uint8 requant (parallelized
 * over rows like the native-int8 requant; the wrapper below is the [0,OH) whole). */
static inline void rocket_out_nchw_to_nhwc_q_band(
        const _Float16 *src, void *dst, int is_unsigned,
        int OC, int OH, int OW, const float *bias, int act,
        float out_scale, int out_zp, int oh0, int oh1)
{
    const int qmin = is_unsigned ? 0 : -128;
    const int qmax = is_unsigned ? 255 : 127;
    const float inv = 1.0f / out_scale;
    const size_t plane = (size_t)OH * OW;
    for (int oc = 0; oc < OC; oc++) {          // oc-outer: contiguous src plane, hoisted bias
        const float b = bias ? bias[oc] : 0.f;
        const _Float16 *sp = src + (size_t)oc * plane;
        for (int oh = oh0; oh < oh1; oh++)
            for (int ow = 0; ow < OW; ow++) {
                const size_t pix = (size_t)oh * OW + ow;
                float v = (float)sp[pix];
                v += b;                 // b is 0 when bias==NULL (hoisted per-oc above)
                v = rocket_apply_act(v, act);
                long q = (long)lrintf(v * inv) + out_zp;
                if (q < qmin) q = qmin;
                if (q > qmax) q = qmax;
                if (is_unsigned) ((unsigned char *)dst)[pix * OC + oc] = (unsigned char)q;
                else             ((signed char  *)dst)[pix * OC + oc] = (signed char)q;
            }
    }
}

static inline void rocket_out_nchw_to_nhwc_q(
        const _Float16 *src, void *dst, int is_unsigned,
        int OC, int OH, int OW, const float *bias, int act,
        float out_scale, int out_zp)
{
    rocket_out_nchw_to_nhwc_q_band(src, dst, is_unsigned, OC, OH, OW, bias, act,
                                   out_scale, out_zp, 0, OH);
}

/* ==========================================================================
 * Native int8 CONV_2D boundary — the EXACT int8 datapath.
 *
 * Unlike the dequant->fp16->requant approximation above, this path runs int8 x int8
 * -> int32 NATIVELY on the NPU (rocket_conv2d_int8) and requantizes the int32
 * accumulator on the host. No fp16 rounding in the accumulate: the result is
 * bit-identical to TFLite's int8 CPU kernel up to the final requant rounding (the NPU
 * does real int32 accumulate; only the multiplier rounding can differ by <=1).
 *
 * The NPU computes the RAW int8 sum  acc[oc] = sum_{ic,kh,kw} in_q * w_q  (no
 * zero-point subtraction). TFLite's int8 conv computes sum (in_q - in_zp)*(w_q - w_zp);
 * for the per-axis filter quant w_zp == 0 (symmetric), so
 *   sum (in_q - in_zp) * w_q = acc - in_zp * sum_kernel w_q.
 * That per-OC input-zp correction is FOLDED into an effective int32 bias together with
 * TFLite's int32 bias (which TFLite adds to the accumulator pre-scale, scale =
 * in_scale*w_scale[oc]):
 *   eff_bias[oc] = bias_q[oc] - in_zp * sum_kernel w_q[oc].
 * Boundary padding is materialized with the input zero-point (NOT real 0): a pad tap
 * contributes in_zp*w_q to acc, which the full-kernel eff_bias correction cancels — so
 * the boundary is exactly TFLite's. Signed int8 in/weight/out only (the common modern
 * scheme); a uint8 model stays on the dequant->fp16 path.
 * ========================================================================== */

/* int8 conv filter TFLite [OC][KH][KW][IC] -> driver [OC][IC][KH][KW], kept int8 (the
 * raw w_q the native int32-raw conv multiplies). Packed ONCE in Prepare. */
static inline void rocket_filter_i8_to_oihw(
        const signed char *src, int8_t *dst, int OC, int KH, int KW, int IC)
{
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    dst[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                        src[(((size_t)oc * KH + kh) * KW + kw) * IC + ic];
}

/* Per-OC effective int32 bias for the native-int8 DIRECT requant:
 *   eff_bias[oc] = (bias_q ? bias_q[oc] : 0) - in_zp * sum_kernel w_q[oc].
 * w_oihw is the int8 filter in driver layout [OC][IC][KH][KW] (the per-OC sum is
 * layout-independent). bias_q may be NULL (no bias). Computed ONCE in Prepare. */
static inline void rocket_eff_bias_per_axis(
        const int8_t *w_oihw, const int32_t *bias_q, int in_zp,
        int32_t *eff_bias, int OC, int IC, int KH, int KW)
{
    const size_t per_oc = (size_t)IC * KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int32_t wsum = 0;
        const int8_t *w = w_oihw + (size_t)oc * per_oc;
        for (size_t i = 0; i < per_oc; i++) wsum += w[i];
        // long intermediate (matches the uint8 sibling): in_zp*wsum can exceed int32
        // range for a large IC*KH*KW before the store back into the int32 field.
        eff_bias[oc] = (int32_t)((bias_q ? bias_q[oc] : 0) - (long)in_zp * wsum);
    }
}

/* int8 input NHWC [IH][IW][IC] -> NCHW [IC][IHp][IWp] int8, padding materialized with
 * the input zero-point (real input at (pad_top,pad_left), halo = in_zp). The native
 * conv runs pad_top=pad_left=0 over it; the in_zp halo is cancelled by the eff_bias
 * correction, reproducing TFLite SAME/VALID exactly. dst fully written. */
static inline void rocket_in_i8_to_nchw_pad(
        const signed char *src, int8_t *dst, int IC, int IH, int IW, int in_zp,
        int pad_top, int pad_left, int IHp, int IWp)
{
    memset(dst, (unsigned char)(signed char)in_zp, (size_t)IC * IHp * IWp * sizeof(int8_t));
    for (int ih = 0; ih < IH; ih++)
        for (int iw = 0; iw < IW; iw++) {
            const signed char *s = src + ((size_t)ih * IW + iw) * IC;
            size_t base = ((size_t)(ih + pad_top)) * IWp + (iw + pad_left);
            for (int ic = 0; ic < IC; ic++)
                dst[(size_t)ic * IHp * IWp + base] = (int8_t)s[ic];
        }
}

/* Depthwise int8 filter TFLite [1][KH][KW][C] -> driver [C][KH][KW], kept int8 (raw
 * w_q). The native int8-out DW path (rocket_conv2d_dw_int8) does its own (C/G,KH,KW,G)
 * cube scatter + uint8-domain centering from this. Per-TENSOR quant only. Packed ONCE. */
static inline void rocket_dw_filter_i8_to_chw(
        const signed char *src, int8_t *dst, int C, int KH, int KW)
{
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                dst[((size_t)c * KH + kh) * KW + kw] =
                    src[((size_t)kh * KW + kw) * C + c];
}

/* int8 NCHW [C][OH][OW] (model domain, from rocket_conv2d_dw_int8) -> TFLite NHWC int8.
 * The DW int8-out path requants ON-CHIP, so this is a pure transpose (no bias/act/requant
 * — all already applied). */
static inline void rocket_out_nchw_to_nhwc_i8(
        const int8_t *src, signed char *dst, int C, int OH, int OW)
{
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            size_t dpix = ((size_t)oh * OW + ow) * C;
            for (int c = 0; c < C; c++)
                dst[dpix + c] = (signed char)src[((size_t)c * OH + oh) * OW + ow];
        }
}

/* Native-int8 DIRECT conv requant: int32 NCHW accumulator (from rocket_conv2d_int8) ->
 * int8/uint8 NHWC. acc[oc,oh,ow] is the raw int8 sum; eff_bias[oc] folds the TFLite
 * int32 bias + the input zero-point correction (rocket_eff_bias_per_axis). Per-OC:
 *   real = in_scale * w_scale[oc] * (acc + eff_bias[oc]);  real = act(real);
 *   q    = clamp(lrintf(real / out_scale) + out_zp).
 * Matches CPU TFLite to within the requant rounding (<=1); exact vs a same-math oracle.
 * eff_bias may be NULL (no bias, in_zp==0). */
/* Output-row band [oh0,oh1) of the native signed/unsigned-int8 DIRECT requant
 * (parallelized over rows exactly like the uint8 band above; the wrapper is [0,OH)). */
static inline void rocket_out_nchw_to_nhwc_q_per_axis_band(
        const int32_t *src, void *dst, int is_unsigned, int OC, int OH, int OW,
        const int32_t *eff_bias, int act, float in_scale, const float *w_scale,
        float out_scale, int out_zp, int oh0, int oh1)
{
    const int qmin = is_unsigned ? 0 : -128;
    const int qmax = is_unsigned ? 255 : 127;
    const float inv = 1.0f / out_scale;
    const size_t plane = (size_t)OH * OW;
    /* The band's rows [oh0,oh1) are contiguous in the NCHW src plane (pix=oh*OW+ow), so
     * flatten to one [p0,p1) pixel run — a longer vectorizable stretch than per-row. */
    const size_t p0 = (size_t)oh0 * OW, p1 = (size_t)oh1 * OW;
#if defined(__ARM_NEON)
    /* oc-OUTER -> the dst write dst[pix*OC+oc] is strided by OC, so (unlike the M-major
     * 1x1 path) the store can't be a vector store; but the expensive per-pixel compute
     * (the int->float requant + lrintf) vectorizes over 4 pixels of the contiguous src
     * plane and the 4 results scatter out one byte each. int32 acc + a single scale mul +
     * vcvtnq (= lrintf RNE) => bit-identical to the scalar tail (same as the M-major path). */
    static int scalar_only = -1;   /* ROCKET_REQUANT_SCALAR=1 forces scalar (A/B + fallback) */
    if (scalar_only < 0) scalar_only = (getenv("ROCKET_REQUANT_SCALAR") != NULL);
    const int32x4_t vlo = vdupq_n_s32(qmin), vhi = vdupq_n_s32(qmax), vzp = vdupq_n_s32(out_zp);
    const float32x4_t vinv = vdupq_n_f32(inv);
#endif
    for (int oc = 0; oc < OC; oc++) {          // oc-outer: contiguous src plane, hoisted scale
        const int32_t eb = eff_bias ? eff_bias[oc] : 0;
        const float scale = in_scale * w_scale[oc];
        const int32_t *sp = src + (size_t)oc * plane;
        size_t pix = p0;
#if defined(__ARM_NEON)
      if (!scalar_only) {
        const int32x4_t veb = vdupq_n_s32(eb);
        const float32x4_t vscale = vdupq_n_f32(scale);
        for (; pix + 4 <= p1; pix += 4) {
            int32x4_t acc = vaddq_s32(vld1q_s32(sp + pix), veb);
            float32x4_t v = vmulq_f32(vscale, vcvtq_f32_s32(acc));
            v = rocket_apply_act_f32x4(v, act);
            int32x4_t qi = vminq_s32(vmaxq_s32(
                    vaddq_s32(vcvtnq_s32_f32(vmulq_f32(v, vinv)), vzp), vlo), vhi);
            if (is_unsigned) {
                unsigned char *d = (unsigned char *)dst + oc;
                d[(pix + 0) * OC] = (unsigned char)vgetq_lane_s32(qi, 0);
                d[(pix + 1) * OC] = (unsigned char)vgetq_lane_s32(qi, 1);
                d[(pix + 2) * OC] = (unsigned char)vgetq_lane_s32(qi, 2);
                d[(pix + 3) * OC] = (unsigned char)vgetq_lane_s32(qi, 3);
            } else {
                signed char *d = (signed char *)dst + oc;
                d[(pix + 0) * OC] = (signed char)vgetq_lane_s32(qi, 0);
                d[(pix + 1) * OC] = (signed char)vgetq_lane_s32(qi, 1);
                d[(pix + 2) * OC] = (signed char)vgetq_lane_s32(qi, 2);
                d[(pix + 3) * OC] = (signed char)vgetq_lane_s32(qi, 3);
            }
        }
      }
#endif
        for (; pix < p1; pix++) {              // scalar tail / non-NEON
            int32_t acc = sp[pix] + eb;
            float v = scale * (float)acc;
            v = rocket_apply_act(v, act);
            long q = (long)lrintf(v * inv) + out_zp;
            if (q < qmin) q = qmin;
            if (q > qmax) q = qmax;
            if (is_unsigned) ((unsigned char *)dst)[pix * OC + oc] = (unsigned char)q;
            else             ((signed char  *)dst)[pix * OC + oc] = (signed char)q;
        }
    }
}

static inline void rocket_out_nchw_to_nhwc_q_per_axis(
        const int32_t *src, void *dst, int is_unsigned, int OC, int OH, int OW,
        const int32_t *eff_bias, int act, float in_scale, const float *w_scale,
        float out_scale, int out_zp)
{
    rocket_out_nchw_to_nhwc_q_per_axis_band(src, dst, is_unsigned, OC, OH, OW, eff_bias,
            act, in_scale, w_scale, out_scale, out_zp, 0, OH);
}

/* ==========================================================================
 * Native UINT8 CONV_2D boundary (uint8 recenter path) — the EXACT uint8 datapath.
 *
 * Most stock detectors in the wild (coral MobileDet, MediaPipe) are uint8 with a
 * NON-symmetric weight zero-point (the real MobileDet conv weights span w_zp 91..177,
 * almost none at 128), so they cannot use the signed-int8 native path (which requires
 * symmetric weights); without this path they fall back to the fp16 approximation. This
 * path runs them on the SAME native int32-raw datapath (rocket_conv2d_int8) by RE-CENTERING
 * the uint8 operands into signed int8 on the host and folding the centering algebra
 * into the requant. No new regcmd: the NPU still computes a raw int8xint8->int32 sum.
 *
 * The NPU is fed  x = in_q - 128  and  y = w_q - 128  (both always land in [-128,127]
 * for any uint8 byte, so -128 is the unique always-safe recenter; recentering by the
 * actual zero-point would overflow when zp is far from 128, and the model has in_zp as
 * low as 0). It computes  acc = sum_k x*y. TFLite wants  T = sum_k (in_q-in_zp)*(w_q-w_zp).
 * Writing  alpha = 128-in_zp,  beta[oc] = 128-w_zp[oc]:
 *     T = acc + beta[oc]*Sx + alpha*Wy[oc] + N*alpha*beta[oc]
 * where  Sx[oh,ow] = sum_window x  (the per-output-pixel box-sum of the recentered input,
 * INCLUDING the materialized padding taps), Wy[oc] = sum_kernel y (a per-OC constant),
 * and N = IC*KH*KW (the full window size). The alpha*Wy + N*alpha*beta terms are per-OC
 * constants folded with TFLite's int32 bias into eff_bias[oc]; the beta[oc]*Sx term is
 * position-dependent (it is the price of an asymmetric weight zero-point) and is added in
 * the requant from a host-computed box-sum. When every w_zp==128 (symmetric uint8) beta==0
 * and the box-sum vanishes (pass Sx=NULL) — the path then matches the signed-int8 shape.
 *
 * Boundary padding is materialized with in_zp (real byte in_zp, recentered to in_zp-128):
 * a pad tap has x+alpha = 0, so it contributes 0 to T exactly like TFLite's in_zp padding,
 * while still counting as a full tap in Sx / Wy / N (which is why padding must be
 * materialized so every window is full-size — same rule as the signed path).
 * ========================================================================== */

/* uint8 conv filter TFLite [OC][KH][KW][IC] -> driver [OC][IC][KH][KW], RE-CENTERED to
 * signed int8 (y = w_q - 128). Packed ONCE in Prepare (weights are model constants). */
static inline void rocket_filter_u8_to_oihw(
        const unsigned char *src, int8_t *dst, int OC, int KH, int KW, int IC)
{
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    dst[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                        (int8_t)((int)src[(((size_t)oc * KH + kh) * KW + kw) * IC + ic] - 128);
}

/* uint8 input NHWC [IH][IW][IC] -> NCHW [IC][IHp][IWp] signed int8, RE-CENTERED (x =
 * in_q - 128) with padding materialized as the input zero-point (halo byte = in_zp-128,
 * so x+alpha=0 there). The native conv runs pad_top=pad_left=0 over it. dst fully written. */
static inline void rocket_in_u8_to_nchw_pad(
        const unsigned char *src, int8_t *dst, int IC, int IH, int IW, int in_zp,
        int pad_top, int pad_left, int IHp, int IWp)
{
    memset(dst, (unsigned char)(signed char)(in_zp - 128), (size_t)IC * IHp * IWp * sizeof(int8_t));
    for (int ih = 0; ih < IH; ih++)
        for (int iw = 0; iw < IW; iw++) {
            const unsigned char *s = src + ((size_t)ih * IW + iw) * IC;
            size_t base = ((size_t)(ih + pad_top)) * IWp + (iw + pad_left);
            for (int ic = 0; ic < IC; ic++)
                dst[(size_t)ic * IHp * IWp + base] = (int8_t)((int)s[ic] - 128);
        }
}

/* Per-OC effective int32 bias for the native-uint8 requant:
 *   eff_bias[oc] = (bias_q?bias_q[oc]:0) + alpha*Wy[oc] + N*alpha*beta[oc],
 * with alpha = 128-in_zp, beta[oc] = 128-w_zp[oc], Wy[oc] = sum_kernel y[oc] (the recentered
 * weight sum, layout-independent), N = IC*KH*KW. w_oihw is the recentered int8 filter from
 * rocket_filter_u8_to_oihw; w_zp is the per-OC (or broadcast per-tensor) uint8 weight zero
 * point. bias_q may be NULL. Computed ONCE in Prepare. (long intermediate avoids overflow.) */
static inline void rocket_eff_bias_u8_per_axis(
        const int8_t *w_oihw, const int32_t *bias_q, int in_zp, const int *w_zp,
        int32_t *eff_bias, int OC, int IC, int KH, int KW)
{
    const size_t per_oc = (size_t)IC * KH * KW;
    const long N = (long)per_oc;
    const long alpha = 128 - in_zp;
    for (int oc = 0; oc < OC; oc++) {
        long wy = 0;
        const int8_t *w = w_oihw + (size_t)oc * per_oc;
        for (size_t i = 0; i < per_oc; i++) wy += w[i];
        const long beta = 128 - (w_zp ? w_zp[oc] : 128);
        eff_bias[oc] = (int32_t)((bias_q ? bias_q[oc] : 0) + alpha * wy + N * alpha * beta);
    }
}

/* Per-output-pixel box-sum Sx[oh*OW+ow] = sum over the conv window of the recentered input
 * x (= the rocket_in_u8_to_nchw_pad buffer, padding included). This is the position-dependent
 * correction the asymmetric weight zero-point needs; it sums the SAME padded NCHW buffer the
 * NPU conv reduces, so the window taps line up tap-for-tap with acc. Only needed when some
 * w_zp!=128 (else beta==0 and the caller skips this). Pure adds, no MACs. */
/* Output-row band [oh0,oh1) of the box-sum (parallelized over rows like the requant;
 * the wrapper below is the [0,OH) whole). Disjoint Sx rows => no races, bit-identical. */
static inline void rocket_in_window_sum_i8_band(
        const int8_t *in_nchw, int32_t *Sx, int IC, int IHp, int IWp,
        int OH, int OW, int KH, int KW, int sy, int sx, int dy, int dx,
        int oh0, int oh1)
{
    (void)OH;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* Contiguous-column box-sum (unit output stride AND dilation, the common post-stem direct
     * conv). Two NEON realizations, both bit-identical to the scalar accumulate below (the sum
     * is over N=IC*KH*KW int8 taps in [-128,127] -> int32; pure integer adds, order-independent):
     *   mode 0 (default, separable): for KW>1 split the IC*KH*KW window sum into a channel+row
     *           reduce T[iw] = sum_{ic,kh} in[ic, oh*sy+kh*dy, iw] (16-wide over the input
     *           columns) followed by a horizontal KW window-sum Sx[oh,ow]=sum_{kw} T[ow+kw]
     *           (4-wide). This cuts the SIMD column-pass count from IC*KH*KW to IC*KH + KW
     *           (a ~KW-fold reduction on 3x3/5x5 kernels at IC>=KW); both passes stay vectorized.
     *   mode 1 (per-window): the prior form — for each (ic,kh,kw) add the contiguous tap row
     *           in_row[ow+kw] across the Sx row 16 columns at a time.
     * ROCKET_BOXSUM_MODE overrides (0 separable / 1 per-window / 2 scalar) for A/B + as the
     * non-NEON fallback. KW==1 (channel reduce, nothing to separate) always uses mode 1. */
    static int boxsum_mode = -1;
    if (boxsum_mode < 0) {
        const char *e = getenv("ROCKET_BOXSUM_MODE");
        boxsum_mode = e ? atoi(e) : 0;
    }
    if (boxsum_mode != 2 && sx == 1 && dx == 1) {
        if (boxsum_mode == 0 && KW > 1) {
            /* T spans the input columns the windows touch: ow in [0,OW), kw in [0,KW),
             * tap = ow + kw, so the max index is (OW-1)+(KW-1) => OW+KW-1 columns. */
            const int Tcols = OW + KW - 1;
            int32_t Tstack[1024];
            int32_t *T = (Tcols <= 1024) ? Tstack
                                         : (int32_t *)malloc((size_t)Tcols * sizeof(int32_t));
            for (int oh = oh0; oh < oh1; oh++) {
                int32_t *srow = Sx + (size_t)oh * OW;
                /* (1) channel + kernel-height reduce into T over the contiguous input columns */
                memset(T, 0, (size_t)Tcols * sizeof(int32_t));
                for (int ic = 0; ic < IC; ic++) {
                    const int8_t *plane = in_nchw + (size_t)ic * IHp * IWp;
                    for (int kh = 0; kh < KH; kh++) {
                        const int8_t *row = plane + (size_t)(oh * sy + kh * dy) * IWp;
                        int iw = 0;
                        for (; iw + 16 <= Tcols; iw += 16) {
                            int8x16_t v  = vld1q_s8(row + iw);
                            int16x8_t lo = vmovl_s8(vget_low_s8(v));
                            int16x8_t hi = vmovl_s8(vget_high_s8(v));
                            vst1q_s32(T + iw,      vaddw_s16(vld1q_s32(T + iw),      vget_low_s16(lo)));
                            vst1q_s32(T + iw + 4,  vaddw_s16(vld1q_s32(T + iw + 4),  vget_high_s16(lo)));
                            vst1q_s32(T + iw + 8,  vaddw_s16(vld1q_s32(T + iw + 8),  vget_low_s16(hi)));
                            vst1q_s32(T + iw + 12, vaddw_s16(vld1q_s32(T + iw + 12), vget_high_s16(hi)));
                        }
                        for (; iw < Tcols; iw++) T[iw] += row[iw];
                    }
                }
                /* (2) horizontal KW window-sum: srow[ow] = sum_{kw} T[ow+kw] */
                int ow = 0;
                for (; ow + 4 <= OW; ow += 4) {
                    int32x4_t acc = vld1q_s32(T + ow);
                    for (int kw = 1; kw < KW; kw++) acc = vaddq_s32(acc, vld1q_s32(T + ow + kw));
                    vst1q_s32(srow + ow, acc);
                }
                for (; ow < OW; ow++) {
                    int32_t s = 0;
                    for (int kw = 0; kw < KW; kw++) s += T[ow + kw];
                    srow[ow] = s;
                }
            }
            if (T != Tstack) free(T);
            return;
        }
        for (int oh = oh0; oh < oh1; oh++) {
            int32_t *srow = Sx + (size_t)oh * OW;
            memset(srow, 0, (size_t)OW * sizeof(int32_t));
            for (int ic = 0; ic < IC; ic++) {
                const int8_t *plane = in_nchw + (size_t)ic * IHp * IWp;
                for (int kh = 0; kh < KH; kh++) {
                    const int8_t *row = plane + (size_t)(oh * sy + kh * dy) * IWp;
                    for (int kw = 0; kw < KW; kw++) {
                        const int8_t *p = row + kw;        /* tap for ow=0; ow stride 1 */
                        int ow = 0;
                        for (; ow + 16 <= OW; ow += 16) {
                            int8x16_t v  = vld1q_s8(p + ow);
                            int16x8_t lo = vmovl_s8(vget_low_s8(v));
                            int16x8_t hi = vmovl_s8(vget_high_s8(v));
                            vst1q_s32(srow + ow,      vaddw_s16(vld1q_s32(srow + ow),      vget_low_s16(lo)));
                            vst1q_s32(srow + ow + 4,  vaddw_s16(vld1q_s32(srow + ow + 4),  vget_high_s16(lo)));
                            vst1q_s32(srow + ow + 8,  vaddw_s16(vld1q_s32(srow + ow + 8),  vget_low_s16(hi)));
                            vst1q_s32(srow + ow + 12, vaddw_s16(vld1q_s32(srow + ow + 12), vget_high_s16(hi)));
                        }
                        for (; ow < OW; ow++) srow[ow] += p[ow];
                    }
                }
            }
        }
        return;
    }
#endif
    for (int oh = oh0; oh < oh1; oh++)
        for (int ow = 0; ow < OW; ow++) {
            long s = 0;
            for (int ic = 0; ic < IC; ic++) {
                const int8_t *plane = in_nchw + (size_t)ic * IHp * IWp;
                for (int kh = 0; kh < KH; kh++) {
                    const int8_t *row = plane + (size_t)(oh * sy + kh * dy) * IWp;
                    for (int kw = 0; kw < KW; kw++)
                        s += row[ow * sx + kw * dx];
                }
            }
            Sx[(size_t)oh * OW + ow] = (int32_t)s;
        }
}

static inline void rocket_in_window_sum_i8(
        const int8_t *in_nchw, int32_t *Sx, int IC, int IHp, int IWp,
        int OH, int OW, int KH, int KW, int sy, int sx, int dy, int dx)
{
    rocket_in_window_sum_i8_band(in_nchw, Sx, IC, IHp, IWp, OH, OW, KH, KW,
                                 sy, sx, dy, dx, 0, OH);
}

/* Native-uint8 DIRECT conv requant: int32 NCHW accumulator (raw sum_k x*y from
 * rocket_conv2d_int8 on the recentered operands) -> uint8 NHWC. Per (oc,oh,ow):
 *   acc_total = acc + eff_bias[oc] + (128-w_zp[oc])*Sx[oh,ow];   (== TFLite's T + bias_q)
 *   real = in_scale*w_scale[oc]*acc_total;  real = act(real);
 *   q = clamp(lrintf(real/out_scale) + out_zp, 0, 255).
 * Sx may be NULL (symmetric weights, beta==0). w_zp is per-OC (or broadcast per-tensor).
 * Output is always uint8. Matches CPU TFLite to within the requant rounding (<=1); exact
 * vs a same-math integer oracle. (long intermediate for the acc_total triple sum.) */
/* Output-row band [oh0,oh1) of the native-uint8 requant. The full requant is the
 * delegate's single largest host cost, so it is parallelized across the big cores by
 * splitting the output ROWS (each band writes a disjoint dst region — no races, and
 * bit-identical to the whole-tensor pass). The wrapper below is the [0,OH) whole. */
static inline void rocket_out_nchw_to_nhwc_q_per_axis_u8_band(
        const int32_t *src, unsigned char *dst, int OC, int OH, int OW,
        const int32_t *eff_bias, const int *w_zp, const int32_t *Sx, int act,
        float in_scale, const float *w_scale, float out_scale, int out_zp,
        int oh0, int oh1)
{
    // oc-OUTER: src[oc] is a contiguous OH*OW plane (the NCHW accumulator), so reading it
    // and Sx by row is sequential (prefetch-friendly), and the per-OC constants
    // (eff_bias/beta/scale) hoist out of the pixel loop instead of reloading every element.
    // The transpose cost moves to the strided uint8 dst write. Bit-identical to the
    // pixel-outer form (same float expression order). Disjoint dst rows per band => no race.
    const float inv = 1.0f / out_scale;
    const size_t plane = (size_t)OH * OW;
    const size_t p0 = (size_t)oh0 * OW, p1 = (size_t)oh1 * OW;   // contiguous band run
#if defined(__ARM_NEON)
    /* Same strided-store vectorization as the signed sibling: compute 4 contiguous pixels
     * at a time (acc + eb + beta*Sx, the requant, lrintf via vcvtnq), scatter the 4 bytes
     * out at dst[pix*OC+oc]. All intermediates fit int32 (sp + eff_bias + beta*Sx <~1.3e8,
     * the same bound the M-major u8 path relies on), so the int32 NEON arithmetic equals the
     * scalar `long` math and the result is byte-identical to the scalar tail below. */
    static int scalar_only = -1;   /* ROCKET_REQUANT_SCALAR=1 forces scalar (A/B + fallback) */
    if (scalar_only < 0) scalar_only = (getenv("ROCKET_REQUANT_SCALAR") != NULL);
    const int32x4_t vzp = vdupq_n_s32(out_zp), vlo = vdupq_n_s32(0), vhi = vdupq_n_s32(255);
    const float32x4_t vinv = vdupq_n_f32(inv);
#endif
    for (int oc = 0; oc < OC; oc++) {
        const long  eb    = eff_bias ? eff_bias[oc] : 0;
        const long  beta  = Sx ? (128 - (w_zp ? w_zp[oc] : 128)) : 0;
        const float scale = in_scale * w_scale[oc];
        const int32_t *sp = src + (size_t)oc * plane;
        size_t pix = p0;
#if defined(__ARM_NEON)
      if (!scalar_only) {
        const int32x4_t veb = vdupq_n_s32((int32_t)eb), vbeta = vdupq_n_s32((int32_t)beta);
        const float32x4_t vscale = vdupq_n_f32(scale);
        for (; pix + 4 <= p1; pix += 4) {
            int32x4_t acc = vaddq_s32(vld1q_s32(sp + pix), veb);
            if (Sx) acc = vaddq_s32(acc, vmulq_s32(vbeta, vld1q_s32(Sx + pix)));
            float32x4_t v = vmulq_f32(vscale, vcvtq_f32_s32(acc));
            v = rocket_apply_act_f32x4(v, act);
            int32x4_t qi = vminq_s32(vmaxq_s32(
                    vaddq_s32(vcvtnq_s32_f32(vmulq_f32(v, vinv)), vzp), vlo), vhi);
            unsigned char *d = dst + oc;
            d[(pix + 0) * OC] = (unsigned char)vgetq_lane_s32(qi, 0);
            d[(pix + 1) * OC] = (unsigned char)vgetq_lane_s32(qi, 1);
            d[(pix + 2) * OC] = (unsigned char)vgetq_lane_s32(qi, 2);
            d[(pix + 3) * OC] = (unsigned char)vgetq_lane_s32(qi, 3);
        }
      }
#endif
        for (; pix < p1; pix++) {              // scalar tail / non-NEON
            long acc = (long)sp[pix] + eb;
            if (Sx) acc += beta * Sx[pix];
            float v = scale * (float)acc;
            v = rocket_apply_act(v, act);
            long q = (long)lrintf(v * inv) + out_zp;
            if (q < 0)   q = 0;
            if (q > 255) q = 255;
            dst[pix * OC + oc] = (unsigned char)q;
        }
    }
}

static inline void rocket_out_nchw_to_nhwc_q_per_axis_u8(
        const int32_t *src, unsigned char *dst, int OC, int OH, int OW,
        const int32_t *eff_bias, const int *w_zp, const int32_t *Sx, int act,
        float in_scale, const float *w_scale, float out_scale, int out_zp)
{
    rocket_out_nchw_to_nhwc_q_per_axis_u8_band(src, dst, OC, OH, OW, eff_bias, w_zp, Sx,
            act, in_scale, w_scale, out_scale, out_zp, 0, OH);
}

/* ==========================================================================
 * Native int8/uint8 1x1-pointwise-via-MATMUL boundary.
 *
 * A 1x1 stride-1 CONV_2D is a matmul: out[m,oc] = sum_c in[m,c]*w[oc,c] over the
 * M=H*W spatial positions. The NHWC input [1,IH,IW,IC] is ALREADY the matmul
 * A[M=H*W, K=IC] row-major, and the matmul output C[M,OC] is ALREADY NHWC — so the
 * resident int8 matmul (rocket_matmul_int8_prepacked, int32-raw, N fanned across the
 * 3 NPU cores) runs these 1x1s with NO layout transpose on either side, and splits the
 * OUTPUT columns across all cores where the single-tile conv path pins them to ONE. The
 * integer accumulate is exact + order-independent, so the requant input is bit-identical
 * to the native conv path's int32 accumulator, hence the OUTPUT is byte-identical.
 *
 * The matmul requires K%32, N%32, M%4||M==1; K (IC) and N (OC) are zero-padded to the
 * next multiple of 32 by the caller (the padded operands carry zeros — transparent to the
 * integer sum and to the per-OC eff_bias, which sums only the real IC weights), so the
 * accumulator C has row stride Ncol = N_padded while only the first OC columns are real.
 * These are the M-major (no-transpose) requant + the uint8 recenter/box-sum, the matmul
 * siblings of the NCHW helpers above (same float expression order => byte-identical).
 * ========================================================================== */

/* Recenter uint8 NHWC input [M][IC] -> int8 [M][Kp] (x = in_q - 128; columns IC..Kp-1
 * zero-padded for the matmul's K%32). When Sx != NULL also compute the per-row channel
 * box-sum Sx[m] = sum_{c<IC} x[m,c] — the 1x1 conv's one-pixel window sum the asymmetric
 * weight zero-point needs (== rocket_in_window_sum_i8 for KH=KW=1, no pad). Band over
 * rows [m0,m1); disjoint rows => race-free, bit-identical to one pass. */
static inline void rocket_recenter_u8_mk_band(
        const unsigned char *src, int8_t *A, int32_t *Sx,
        int M, int IC, int Kp, int m0, int m1)
{
    (void)M;
    for (int m = m0; m < m1; m++) {
        const unsigned char *s = src + (size_t)m * IC;
        int8_t *a = A + (size_t)m * Kp;
        long sum = 0;
        for (int c = 0; c < IC; c++) { int x = (int)s[c] - 128; a[c] = (int8_t)x; sum += x; }
        for (int c = IC; c < Kp; c++) a[c] = 0;
        if (Sx) Sx[m] = (int32_t)sum;
    }
}
static inline void rocket_recenter_u8_mk(
        const unsigned char *src, int8_t *A, int32_t *Sx, int M, int IC, int Kp)
{
    rocket_recenter_u8_mk_band(src, A, Sx, M, IC, Kp, 0, M);
}

/* Copy int8 NHWC input [M][IC] -> int8 [M][Kp], zero-padding columns IC..Kp-1 for the
 * matmul's K%32 (used only when Kp>IC; when IC%32==0 the caller feeds the raw input
 * directly as A — no copy). Band over rows [m0,m1). */
static inline void rocket_pad_i8_mk_band(
        const signed char *src, int8_t *A, int M, int IC, int Kp, int m0, int m1)
{
    (void)M;
    for (int m = m0; m < m1; m++) {
        memcpy(A + (size_t)m * Kp, src + (size_t)m * IC, (size_t)IC);
        for (int c = IC; c < Kp; c++) A[(size_t)m * Kp + c] = 0;
    }
}

/* Native int8 1x1-matmul DIRECT requant: int32 [M][Ncol] accumulator (Ncol = padded N,
 * the row stride; only the first OC columns are real) -> int8/uint8 NHWC [M][OC]. M-major
 * (both src and dst rows contiguous — no transpose). Same per-OC eff_bias + scale + act +
 * lrintf requant as rocket_out_nchw_to_nhwc_q_per_axis, with the IDENTICAL float
 * expression order (scale = in_scale*w_scale[oc] then *acc), so the output is byte-identical
 * to the conv path's. eff_bias may be NULL. Band over rows [m0,m1). */
static inline void rocket_out_mn_to_nhwc_q_per_axis_band(
        const int32_t *src, void *dst, int is_unsigned, int M, int OC, int Ncol,
        const int32_t *eff_bias, int act, float in_scale, const float *w_scale,
        float out_scale, int out_zp, int m0, int m1)
{
    (void)M;
    const int qmin = is_unsigned ? 0 : -128;
    const int qmax = is_unsigned ? 255 : 127;
    const float inv = 1.0f / out_scale;
    const int has_eb = eff_bias != NULL;
#if defined(__ARM_NEON)   /* OC contiguous both sides; same bit-exact NEON requant as the u8 path */
    static int scalar_only = -1;
    if (scalar_only < 0) scalar_only = (getenv("ROCKET_REQUANT_SCALAR") != NULL);
    const int32x4_t vlo = vdupq_n_s32(qmin), vhi = vdupq_n_s32(qmax), vzp = vdupq_n_s32(out_zp);
    const float32x4_t vis = vdupq_n_f32(in_scale), vinv = vdupq_n_f32(inv);
#endif
    for (int m = m0; m < m1; m++) {
        const int32_t *sp = src + (size_t)m * Ncol;
        int oc = 0;
#if defined(__ARM_NEON)
      if (!scalar_only) {
        for (; oc + 8 <= OC; oc += 8) {
            int32x4_t q[2];
            for (int h = 0; h < 2; h++) {
                int o = oc + 4 * h;
                int32x4_t acc = vld1q_s32(sp + o);
                if (has_eb) acc = vaddq_s32(acc, vld1q_s32(eff_bias + o));
                float32x4_t v = vmulq_f32(vmulq_f32(vis, vld1q_f32(w_scale + o)), vcvtq_f32_s32(acc));
                v = rocket_apply_act_f32x4(v, act);
                int32x4_t qi = vaddq_s32(vcvtnq_s32_f32(vmulq_f32(v, vinv)), vzp);
                q[h] = vminq_s32(vmaxq_s32(qi, vlo), vhi);
            }
            if (is_unsigned)
                vst1_u8((unsigned char *)dst + (size_t)m * OC + oc,
                        vqmovn_u16(vcombine_u16(vqmovun_s32(q[0]), vqmovun_s32(q[1]))));
            else
                vst1_s8((signed char *)dst + (size_t)m * OC + oc,
                        vqmovn_s16(vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]))));
        }
      }
#endif
        for (; oc < OC; oc++) {
            int32_t acc = sp[oc] + (eff_bias ? eff_bias[oc] : 0);
            float v = (in_scale * w_scale[oc]) * (float)acc;
            v = rocket_apply_act(v, act);
            long q = (long)lrintf(v * inv) + out_zp;
            if (q < qmin) q = qmin;
            if (q > qmax) q = qmax;
            if (is_unsigned) ((unsigned char *)dst)[(size_t)m * OC + oc] = (unsigned char)q;
            else             ((signed char  *)dst)[(size_t)m * OC + oc] = (signed char)q;
        }
    }
}
static inline void rocket_out_mn_to_nhwc_q_per_axis(
        const int32_t *src, void *dst, int is_unsigned, int M, int OC, int Ncol,
        const int32_t *eff_bias, int act, float in_scale, const float *w_scale,
        float out_scale, int out_zp)
{
    rocket_out_mn_to_nhwc_q_per_axis_band(src, dst, is_unsigned, M, OC, Ncol, eff_bias,
            act, in_scale, w_scale, out_scale, out_zp, 0, M);
}

/* Native uint8 1x1-matmul DIRECT requant: int32 [M][Ncol] (Ncol = padded N) -> uint8 NHWC
 * [M][OC], adding the per-OC eff_bias + the asymmetric-weight box-sum term beta[oc]*Sx[m]
 * (Sx the per-row channel sum from rocket_recenter_u8_mk; NULL when all w_zp==128). Same
 * math + float expression order as rocket_out_nchw_to_nhwc_q_per_axis_u8 => byte-identical
 * to the conv path. M-major (no transpose). Band over rows [m0,m1). */
static inline void rocket_out_mn_to_nhwc_q_per_axis_u8_band(
        const int32_t *src, unsigned char *dst, int M, int OC, int Ncol,
        const int32_t *eff_bias, const int *w_zp, const int32_t *Sx, int act,
        float in_scale, const float *w_scale, float out_scale, int out_zp,
        int m0, int m1)
{
    (void)M;
    const float inv = 1.0f / out_scale;
    /* The 1x1->matmul output [M][Ncol] requants to NHWC [M][OC] with OC contiguous on
     * BOTH sides (read sp[oc], write dst[m*OC+oc]), so the per-oc compute vectorizes 8
     * at a time with no scatter. All intermediates fit int32 (sp + eff_bias + beta*Sx[m]
     * <~1.3e8), so the int32 NEON arithmetic equals the scalar `long` math, and
     * vcvtnq_s32_f32 is lrintf's round-nearest-ties-even -> bit-identical to the scalar. */
    const int has_eb = eff_bias != NULL;
    const int has_beta = Sx != NULL && w_zp != NULL;   /* beta=128-w_zp only contributes here */
#if defined(__ARM_NEON)
    static int scalar_only = -1;   /* ROCKET_REQUANT_SCALAR=1 forces the scalar path (A/B + fallback) */
    if (scalar_only < 0) scalar_only = (getenv("ROCKET_REQUANT_SCALAR") != NULL);
#endif
    for (int m = m0; m < m1; m++) {
        const int32_t *sp = src + (size_t)m * Ncol;
        const long sx = Sx ? Sx[m] : 0;
        unsigned char *dp = dst + (size_t)m * OC;
        int oc = 0;
#if defined(__ARM_NEON)
      if (!scalar_only) {
        const int32x4_t v128 = vdupq_n_s32(128), vsx = vdupq_n_s32((int32_t)sx);
        const float32x4_t vis = vdupq_n_f32(in_scale), vinv = vdupq_n_f32(inv);
        const int32x4_t vzp = vdupq_n_s32(out_zp), vlo = vdupq_n_s32(0), vhi = vdupq_n_s32(255);
        for (; oc + 8 <= OC; oc += 8) {
            int32x4_t q[2];
            for (int h = 0; h < 2; h++) {
                int o = oc + 4 * h;
                int32x4_t acc = vld1q_s32(sp + o);
                if (has_eb)   acc = vaddq_s32(acc, vld1q_s32(eff_bias + o));
                if (has_beta) acc = vaddq_s32(acc, vmulq_s32(vsubq_s32(v128,
                                       vld1q_s32((const int32_t *)(w_zp + o))), vsx));
                float32x4_t v = vmulq_f32(vmulq_f32(vis, vld1q_f32(w_scale + o)),
                                          vcvtq_f32_s32(acc));
                v = rocket_apply_act_f32x4(v, act);
                int32x4_t qi = vaddq_s32(vcvtnq_s32_f32(vmulq_f32(v, vinv)), vzp);
                q[h] = vminq_s32(vmaxq_s32(qi, vlo), vhi);
            }
            uint8x8_t q8 = vqmovn_u16(vcombine_u16(vqmovun_s32(q[0]), vqmovun_s32(q[1])));
            vst1_u8(dp + oc, q8);
        }
      }
#endif
        for (; oc < OC; oc++) {           /* scalar tail / non-NEON */
            const long eb   = eff_bias ? eff_bias[oc] : 0;
            const long beta = Sx ? (128 - (w_zp ? w_zp[oc] : 128)) : 0;
            long acc = (long)sp[oc] + eb;
            if (Sx) acc += beta * sx;
            float v = (in_scale * w_scale[oc]) * (float)acc;
            v = rocket_apply_act(v, act);
            long q = (long)lrintf(v * inv) + out_zp;
            if (q < 0)   q = 0;
            if (q > 255) q = 255;
            dp[oc] = (unsigned char)q;
        }
    }
}
static inline void rocket_out_mn_to_nhwc_q_per_axis_u8(
        const int32_t *src, unsigned char *dst, int M, int OC, int Ncol,
        const int32_t *eff_bias, const int *w_zp, const int32_t *Sx, int act,
        float in_scale, const float *w_scale, float out_scale, int out_zp)
{
    rocket_out_mn_to_nhwc_q_per_axis_u8_band(src, dst, M, OC, Ncol, eff_bias, w_zp, Sx,
            act, in_scale, w_scale, out_scale, out_zp, 0, M);
}

#endif /* ROCKET_CONVERT_H */
