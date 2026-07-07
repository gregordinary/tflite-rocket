// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
#ifndef ROCKET_OPS_H
#define ROCKET_OPS_H

#include "rocket_convert.h"   /* rocket_qread, rocket_apply_act, ROCKET_ACT_* */
#include <math.h>             /* lrintf, INFINITY (also pulled via rocket_convert.h) */

/*
 * rocket_ops.h — pure NHWC HOST kernels for the elementwise / pooling / join ops a
 * real detector graph carries AROUND its convolutions (ADD residuals, AVERAGE/MAX
 * pooling, CONCATENATION joins). Like rocket_convert.h these carry NO TFLite (and no
 * hardware) dependency so they are unit-tested on x86 against an independent NHWC
 * oracle (tests/convert_test.cpp).
 *
 * WHY HOST (not NPU): these are memory-bound elementwise / reduction ops. Running a
 * thin host kernel inside the delegate keeps the delegated partition CONTIGUOUS
 * across them (conv -> add -> conv stays one partition instead of three) without
 * inventing new regcmd. The op is computed on the CPU exactly as TFLite's own kernel
 * would; the structural win is fewer/larger partitions today, with NCHW-resident
 * inter-op NPU buffers (which would also let conv skip the per-op transpose) as a
 * documented follow-up. An NPU primitive (e.g. a DPU eltwise-add) is only worth it
 * with a clear win AND a HW-validatable path — not the case for a v1.
 *
 * QUANT (int8 / uint8): mirror the conv boundary in rocket_convert.h. Operands are
 * dequantized to float, the op is computed in float, the result is requantized to
 * the output type — an fp32-host approximation of TFLite's fixed-point int8 kernels
 * (NOT bit-identical to them), validated against an oracle that runs the IDENTICAL
 * dequant->op->requant path. Input/output quant here is PER-TENSOR (single
 * scale/zero_point each), unlike the conv filter's per-axis params.
 *
 * Layout: TFLite-native NHWC throughout (no NCHW transpose — that is the conv path's
 * concern). input/output [N=1][H][W][C], elem(h,w,c) = (h*W + w)*C + c.
 */

/* ---- per-element quant helpers (per-tensor: one scale, one zero_point) ---- */

/* real = scale * (q - zero_point) */
static inline float rocket_dq1(int q, float scale, int zp)
{
    return scale * (float)(q - zp);
}

/* q = clamp(round(real / scale) + zero_point). Takes inv_scale = 1/scale so the
 * caller hoists the reciprocal; lrintf is round-to-nearest-ties-to-even, matched by
 * the oracle so the proof is exact. */
static inline int rocket_rq1(float v, float inv_scale, int zp, int qmin, int qmax)
{
    long q = (long)lrintf(v * inv_scale) + zp;
    if (q < qmin) q = qmin;
    if (q > qmax) q = qmax;
    return (int)q;
}

static inline void rocket_qrange(int is_unsigned, int *qmin, int *qmax)
{
    *qmin = is_unsigned ? 0   : -128;
    *qmax = is_unsigned ? 255 :  127;
}

static inline void rocket_qwrite(void *dst, size_t k, int v, int is_unsigned)
{
    if (is_unsigned) ((unsigned char *)dst)[k] = (unsigned char)v;
    else             ((signed char  *)dst)[k] = (signed char)v;
}

/* ==========================================================================
 * UNARY ACTIVATION — HARD_SWISH / LOGISTIC(sigmoid), standalone TFLite ops.
 *
 * Unlike Relu/Relu6 (conv-FUSED activations, applied in the conv epilogue),
 * HardSwish and Sigmoid are their OWN TFLite builtins (HARD_SWISH=117,
 * LOGISTIC=14) — a MobileNetV3 / MobileDet backbone emits them as separate
 * nodes between convs. Claiming them in the delegate keeps the partition
 * CONTIGUOUS (conv -> hardswish -> conv stays one partition) and un-spills the
 * nonlinearity from the CPU. Computed here as exact host kernels (float; quant =
 * dequant -> f -> requant, the fp32-host approximation the other aux ops use);
 * the delegate can alternatively route them onto the NPU DPU LUT
 * (rocket_activation_fp16) under its act_npu option. Pure elementwise (in/out
 * SAME shape), so layout-agnostic — NHWC or any rank, one flat pass.
 *
 * The kind codes are a TFLite-free mirror; the math matches the driver's
 * rocket_activation.c (and TFLite): HardSwish(x) = x*clip(x/6+0.5,0,1).
 * ========================================================================== */
enum {
    ROCKET_UNARY_HARDSWISH   = 0,   /* x * clip(x/6 + 0.5, 0, 1)  (TFLite HARD_SWISH) */
    ROCKET_UNARY_SIGMOID     = 1,   /* 1 / (1 + exp(-x))          (TFLite LOGISTIC)   */
    ROCKET_UNARY_HARDSIGMOID = 2,   /* clip(x/6 + 0.5, 0, 1)      (parity w/ the LUT) */
    ROCKET_UNARY_TANH        = 3,   /* tanh(x)                    (TFLite TANH)       */
    ROCKET_UNARY_ELU         = 4,   /* x>=0 ? x : exp(x)-1        (TFLite ELU, a=1)   */
    ROCKET_UNARY_LOG         = 5,   /* ln(x)                      (TFLite LOG, x>0)   */
    /* Exact-arithmetic siblings (the result is the same float ops TFLite's reference
     * kernel runs, so the float host path is byte-exact to CPU, not an approximation
     * like the transcendentals above). `param` carries LEAKY_RELU's slope; the rest
     * ignore it. */
    ROCKET_UNARY_RELU        = 6,   /* max(x, 0)                  (TFLite RELU)        */
    ROCKET_UNARY_RELU6       = 7,   /* clip(x, 0, 6)              (TFLite RELU6)       */
    ROCKET_UNARY_RELU_N1_1   = 8,   /* clip(x, -1, 1)             (TFLite RELU_N1_TO_1)*/
    ROCKET_UNARY_LEAKY_RELU  = 9,   /* x>=0 ? x : param*x         (TFLite LEAKY_RELU)  */
    ROCKET_UNARY_EXP         = 10,  /* exp(x)                     (TFLite EXP)         */
    ROCKET_UNARY_SQRT        = 11,  /* sqrt(x), x>=0              (TFLite SQRT)        */
    ROCKET_UNARY_RSQRT       = 12,  /* 1/sqrt(x), x>0             (TFLite RSQRT)       */
    ROCKET_UNARY_ABS         = 13,  /* |x|                        (TFLite ABS)         */
    ROCKET_UNARY_NEG         = 14,  /* -x                         (TFLite NEG)         */
    ROCKET_UNARY_SQUARE      = 15,  /* x*x                        (TFLite SQUARE)      */
    ROCKET_UNARY_FLOOR       = 16,  /* floor(x)                   (TFLite FLOOR)       */
};

static inline float rocket_unary_eval(int kind, float x, float param)
{
    switch (kind) {
    case ROCKET_UNARY_HARDSWISH: {
        float r = x / 6.0f + 0.5f;
        r = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
        return x * r;
    }
    case ROCKET_UNARY_SIGMOID:
        return 1.0f / (1.0f + expf(-x));
    case ROCKET_UNARY_HARDSIGMOID: {
        float r = x / 6.0f + 0.5f;
        return r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
    }
    case ROCKET_UNARY_TANH:
        return tanhf(x);
    case ROCKET_UNARY_ELU:
        /* TFLite ELU is fixed alpha=1: x>=0 ? x : exp(x)-1 (C0-continuous at 0). */
        return x >= 0.f ? x : (expf(x) - 1.0f);
    case ROCKET_UNARY_LOG:
        return logf(x);             /* TFLite LOG (x>0; matches std::log, incl. inf/nan) */
    case ROCKET_UNARY_RELU:
        return x > 0.f ? x : 0.f;
    case ROCKET_UNARY_RELU6:
        return x < 0.f ? 0.f : (x > 6.f ? 6.f : x);
    case ROCKET_UNARY_RELU_N1_1:
        return x < -1.f ? -1.f : (x > 1.f ? 1.f : x);
    case ROCKET_UNARY_LEAKY_RELU:
        return x >= 0.f ? x : param * x;
    case ROCKET_UNARY_EXP:
        return expf(x);
    case ROCKET_UNARY_SQRT:
        return sqrtf(x);            /* x>=0 (matches std::sqrt, incl. nan for x<0) */
    case ROCKET_UNARY_RSQRT:
        return 1.0f / sqrtf(x);     /* x>0 */
    case ROCKET_UNARY_ABS:
        return fabsf(x);
    case ROCKET_UNARY_NEG:
        return -x;
    case ROCKET_UNARY_SQUARE:
        return x * x;
    case ROCKET_UNARY_FLOOR:
        return floorf(x);
    default:
        return x;
    }
}

static inline void rocket_unary_f(const float *in, float *out, size_t n, int kind,
                                  float param)
{
    for (size_t i = 0; i < n; i++) out[i] = rocket_unary_eval(kind, in[i], param);
}

/* int8/uint8: dequant per-tensor -> f(x) in float -> requant to the output type. */
static inline void rocket_unary_q(const void *in, void *out, size_t n, int kind,
                                  int in_uns, int out_uns,
                                  float in_scale, int in_zp,
                                  float out_scale, int out_zp, float param)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (size_t i = 0; i < n; i++) {
        float v = rocket_dq1(rocket_qread(in, i, in_uns), in_scale, in_zp);
        v = rocket_unary_eval(kind, v, param);
        rocket_qwrite(out, i, rocket_rq1(v, inv, out_zp, qmin, qmax), out_uns);
    }
}

/* ==========================================================================
 * ADD — elementwise residual add, SAME shape (no broadcast in v1), fused act.
 * ========================================================================== */

static inline void rocket_add_f(const float *a, const float *b, float *o,
                                size_t n, int act)
{
    for (size_t i = 0; i < n; i++)
        o[i] = rocket_apply_act(a[i] + b[i], act);
}

/* int8/uint8: each input has its own per-tensor scale/zp; output its own. */
static inline void rocket_add_q(const void *a, const void *b, void *o, size_t n,
                                int a_uns, int b_uns, int o_uns,
                                float a_scale, int a_zp, float b_scale, int b_zp,
                                float o_scale, int o_zp, int act)
{
    int qmin, qmax; rocket_qrange(o_uns, &qmin, &qmax);
    const float inv = 1.0f / o_scale;
    for (size_t i = 0; i < n; i++) {
        float va = rocket_dq1(rocket_qread(a, i, a_uns), a_scale, a_zp);
        float vb = rocket_dq1(rocket_qread(b, i, b_uns), b_scale, b_zp);
        float v  = rocket_apply_act(va + vb, act);
        rocket_qwrite(o, i, rocket_rq1(v, inv, o_zp, qmin, qmax), o_uns);
    }
}

/* ==========================================================================
 * BINARY elementwise MAXIMUM / MINIMUM — two tensors, SAME shape (no broadcast
 * in v1). TFLite Maximum(55)/Minimum(57) carry NO fused-activation param, so the
 * op IS the whole node. Like ADD these are memory-bound, so the default is an
 * exact host kernel that keeps the partition contiguous (conv -> max -> conv); the
 * delegate can route them onto the NPU DPU EW ALU (rocket_ew_max/min_fp16) under
 * ew_npu. float | int8 | uint8 (per-tensor quant, dequant -> op -> requant).
 * ========================================================================== */
enum {
    ROCKET_BINOP_MAX = 0,   /* max(a, b)  (TFLite MAXIMUM) */
    ROCKET_BINOP_MIN = 1,   /* min(a, b)  (TFLite MINIMUM) */
};

static inline float rocket_binop_eval(int op, float a, float b)
{
    return op == ROCKET_BINOP_MIN ? (a < b ? a : b) : (a > b ? a : b);
}

static inline void rocket_binary_f(const float *a, const float *b, float *o,
                                   size_t n, int op)
{
    for (size_t i = 0; i < n; i++) o[i] = rocket_binop_eval(op, a[i], b[i]);
}

/* int8/uint8: each input has its own per-tensor scale/zp; output its own. */
static inline void rocket_binary_q(const void *a, const void *b, void *o, size_t n,
                                   int op, int a_uns, int b_uns, int o_uns,
                                   float a_scale, int a_zp, float b_scale, int b_zp,
                                   float o_scale, int o_zp)
{
    int qmin, qmax; rocket_qrange(o_uns, &qmin, &qmax);
    const float inv = 1.0f / o_scale;
    for (size_t i = 0; i < n; i++) {
        float va = rocket_dq1(rocket_qread(a, i, a_uns), a_scale, a_zp);
        float vb = rocket_dq1(rocket_qread(b, i, b_uns), b_scale, b_zp);
        float v  = rocket_binop_eval(op, va, vb);
        rocket_qwrite(o, i, rocket_rq1(v, inv, o_zp, qmin, qmax), o_uns);
    }
}

/* ==========================================================================
 * ARITHMETIC ELEMENTWISE — MUL / SUB / DIV (TFLite MUL=18 / SUB=41 / DIV=42).
 * Same shape (no broadcast). Unlike MAX/MIN these DO carry a fused activation
 * (TfLite{Mul,Sub,Div}Params), so they take an `act` like ADD. Memory-bound, so
 * the default is an exact host kernel that keeps the partition contiguous; the
 * driver's rocket_ew_{mul,sub,div}_fp16 exist but feed a conv-main EW job with no
 * fused act, so host is the default. SUB and DIV are NOT commutative — the operand
 * order (a op b) is in0 op in1. The op codes share AddP.op with MAX(0)/MIN(1), so
 * they start at 2 and are distinct from ADD's sentinel -1. float | int8 | uint8.
 * ========================================================================== */
enum {
    ROCKET_ARITH_MUL = 2,   /* a * b  (TFLite MUL) */
    ROCKET_ARITH_SUB = 3,   /* a - b  (TFLite SUB) */
    ROCKET_ARITH_DIV = 4,   /* a / b  (TFLite DIV) */
};

static inline float rocket_arith_eval(int op, float a, float b)
{
    switch (op) {
    case ROCKET_ARITH_MUL: return a * b;
    case ROCKET_ARITH_SUB: return a - b;
    case ROCKET_ARITH_DIV: return a / b;
    default:               return a + b;
    }
}

static inline void rocket_arith_f(const float *a, const float *b, float *o,
                                  size_t n, int op, int act)
{
    for (size_t i = 0; i < n; i++)
        o[i] = rocket_apply_act(rocket_arith_eval(op, a[i], b[i]), act);
}

static inline void rocket_arith_q(const void *a, const void *b, void *o, size_t n,
                                  int op, int a_uns, int b_uns, int o_uns,
                                  float a_scale, int a_zp, float b_scale, int b_zp,
                                  float o_scale, int o_zp, int act)
{
    int qmin, qmax; rocket_qrange(o_uns, &qmin, &qmax);
    const float inv = 1.0f / o_scale;
    for (size_t i = 0; i < n; i++) {
        float va = rocket_dq1(rocket_qread(a, i, a_uns), a_scale, a_zp);
        float vb = rocket_dq1(rocket_qread(b, i, b_uns), b_scale, b_zp);
        float v  = rocket_apply_act(rocket_arith_eval(op, va, vb), act);
        rocket_qwrite(o, i, rocket_rq1(v, inv, o_zp, qmin, qmax), o_uns);
    }
}

/* ==========================================================================
 * PRELU — parametric ReLU with a PER-CHANNEL negative slope (TFLite PRELU=54;
 * YOLO / segmentation). out = x>=0 ? x : alpha[c]*x. NHWC, so the channel is the
 * innermost index (c = i % C). alpha is a model constant (one slope per channel),
 * dequantized to float[C] by the delegate before the call. Like the other aux ops
 * the default is an exact host kernel; the delegate can route float PReLU onto the
 * NPU (rocket_prelu_fp16, no LUT) under act_npu. float | int8 | uint8 (per-tensor
 * x/out quant, dequant -> op -> requant).
 * ========================================================================== */
static inline void rocket_prelu_f(const float *in, float *out, size_t n, int C,
                                  const float *alpha)
{
    for (size_t i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x >= 0.f ? x : alpha[i % (size_t)C] * x;
    }
}

static inline void rocket_prelu_q(const void *in, void *out, size_t n, int C,
                                  const float *alpha, int in_uns, int out_uns,
                                  float in_scale, int in_zp, float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (size_t i = 0; i < n; i++) {
        float x = rocket_dq1(rocket_qread(in, i, in_uns), in_scale, in_zp);
        float v = x >= 0.f ? x : alpha[i % (size_t)C] * x;
        rocket_qwrite(out, i, rocket_rq1(v, inv, out_zp, qmin, qmax), out_uns);
    }
}

/* ==========================================================================
 * POOL — AVERAGE / MAX, NHWC, per-channel. The window is clamped to the image,
 * so (like TFLite) AVERAGE divides by the count of in-image cells, NOT KH*KW —
 * padding cells are excluded, never counted as zero. pad_top/pad_left are the
 * conv-style total_pad/2 (VALID => 0). is_avg selects mean vs max.
 * ========================================================================== */

static inline void rocket_pool_f(const float *in, float *out,
                                 int IH, int IW, int C, int OH, int OW,
                                 int KH, int KW, int sy, int sx,
                                 int pad_top, int pad_left, int is_avg, int act)
{
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < C; c++) {
                float v = is_avg ? 0.f : -INFINITY;
                int cnt = 0;
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh * sy + kh - pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < KW; kw++) {
                        int iw = ow * sx + kw - pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        float s = in[((size_t)ih * IW + iw) * C + c];
                        if (is_avg) v += s; else if (s > v) v = s;
                        cnt++;
                    }
                }
                // MAX of an all-OOB window keeps the max identity (-INFINITY, already in
                // v) rather than 0.f, which would beat genuine negative maxima.
                float r = is_avg ? (cnt ? v / (float)cnt : 0.f) : v;
                out[((size_t)oh * OW + ow) * C + c] = rocket_apply_act(r, act);
            }
}

static inline void rocket_pool_q(const void *in, void *out,
                                 int IH, int IW, int C, int OH, int OW,
                                 int KH, int KW, int sy, int sx,
                                 int pad_top, int pad_left, int is_avg, int act,
                                 int in_uns, int out_uns,
                                 float in_scale, int in_zp,
                                 float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < C; c++) {
                float v = is_avg ? 0.f : -INFINITY;
                int cnt = 0;
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh * sy + kh - pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < KW; kw++) {
                        int iw = ow * sx + kw - pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        float s = rocket_dq1(
                            rocket_qread(in, ((size_t)ih * IW + iw) * C + c, in_uns),
                            in_scale, in_zp);
                        if (is_avg) v += s; else if (s > v) v = s;
                        cnt++;
                    }
                }
                // MAX all-OOB keeps the max identity (-INFINITY in v); rocket_rq1 clamps
                // it to qmin, the correct lowest quantized value.
                float r = is_avg ? (cnt ? v / (float)cnt : 0.f) : v;
                r = rocket_apply_act(r, act);
                rocket_qwrite(out, ((size_t)oh * OW + ow) * C + c,
                              rocket_rq1(r, inv, out_zp, qmin, qmax), out_uns);
            }
}

/* ==========================================================================
 * SPATIAL REDUCE — MEAN / MAX / MIN over the [H,W] axes, per channel (TFLite
 * MEAN / REDUCE_MAX / REDUCE_MIN with axis=[1,2]; GlobalAvg/Max/MinPool). NHWC, so
 * the channel is the innermost index; out is [C] (the delegate's output tensor is
 * [1,C] or [1,1,1,C] per keep_dims, same C elements). Like the other aux ops the
 * default is an exact host kernel (fp32 accumulate for the mean); the delegate can
 * route float reduces onto the NPU PPU (rocket_global_{avg,max,min}pool_fp16) under
 * pool_npu. float | int8 | uint8 (per-tensor quant, dequant -> reduce -> requant).
 * ========================================================================== */
enum {
    ROCKET_REDUCE_MEAN = 0,   /* sum/(H*W)  (TFLite MEAN axis [1,2]) */
    ROCKET_REDUCE_MAX  = 1,   /* max        (TFLite REDUCE_MAX)      */
    ROCKET_REDUCE_MIN  = 2,   /* min        (TFLite REDUCE_MIN)      */
};

static inline void rocket_reduce_spatial_f(const float *in, float *out,
                                           int H, int W, int C, int op)
{
    const size_t HW = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        float acc = op == ROCKET_REDUCE_MEAN ? 0.f
                  : op == ROCKET_REDUCE_MIN  ? INFINITY : -INFINITY;
        for (size_t p = 0; p < HW; p++) {
            float v = in[p * (size_t)C + c];
            if      (op == ROCKET_REDUCE_MEAN) acc += v;
            else if (op == ROCKET_REDUCE_MIN)  { if (v < acc) acc = v; }
            else                               { if (v > acc) acc = v; }
        }
        out[c] = op == ROCKET_REDUCE_MEAN ? (HW ? acc / (float)HW : 0.f) : acc;
    }
}

static inline void rocket_reduce_spatial_q(const void *in, void *out,
                                           int H, int W, int C, int op,
                                           int in_uns, int out_uns,
                                           float in_scale, int in_zp,
                                           float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    const size_t HW = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        float acc = op == ROCKET_REDUCE_MEAN ? 0.f
                  : op == ROCKET_REDUCE_MIN  ? INFINITY : -INFINITY;
        for (size_t p = 0; p < HW; p++) {
            float v = rocket_dq1(rocket_qread(in, p * (size_t)C + c, in_uns), in_scale, in_zp);
            if      (op == ROCKET_REDUCE_MEAN) acc += v;
            else if (op == ROCKET_REDUCE_MIN)  { if (v < acc) acc = v; }
            else                               { if (v > acc) acc = v; }
        }
        float r = op == ROCKET_REDUCE_MEAN ? (HW ? acc / (float)HW : 0.f) : acc;
        rocket_qwrite(out, c, rocket_rq1(r, inv, out_zp, qmin, qmax), out_uns);
    }
}

/* ==========================================================================
 * CONCATENATION — join N inputs along `axis`. Generalized for any rank/axis:
 *   outer    = prod(dims  < axis)        (shared by inputs + output)
 *   inner    = prod(dims  > axis)        (shared)
 *   out_axis = output extent along axis;  in_axis = this input's extent
 *   axis_off = this input's start offset along axis in the output
 * The delegate loops the inputs accumulating axis_off; these copy ONE input's
 * slab into place. Float = copy(+act); quant = requantize each input to the
 * output scale/zp (inputs may carry different scales).
 * ========================================================================== */

static inline void rocket_concat_in_f(const float *in, float *out,
                                      int outer, int in_axis, int out_axis,
                                      int inner, int axis_off, int act)
{
    for (int o = 0; o < outer; o++)
        for (int j = 0; j < in_axis; j++) {
            const float *src = in  + ((size_t)o * in_axis  + j)            * inner;
            float       *dst = out + ((size_t)o * out_axis + axis_off + j) * inner;
            for (int k = 0; k < inner; k++) dst[k] = rocket_apply_act(src[k], act);
        }
}

static inline void rocket_concat_in_q(const void *in, void *out,
                                      int outer, int in_axis, int out_axis,
                                      int inner, int axis_off, int act,
                                      int in_uns, int out_uns,
                                      float in_scale, int in_zp,
                                      float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (int o = 0; o < outer; o++)
        for (int j = 0; j < in_axis; j++) {
            size_t sbase = ((size_t)o * in_axis  + j)            * inner;
            size_t dbase = ((size_t)o * out_axis + axis_off + j) * inner;
            for (int k = 0; k < inner; k++) {
                float v = rocket_dq1(rocket_qread(in, sbase + k, in_uns), in_scale, in_zp);
                v = rocket_apply_act(v, act);
                rocket_qwrite(out, dbase + k, rocket_rq1(v, inv, out_zp, qmin, qmax), out_uns);
            }
        }
}

/* ==========================================================================
 * RESIZE / UPSAMPLE — RESIZE_NEAREST_NEIGHBOR / RESIZE_BILINEAR, NHWC, per channel
 * (FPN / decoder neck). The coordinate math mirrors TFLite's reference kernels
 * (align_corners / half_pixel_centers) so a PASS is bit-exact-vs-CPU; the delegate
 * can route an integer-factor half-pixel resize onto the NPU (rocket_upsample_*)
 * under a narrow gate. Quant is a value pass-through (resize never rescales) — int8/
 * uint8 just gather the SAME stored bytes (nearest) or lerp the dequantized values
 * (bilinear, requantized to the unchanged output scale/zp).
 * ========================================================================== */

static inline float rocket_clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* TFLite reference GetNearestNeighbor (resize_nearest_neighbor.h). */
static inline int rocket_nn_src(int o, int in_size, int out_size,
                                int align_corners, int half_pixel)
{
    const float scale = (align_corners && out_size > 1)
                        ? (float)(in_size - 1) / (float)(out_size - 1)
                        : (float)in_size / (float)out_size;
    const float off = half_pixel ? 0.5f : 0.0f;
    int v = align_corners ? (int)lrintf(((float)o + off) * scale - off)
                          : (int)floorf(((float)o + off) * scale);
    if (v > in_size - 1) v = in_size - 1;
    if (half_pixel && v < 0) v = 0;
    return v;
}

/* TFLite reference bilinear source coordinate (float, before clamp/floor). */
static inline float rocket_bilin_src(int o, int in_size, int out_size,
                                     int align_corners, int half_pixel)
{
    const float scale = (align_corners && out_size > 1)
                        ? (float)(in_size - 1) / (float)(out_size - 1)
                        : (float)in_size / (float)out_size;
    float v = half_pixel ? ((float)o + 0.5f) * scale - 0.5f : (float)o * scale;
    return v;
}

static inline void rocket_resize_nearest_f(const float *in, float *out,
                                           int IH, int IW, int C, int OH, int OW,
                                           int align_corners, int half_pixel)
{
    for (int oh = 0; oh < OH; oh++) {
        int ih = rocket_nn_src(oh, IH, OH, align_corners, half_pixel);
        for (int ow = 0; ow < OW; ow++) {
            int iw = rocket_nn_src(ow, IW, OW, align_corners, half_pixel);
            const float *src = in  + ((size_t)ih * IW + iw) * C;
            float       *dst = out + ((size_t)oh * OW + ow) * C;
            for (int c = 0; c < C; c++) dst[c] = src[c];
        }
    }
}

static inline void rocket_resize_bilinear_f(const float *in, float *out,
                                            int IH, int IW, int C, int OH, int OW,
                                            int align_corners, int half_pixel)
{
    for (int oh = 0; oh < OH; oh++) {
        float fy = rocket_bilin_src(oh, IH, OH, align_corners, half_pixel);
        fy = rocket_clampf(fy, 0.f, (float)(IH - 1));
        int y0 = (int)floorf(fy); int y1 = y0 < IH - 1 ? y0 + 1 : y0; float wy = fy - (float)y0;
        for (int ow = 0; ow < OW; ow++) {
            float fx = rocket_bilin_src(ow, IW, OW, align_corners, half_pixel);
            fx = rocket_clampf(fx, 0.f, (float)(IW - 1));
            int x0 = (int)floorf(fx); int x1 = x0 < IW - 1 ? x0 + 1 : x0; float wx = fx - (float)x0;
            const float *r00 = in + ((size_t)y0 * IW + x0) * C, *r01 = in + ((size_t)y0 * IW + x1) * C;
            const float *r10 = in + ((size_t)y1 * IW + x0) * C, *r11 = in + ((size_t)y1 * IW + x1) * C;
            float *dst = out + ((size_t)oh * OW + ow) * C;
            for (int c = 0; c < C; c++) {
                float top = r00[c] + (r01[c] - r00[c]) * wx;
                float bot = r10[c] + (r11[c] - r10[c]) * wx;
                dst[c] = top + (bot - top) * wy;
            }
        }
    }
}

/* Quant nearest = exact byte gather (no rescale). */
static inline void rocket_resize_nearest_q(const void *in, void *out,
                                           int IH, int IW, int C, int OH, int OW,
                                           int align_corners, int half_pixel, int elem_size)
{
    for (int oh = 0; oh < OH; oh++) {
        int ih = rocket_nn_src(oh, IH, OH, align_corners, half_pixel);
        for (int ow = 0; ow < OW; ow++) {
            int iw = rocket_nn_src(ow, IW, OW, align_corners, half_pixel);
            memcpy((char *)out + ((size_t)oh * OW + ow) * C * elem_size,
                   (const char *)in + ((size_t)ih * IW + iw) * C * elem_size,
                   (size_t)C * elem_size);
        }
    }
}

/* Quant bilinear: dequant -> lerp -> requant to the SAME (in==out) scale/zp. */
static inline void rocket_resize_bilinear_q(const void *in, void *out,
                                            int IH, int IW, int C, int OH, int OW,
                                            int align_corners, int half_pixel, int uns,
                                            float scale, int zp)
{
    int qmin, qmax; rocket_qrange(uns, &qmin, &qmax);
    const float inv = 1.0f / scale;
    for (int oh = 0; oh < OH; oh++) {
        float fy = rocket_bilin_src(oh, IH, OH, align_corners, half_pixel);
        fy = rocket_clampf(fy, 0.f, (float)(IH - 1));
        int y0 = (int)floorf(fy); int y1 = y0 < IH - 1 ? y0 + 1 : y0; float wy = fy - (float)y0;
        for (int ow = 0; ow < OW; ow++) {
            float fx = rocket_bilin_src(ow, IW, OW, align_corners, half_pixel);
            fx = rocket_clampf(fx, 0.f, (float)(IW - 1));
            int x0 = (int)floorf(fx); int x1 = x0 < IW - 1 ? x0 + 1 : x0; float wx = fx - (float)x0;
            for (int c = 0; c < C; c++) {
                float v00 = rocket_dq1(rocket_qread(in, ((size_t)y0 * IW + x0) * C + c, uns), scale, zp);
                float v01 = rocket_dq1(rocket_qread(in, ((size_t)y0 * IW + x1) * C + c, uns), scale, zp);
                float v10 = rocket_dq1(rocket_qread(in, ((size_t)y1 * IW + x0) * C + c, uns), scale, zp);
                float v11 = rocket_dq1(rocket_qread(in, ((size_t)y1 * IW + x1) * C + c, uns), scale, zp);
                float top = v00 + (v01 - v00) * wx, bot = v10 + (v11 - v10) * wx;
                float v = top + (bot - top) * wy;
                rocket_qwrite(out, ((size_t)oh * OW + ow) * C + c,
                              rocket_rq1(v, inv, zp, qmin, qmax), uns);
            }
        }
    }
}

/* ==========================================================================
 * L2_NORMALIZATION — normalize each spatial position over its CHANNEL vector
 * (TFLite L2_NORMALIZATION, the last/depth axis): out[c] = x[c]/sqrt(sum_c x[c]^2).
 * NHWC, so each row of C contiguous channels is one normalized vector (M = N*H*W
 * rows). Default = exact host kernel (fp32 accumulate); the delegate can route float
 * L2Norm onto the NPU (rocket_l2norm_fp16, [M][C]) under norm_npu. A zero vector maps
 * to zeros (guarded). float | int8 | uint8 (per-tensor quant; the int8 output scale is
 * the standard 1/128 [-1,1] range — dequant -> normalize -> requant).
 * ========================================================================== */
static inline void rocket_l2norm_f(const float *in, float *out, int M, int C)
{
    for (int m = 0; m < M; m++) {
        const float *x = in + (size_t)m * C;
        float *y = out + (size_t)m * C;
        float ss = 0.f;
        for (int c = 0; c < C; c++) ss += x[c] * x[c];
        float inv = ss > 0.f ? 1.0f / sqrtf(ss) : 0.f;
        for (int c = 0; c < C; c++) y[c] = x[c] * inv;
    }
}

static inline void rocket_l2norm_q(const void *in, void *out, int M, int C,
                                   int in_uns, int out_uns, float in_scale, int in_zp,
                                   float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv_os = 1.0f / out_scale;
    for (int m = 0; m < M; m++) {
        float ss = 0.f;
        for (int c = 0; c < C; c++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * C + c, in_uns), in_scale, in_zp);
            ss += v * v;
        }
        float inv = ss > 0.f ? 1.0f / sqrtf(ss) : 0.f;
        for (int c = 0; c < C; c++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * C + c, in_uns), in_scale, in_zp) * inv;
            rocket_qwrite(out, (size_t)m * C + c, rocket_rq1(v, inv_os, out_zp, qmin, qmax), out_uns);
        }
    }
}

/* ==========================================================================
 * TRANSPOSE_CONV — learned upsample / "deconvolution" (segmentation / decoder /
 * super-res). NHWC input [IH,IW,IC], TFLite weight layout [OC,KH,KW,IC], output
 * [OH,OW,OC]. The scatter convention matches TFLite's reference exactly:
 *     out[oh,ow,oc] += in[ih,iw,ic] * W[oc,kh,kw,ic],
 *     oh = ih*sy + kh - pad_h,  ow = iw*sx + kw - pad_w   (in-range only)
 * so a PASS is bit-exact-vs-CPU (float). The default is this exact host kernel
 * (claims the op so a decoder's conv->tconv->conv stays one partition); the NPU
 * route (rocket_conv_transpose2d_fp16, same scatter convention) is a documented
 * follow-on (weight repack [OC,KH,KW,IC]->[IC,OC,KH,KW] + the pad_h/opad split).
 * float (v1); int8/uint8 (per-axis weight scales) deferred -> those stay on CPU.
 * ========================================================================== */
static inline void rocket_transpose_conv_f(const float *in, const float *W,
                                           const float *bias, float *out,
                                           int IH, int IW, int IC, int OC,
                                           int KH, int KW, int sy, int sx,
                                           int pad_h, int pad_w, int OH, int OW, int act)
{
    const size_t outn = (size_t)OH * OW * OC;
    for (size_t o = 0; o < outn; o++) out[o] = 0.f;
    for (int ih = 0; ih < IH; ih++)
        for (int iw = 0; iw < IW; iw++)
            for (int kh = 0; kh < KH; kh++) {
                int oh = ih * sy + kh - pad_h;
                if (oh < 0 || oh >= OH) continue;
                for (int kw = 0; kw < KW; kw++) {
                    int ow = iw * sx + kw - pad_w;
                    if (ow < 0 || ow >= OW) continue;
                    const float *vin = in + ((size_t)ih * IW + iw) * IC;
                    float *dst = out + ((size_t)oh * OW + ow) * OC;
                    for (int oc = 0; oc < OC; oc++) {
                        // W[oc,kh,kw,ic]
                        const float *wc = W + (((size_t)oc * KH + kh) * KW + kw) * IC;
                        float acc = 0.f;
                        for (int ic = 0; ic < IC; ic++) acc += vin[ic] * wc[ic];
                        dst[oc] += acc;
                    }
                }
            }
    if (bias) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) {
        float *dst = out + ((size_t)oh * OW + ow) * OC;
        for (int oc = 0; oc < OC; oc++) dst[oc] += bias[oc];
    }
    for (size_t o = 0; o < outn; o++) out[o] = rocket_apply_act(out[o], act);
}

/* ==========================================================================
 * LOG_SOFTMAX — out = x - logsumexp(x) over the last axis (TFLite LOG_SOFTMAX).
 * Numerically stable (subtract the row max first). NHWC etc.: M = product of all
 * dims except last, N = last. Default = exact host kernel; the NPU route
 * (rocket_logsoftmax_fp16, [M][N]) is a documented follow-on. float | int8 | uint8;
 * the int8/uint8 output uses TFLite's fixed [0,1)->log range (out scale 16/256, zp
 * 127 for int8 — dequant -> logsoftmax -> requant, the fp32-host approximation).
 * ========================================================================== */
static inline void rocket_logsoftmax_f(const float *in, float *out, int M, int N)
{
    for (int m = 0; m < M; m++) {
        const float *x = in + (size_t)m * N; float *y = out + (size_t)m * N;
        float mx = -INFINITY; for (int n = 0; n < N; n++) if (x[n] > mx) mx = x[n];
        float s = 0.f; for (int n = 0; n < N; n++) s += expf(x[n] - mx);
        float lse = mx + logf(s);
        for (int n = 0; n < N; n++) y[n] = x[n] - lse;
    }
}

static inline void rocket_logsoftmax_q(const void *in, void *out, int M, int N,
                                       int in_uns, int out_uns, float in_scale, int in_zp,
                                       float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (int m = 0; m < M; m++) {
        float mx = -INFINITY;
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            if (v > mx) mx = v;
        }
        float s = 0.f;
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            s += expf(v - mx);
        }
        float lse = mx + logf(s);
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp) - lse;
            rocket_qwrite(out, (size_t)m * N + n, rocket_rq1(v, inv, out_zp, qmin, qmax), out_uns);
        }
    }
}

/* ==========================================================================
 * SOFTMAX — out_i = exp(beta*x_i) / sum_j exp(beta*x_j) over the last axis
 * (TFLite SOFTMAX). Numerically stable (subtract the row max first). M = product
 * of all dims except last, N = last. Default = exact host kernel (the same float
 * ops the TFLite reference runs); the NPU route (rocket_softmax_fp16, [M][N]) is a
 * documented follow-on. float | int8 | uint8 (per-tensor quant; the int8/uint8 path
 * is the fp32-host dequant->softmax->requant approximation, like LOG_SOFTMAX).
 * ========================================================================== */
static inline void rocket_softmax_f(const float *in, float *out, int M, int N, float beta)
{
    for (int m = 0; m < M; m++) {
        const float *x = in + (size_t)m * N; float *y = out + (size_t)m * N;
        float mx = -INFINITY; for (int n = 0; n < N; n++) if (x[n] > mx) mx = x[n];
        float s = 0.f; for (int n = 0; n < N; n++) { float e = expf(beta * (x[n] - mx)); y[n] = e; s += e; }
        float inv = 1.0f / s; for (int n = 0; n < N; n++) y[n] *= inv;
    }
}

static inline void rocket_softmax_q(const void *in, void *out, int M, int N, float beta,
                                    int in_uns, int out_uns, float in_scale, int in_zp,
                                    float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (int m = 0; m < M; m++) {
        float mx = -INFINITY;
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            if (v > mx) mx = v;
        }
        float s = 0.f;
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            s += expf(beta * (v - mx));
        }
        float den = 1.0f / s;
        for (int n = 0; n < N; n++) {
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            float p = expf(beta * (v - mx)) * den;
            rocket_qwrite(out, (size_t)m * N + n, rocket_rq1(p, inv, out_zp, qmin, qmax), out_uns);
        }
    }
}

/* ==========================================================================
 * CUMSUM — prefix sum along the LAST axis (TFLite CUMSUM, axis = last). M = product
 * of all dims except last, N = last. exclusive/reverse select the four variants.
 * Default = exact host kernel (fp32 accumulate); the NPU route (rocket_cumsum_fp16,
 * cumsum-as-triangular-matmul) is a documented follow-on. float | int8 | uint8
 * (per-tensor quant, dequant -> prefix-sum -> requant).
 * ========================================================================== */
static inline void rocket_cumsum_f(const float *in, float *out, int M, int N,
                                   int exclusive, int reverse)
{
    for (int m = 0; m < M; m++) {
        const float *x = in + (size_t)m * N; float *y = out + (size_t)m * N;
        float acc = 0.f;
        for (int i = 0; i < N; i++) {
            int n = reverse ? N - 1 - i : i;
            if (exclusive) { y[n] = acc; acc += x[n]; }
            else           { acc += x[n]; y[n] = acc; }
        }
    }
}

static inline void rocket_cumsum_q(const void *in, void *out, int M, int N,
                                   int exclusive, int reverse, int in_uns, int out_uns,
                                   float in_scale, int in_zp, float out_scale, int out_zp)
{
    int qmin, qmax; rocket_qrange(out_uns, &qmin, &qmax);
    const float inv = 1.0f / out_scale;
    for (int m = 0; m < M; m++) {
        float acc = 0.f;
        for (int i = 0; i < N; i++) {
            int n = reverse ? N - 1 - i : i;
            float v = rocket_dq1(rocket_qread(in, (size_t)m * N + n, in_uns), in_scale, in_zp);
            float w;
            if (exclusive) { w = acc; acc += v; } else { acc += v; w = acc; }
            rocket_qwrite(out, (size_t)m * N + n, rocket_rq1(w, inv, out_zp, qmin, qmax), out_uns);
        }
    }
}

/* ==========================================================================
 * RESHAPE — shape-only passthrough: a pure byte copy. The quant scale/zero_point
 * are preserved (TFLite never rescales a reshape), so there is no arithmetic to
 * approximate — float and int8/uint8 are the identical copy. elem_size is bytes
 * per element (1 for int8/uint8, 4 for float32). The delegate skips the copy when
 * TFLite has aliased the input/output buffers.
 * ========================================================================== */
static inline void rocket_reshape_copy(void *dst, const void *src, size_t n, int elem_size)
{
    memcpy(dst, src, n * (size_t)elem_size);
}

/* ==========================================================================
 * LAYOUT OPS — TRANSPOSE / PAD / SLICE / SPLIT.
 *
 * These reorder, crop, or extend a tensor; none touch the values, so (like
 * RESHAPE) they are pure byte moves: float and int8/uint8 are the IDENTICAL
 * copy, and the quant scale/zero_point pass through unchanged (TFLite never
 * rescales a layout op). elem_size is bytes per element (1 int8/uint8, 4
 * float32/int32). The NPU cannot help — it has no on-chip layout-conversion
 * engine (the host packB scatter is irreducible, and the PPU is a pooling
 * engine, not a reshape engine) — so the value is keeping a real graph's
 * `conv -> layout-op -> conv` in ONE delegated partition instead of splitting
 * around a CPU node. Generic over rank (<= ROCKET_MAX_RANK), NHWC or any layout.
 * ========================================================================== */
#define ROCKET_MAX_RANK 8

/* Row-major ELEMENT strides for dims[0..rank) into stride[]. */
static inline void rocket_row_strides(const int *dims, int rank, long *stride)
{
    long s = 1;
    for (int d = rank - 1; d >= 0; d--) { stride[d] = s; s *= dims[d]; }
}

/* TRANSPOSE: out has dims in_dims[perm[d]]; out[oc] = in[ic] where ic[perm[d]]
 * = oc[d]. Output-driven gather over the output odometer. */
static inline void rocket_transpose_bytes(void *dst, const void *src,
                                          int rank, const int *in_dims,
                                          const int *perm, int elem_size)
{
    int  out_dims[ROCKET_MAX_RANK];
    long in_stride[ROCKET_MAX_RANK];
    long total = 1;
    for (int d = 0; d < rank; d++) { out_dims[d] = in_dims[perm[d]]; total *= out_dims[d]; }
    rocket_row_strides(in_dims, rank, in_stride);
    int idx[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) idx[d] = 0;
    const char *s = (const char *)src; char *o = (char *)dst;
    for (long k = 0; k < total; k++) {
        long in_off = 0;
        for (int d = 0; d < rank; d++) in_off += (long)idx[d] * in_stride[perm[d]];
        memcpy(o + k * elem_size, s + in_off * (long)elem_size, (size_t)elem_size);
        for (int d = rank - 1; d >= 0; d--) { if (++idx[d] < out_dims[d]) break; idx[d] = 0; }
    }
}

/* SLICE: out[oc] = in[oc + begin]. out_dims is the slice extent per axis.
 * SPLIT is N slices along one axis (begin[axis] = the per-output offset, 0
 * elsewhere; out_dims = the output's dims) — the delegate reuses this kernel. */
static inline void rocket_slice_bytes(void *dst, const void *src,
                                      int rank, const int *in_dims,
                                      const int *begin, const int *out_dims,
                                      int elem_size)
{
    long in_stride[ROCKET_MAX_RANK];
    rocket_row_strides(in_dims, rank, in_stride);
    long total = 1; for (int d = 0; d < rank; d++) total *= out_dims[d];
    int idx[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) idx[d] = 0;
    const char *s = (const char *)src; char *o = (char *)dst;
    for (long k = 0; k < total; k++) {
        long in_off = 0;
        for (int d = 0; d < rank; d++) in_off += (long)(idx[d] + begin[d]) * in_stride[d];
        memcpy(o + k * elem_size, s + in_off * (long)elem_size, (size_t)elem_size);
        for (int d = rank - 1; d >= 0; d--) { if (++idx[d] < out_dims[d]) break; idx[d] = 0; }
    }
}

/* STRIDED_SLICE with a signed per-axis stride: out[oc] = in[begin[d] + oc[d]*stride[d]].
 * out_dims is the per-axis loop-step count the delegate resolved from TFLite's
 * Start/Stop rules; begin[] is the FIRST gathered index (the HIGH index when stride[d]<0,
 * walking down) and stride[d] is signed (!=0). (ellipsis / new-axis masks stay on CPU —
 * those remap the begin/end/stride axes; see the delegate gate.) */
static inline void rocket_strided_slice_bytes(void *dst, const void *src,
                                              int rank, const int *in_dims,
                                              const int *begin, const int *stride,
                                              const int *out_dims, int elem_size)
{
    long in_stride[ROCKET_MAX_RANK];
    rocket_row_strides(in_dims, rank, in_stride);
    long total = 1; for (int d = 0; d < rank; d++) total *= out_dims[d];
    int idx[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) idx[d] = 0;
    const char *s = (const char *)src; char *o = (char *)dst;
    for (long k = 0; k < total; k++) {
        long in_off = 0;
        for (int d = 0; d < rank; d++)
            in_off += (long)(begin[d] + idx[d] * stride[d]) * in_stride[d];
        memcpy(o + k * elem_size, s + in_off * (long)elem_size, (size_t)elem_size);
        for (int d = rank - 1; d >= 0; d--) { if (++idx[d] < out_dims[d]) break; idx[d] = 0; }
    }
}

/* Broadcast-copy `in` (shape in_dims[in_rank]) into the out-shaped buffer `dst`
 * (out_dims[out_rank]) with NumPy / TFLite right-aligned semantics: aligning the
 * shapes at their trailing axis, each in dim must be 1 or equal to the out dim; a
 * size-1 (or absent leading) in dim contributes stride 0, so it repeats. Byte-wise
 * (elem_size 1/2/4) so it serves float and int8/uint8 alike. Lowers a broadcasting
 * elementwise op to the SAME-shape host kernels: materialize each operand to the
 * output shape, then run rocket_add/arith/binary unchanged (bit-exact). */
static inline void rocket_broadcast_copy(const void *in, void *dst,
                                         const int *in_dims, int in_rank,
                                         const int *out_dims, int out_rank,
                                         int elem_size)
{
    long in_row[ROCKET_MAX_RANK];
    rocket_row_strides(in_dims, in_rank, in_row);
    long bstride[ROCKET_MAX_RANK];               /* in's stride per OUTPUT dim (0 = broadcast) */
    int off = out_rank - in_rank;                /* right-align: out dim d -> in dim d-off */
    for (int d = 0; d < out_rank; d++) {
        int id = d - off;
        bstride[d] = (id < 0 || in_dims[id] == 1) ? 0 : in_row[id];
    }
    long total = 1; for (int d = 0; d < out_rank; d++) total *= out_dims[d];
    int idx[ROCKET_MAX_RANK]; for (int d = 0; d < out_rank; d++) idx[d] = 0;
    const char *s = (const char *)in; char *o = (char *)dst;
    for (long k = 0; k < total; k++) {
        long in_off = 0;
        for (int d = 0; d < out_rank; d++) in_off += (long)idx[d] * bstride[d];
        memcpy(o + k * elem_size, s + in_off * (long)elem_size, (size_t)elem_size);
        for (int d = out_rank - 1; d >= 0; d--) { if (++idx[d] < out_dims[d]) break; idx[d] = 0; }
    }
}

/* PAD: out_dims[d] = in_dims[d] + pad_before[d] + pad_after[d]; the border is
 * filled with pad_elem (elem_size bytes — 0.0f for float PAD, the constant for
 * PADV2, the zero_point byte for quant) and the input scattered into the
 * interior. Input-driven (writes the input block + a one-pass border fill). */
static inline void rocket_pad_bytes(void *dst, const void *src,
                                    int rank, const int *in_dims,
                                    const int *pad_before, const int *out_dims,
                                    int elem_size, const void *pad_elem)
{
    long out_stride[ROCKET_MAX_RANK];
    rocket_row_strides(out_dims, rank, out_stride);
    long out_total = 1; for (int d = 0; d < rank; d++) out_total *= out_dims[d];
    char *o = (char *)dst;
    for (long k = 0; k < out_total; k++) memcpy(o + k * elem_size, pad_elem, (size_t)elem_size);
    long in_total = 1; for (int d = 0; d < rank; d++) in_total *= in_dims[d];
    int idx[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) idx[d] = 0;
    const char *s = (const char *)src;
    for (long k = 0; k < in_total; k++) {
        long out_off = 0;
        for (int d = 0; d < rank; d++) out_off += (long)(idx[d] + pad_before[d]) * out_stride[d];
        memcpy(o + out_off * (long)elem_size, s + k * elem_size, (size_t)elem_size);
        for (int d = rank - 1; d >= 0; d--) { if (++idx[d] < in_dims[d]) break; idx[d] = 0; }
    }
}

#endif /* ROCKET_OPS_H */
