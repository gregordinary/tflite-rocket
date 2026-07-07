// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
/*
 * convert_test.cpp — off-hardware proof of the delegate's NHWC<->NCHW + SAME/VALID
 * padding + bias/activation glue (rocket_convert.h).
 *
 * The delegate's only genuinely new logic on top of the HW-validated
 * rocket_conv2d_fp16 is the layout/padding plumbing between TFLite's NHWC float
 * tensors and the driver's NCHW fp16 conv. This test runs that EXACT plumbing —
 * transpose+materialize-pad in, rocket_conv2d_fp16, transpose+bias+act out — and
 * compares to an independent NHWC convolution oracle. No TFLite, no NPU: it links
 * the driver with fd=-1, so rocket_conv2d_fp16 falls back to its CPU oracle and we
 * are validating the GLUE, not the device (the device path is the driver's own
 * HW-validated gate). Same discipline as the driver's cube-self-check.
 *
 * Data is kept to {-1,0,1} with small kernels so every partial sum is exact in
 * fp16 (|sum| <= 2048): a PASS is then a clean statement about layout/pad, with no
 * fp16 rounding to hide a transpose bug. Shapes cover 1x1, 3x3 SAME (incl. an even
 * input under stride 2 => ASYMMETRIC SAME pad), VALID, dilation, a 5x5, an
 * asymmetric 1x5, and the IC=3 RGB stem — plus bias and each fused activation.
 *
 * Build (x86, no cmake): see the run recipe in the tflite-rocket README — gcc the C
 * driver sources, g++ this, link with -lm.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
}
#include "rocket_convert.h"
#include "rocket_ops.h"

struct Shape {
    int IC, IH, IW, OC, KH, KW, sy, sx, dy, dx;
    int same;       /* 1 = SAME padding, 0 = VALID */
    int use_bias;
    int act;        /* ROCKET_ACT_* */
    const char *name;
};

/* Independent NHWC convolution oracle. Mirrors the glue's NUMERIC path (fp32
 * accumulate, narrow to fp16 like the NPU output, then +bias in float, then the
 * fused activation) so a layout bug — not fp16 rounding — is what a mismatch
 * means. pad_top/pad_left are TFLite's pad_before = total_pad/2. */
static void nhwc_oracle(const Shape &s, const float *in, const float *flt,
                        const float *bias, int OH, int OW,
                        int pad_top, int pad_left, float *out)
{
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < s.OC; oc++) {
                float acc = 0.f;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    if (ih < 0 || ih >= s.IH) continue;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        if (iw < 0 || iw >= s.IW) continue;
                        for (int ic = 0; ic < s.IC; ic++)
                            acc += in[((size_t)ih * s.IW + iw) * s.IC + ic] *
                                   flt[(((size_t)oc * s.KH + kh) * s.KW + kw) * s.IC + ic];
                    }
                }
                float v = (float)(_Float16)acc;          /* narrow like the NPU fp16 out */
                if (bias) v += bias[oc];
                out[((size_t)oh * OW + ow) * s.OC + oc] = rocket_apply_act(v, s.act);
            }
}

static int run_shape(int fd, const Shape &s)
{
    const int eff_kh = rocket_eff_k(s.KH, s.dy), eff_kw = rocket_eff_k(s.KW, s.dx);
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);

    printf("%-22s IC=%d %dx%d -> OC=%d K=%dx%d s=%dx%d d=%dx%d %s bias=%d act=%d (OH=%d OW=%d)\n",
           s.name, s.IC, s.IH, s.IW, s.OC, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.use_bias, s.act, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }

    /* TFLite NHWC buffers, small integer values (exact in fp16). */
    std::vector<float> in((size_t)s.IH * s.IW * s.IC);
    std::vector<float> flt((size_t)s.OC * s.KH * s.KW * s.IC);
    std::vector<float> bias(s.OC);
    for (size_t i = 0; i < in.size();  i++) in[i]  = (float)((int)(i % 3) - 1);
    for (size_t i = 0; i < flt.size(); i++) flt[i] = (float)((int)(i % 3) - 1);
    for (int i = 0; i < s.OC; i++) bias[i] = s.use_bias ? (float)((i % 5) - 2) : 0.f;
    const float *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly what the delegate's Eval does) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<_Float16> in_nchw((size_t)s.IC * IHp * IWp);
    std::vector<_Float16> w_oihw((size_t)s.OC * s.IC * s.KH * s.KW);
    std::vector<_Float16> out_nchw((size_t)s.OC * OH * OW);
    std::vector<float>    got((size_t)OH * OW * s.OC);

    rocket_in_nhwc_to_nchw_pad(in.data(), in_nchw.data(), s.IC, s.IH, s.IW,
                               pad_top, pad_left, IHp, IWp);
    rocket_filter_ohwi_to_oihw(flt.data(), w_oihw.data(), s.OC, s.KH, s.KW, s.IC);

    rocket_conv2d_desc d = {};
    d.ic = s.IC; d.ih = IHp; d.iw = IWp; d.oc = s.OC;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 0;

    /* the materialized desc must reproduce TFLite's output dims */
    int drv_oh = rocket_conv2d_oh(&d), drv_ow = rocket_conv2d_ow(&d);
    if (drv_oh != OH || drv_ow != OW) {
        printf("  driver OH/OW (%d,%d) != TFLite (%d,%d) — FAIL\n", drv_oh, drv_ow, OH, OW);
        return 1;
    }
    int plan = rocket_conv2d_plan(&d);
    if (plan) { printf("  rocket_conv2d_plan = %d — FAIL\n", plan); return 1; }

    int r = rocket_conv2d_fp16(fd, &d, in_nchw.data(), w_oihw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_fp16 = %d — FAIL\n", r); return 1; }

    rocket_out_nchw_to_nhwc_bias_act(out_nchw.data(), got.data(), s.OC, OH, OW, biasp, s.act);

    /* --- independent oracle + compare --- */
    std::vector<float> ref((size_t)OH * OW * s.OC);
    nhwc_oracle(s, in.data(), flt.data(), biasp, OH, OW, pad_top, pad_left, ref.data());

    double max_abs = 0; int bad = 0, nonfinite = 0;
    for (size_t i = 0; i < got.size(); i++) {
        // A NaN got[i] makes ad NaN, and NaN>max_abs is false, so max_abs would
        // stay 0 and the test would PASS while masking the failure. Count
        // non-finite outputs explicitly and force a FAIL on them.
        if (!std::isfinite(got[i])) {
            if (bad < 6) { printf("    [%zu] ref=%.3f got=%.3f (non-finite)\n", i, ref[i], got[i]); bad++; }
            nonfinite++;
            continue;
        }
        double ad = std::fabs(got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0 && bad < 6) {
            printf("    [%zu] ref=%.3f got=%.3f\n", i, ref[i], got[i]); bad++;
        }
    }
    (void)eff_kh; (void)eff_kw;
    const bool ok = (max_abs == 0.0 && nonfinite == 0);
    printf("  %s: max_abs=%.4f nonfinite=%d -> %s\n", fd >= 0 ? "HW glue" : "CPU-oracle glue",
           max_abs, nonfinite, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ==========================================================================
 * Quantized (int8 / uint8) CONV_2D — same glue/oracle discipline, for the
 * dequant->fp16-conv->requant boundary (rocket_*_q helpers in rocket_convert.h).
 *
 * The proof compares the delegate's EXACT quant data path against an independent
 * oracle that does the IDENTICAL numeric path (dequant operands to fp16, fp32
 * MAC, narrow to fp16, +bias, fused act, requantize with the same lrintf). It is
 * NOT compared to TFLite's int8 kernel — the dequant boundary is an fp16
 * approximation of it on purpose. Dequantized operands are kept to small integers
 * (integer scales / in-range zero points, |partial sum| << 2048) so every fp16
 * partial sum is exact and accumulation order can't hide a layout/quant bug, just
 * as the float test above pins data to {-1,0,1}.
 * ========================================================================== */
struct QShape {
    int IC, IH, IW, OC, KH, KW, sy, sx, dy, dx;
    int same, use_bias, act;
    int is_unsigned;            /* 1 = uint8 storage, 0 = int8 */
    float in_scale; int in_zp;
    float out_scale; int out_zp;
    int w_zp;                   /* shared filter zero point (0 = symmetric) */
    const char *name;
};

/* per-axis filter scale: alternate 1.0 / 2.0 across output channels so the test
 * genuinely exercises per-channel dequant (both stay integer-exact). */
static float qshape_w_scale(int oc) { return (oc & 1) ? 2.f : 1.f; }

static void set_q(unsigned char *buf, size_t k, int v, int is_unsigned)
{
    buf[k] = is_unsigned ? (unsigned char)v : (unsigned char)(signed char)v;
}

/* Independent NHWC quantized oracle: mirrors the glue's numeric path exactly
 * (operands dequantized + narrowed to fp16, fp32 accumulate, fp16 narrow, +bias,
 * act, requantize). Padding contributes real 0 (skipped), matching the glue's
 * zero-haloed input. */
static void nhwc_q_oracle(const QShape &s, const unsigned char *in,
                          const unsigned char *flt, const float *w_scale,
                          const int32_t *bias_q, int OH, int OW,
                          int pad_top, int pad_left, unsigned char *out)
{
    const int qmin = s.is_unsigned ? 0 : -128;
    const int qmax = s.is_unsigned ? 255 : 127;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < s.OC; oc++) {
                float acc = 0.f;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    if (ih < 0 || ih >= s.IH) continue;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        if (iw < 0 || iw >= s.IW) continue;
                        for (int ic = 0; ic < s.IC; ic++) {
                            int q = rocket_qread(in,
                                ((size_t)ih * s.IW + iw) * s.IC + ic, s.is_unsigned);
                            int wq = rocket_qread(flt,
                                (((size_t)oc * s.KH + kh) * s.KW + kw) * s.IC + ic,
                                s.is_unsigned);
                            float din = (float)(_Float16)(s.in_scale * (float)(q - s.in_zp));
                            float dw  = (float)(_Float16)(w_scale[oc] * (float)(wq - s.w_zp));
                            acc += din * dw;
                        }
                    }
                }
                float v = (float)(_Float16)acc;
                if (s.use_bias) v += (float)bias_q[oc] * s.in_scale * w_scale[oc];
                v = rocket_apply_act(v, s.act);
                long qo = (long)lrintf(v / s.out_scale) + s.out_zp;
                if (qo < qmin) qo = qmin;
                if (qo > qmax) qo = qmax;
                set_q(out, ((size_t)oh * OW + ow) * s.OC + oc, (int)qo, s.is_unsigned);
            }
}

static int run_q_shape(int fd, const QShape &s)
{
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);

    printf("%-26s IC=%d %dx%d -> OC=%d K=%dx%d s=%dx%d d=%dx%d %s %s bias=%d act=%d "
           "in(%.2g,%d) out(%.2g,%d) (OH=%d OW=%d)\n",
           s.name, s.IC, s.IH, s.IW, s.OC, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.is_unsigned ? "u8" : "i8", s.use_bias, s.act,
           s.in_scale, s.in_zp, s.out_scale, s.out_zp, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }

    /* TFLite NHWC quantized buffers. Dequantized values stay small integers:
     * input (q-zp) in {-1,0,1}, weights (wq-zp) in {-1,0,1}, scales in {1,2}. */
    std::vector<unsigned char> in((size_t)s.IH * s.IW * s.IC);
    std::vector<unsigned char> flt((size_t)s.OC * s.KH * s.KW * s.IC);
    std::vector<int32_t>       bias(s.OC);
    std::vector<float>         w_scale(s.OC);
    std::vector<int>           w_zp(s.OC, s.w_zp);
    for (int oc = 0; oc < s.OC; oc++) w_scale[oc] = qshape_w_scale(oc);
    for (size_t i = 0; i < in.size();  i++)
        set_q(in.data(),  i, s.in_zp + ((int)(i % 3) - 1), s.is_unsigned);
    for (size_t i = 0; i < flt.size(); i++)
        set_q(flt.data(), i, s.w_zp  + ((int)(i % 3) - 1), s.is_unsigned);
    for (int oc = 0; oc < s.OC; oc++) bias[oc] = s.use_bias ? ((oc % 5) - 2) : 0;
    const int32_t *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly what the delegate's quant Eval does) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<_Float16> in_nchw((size_t)s.IC * IHp * IWp);
    std::vector<_Float16> w_oihw((size_t)s.OC * s.IC * s.KH * s.KW);
    std::vector<_Float16> out_nchw((size_t)s.OC * OH * OW);
    std::vector<float>    bias_f(s.OC);
    std::vector<unsigned char> got((size_t)OH * OW * s.OC);

    rocket_in_q_to_nchw_pad(in.data(), s.is_unsigned, s.in_scale, s.in_zp,
                            in_nchw.data(), s.IC, s.IH, s.IW, pad_top, pad_left, IHp, IWp);
    rocket_filter_q_to_oihw(flt.data(), s.is_unsigned, w_scale.data(), w_zp.data(),
                            w_oihw.data(), s.OC, s.KH, s.KW, s.IC);
    if (biasp)
        rocket_dequant_bias(biasp, s.in_scale, w_scale.data(), bias_f.data(), s.OC);

    rocket_conv2d_desc d = {};
    d.ic = s.IC; d.ih = IHp; d.iw = IWp; d.oc = s.OC;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 0;
    if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) {
        printf("  driver OH/OW mismatch — FAIL\n"); return 1;
    }
    if (rocket_conv2d_plan(&d)) { printf("  rocket_conv2d_plan != 0 — FAIL\n"); return 1; }

    int r = rocket_conv2d_fp16(fd, &d, in_nchw.data(), w_oihw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_fp16 = %d — FAIL\n", r); return 1; }

    rocket_out_nchw_to_nhwc_q(out_nchw.data(), got.data(), s.is_unsigned, s.OC, OH, OW,
                              biasp ? bias_f.data() : nullptr, s.act, s.out_scale, s.out_zp);

    /* --- independent quant oracle + compare (exact int output) --- */
    std::vector<unsigned char> ref((size_t)OH * OW * s.OC);
    nhwc_q_oracle(s, in.data(), flt.data(), w_scale.data(), biasp, OH, OW,
                  pad_top, pad_left, ref.data());

    int bad = 0, maxd = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int g = rocket_qread(got.data(), i, s.is_unsigned);
        int e = rocket_qread(ref.data(), i, s.is_unsigned);
        int ad = abs(g - e);
        if (ad > maxd) maxd = ad;
        if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, e, g); bad++; }
    }
    printf("  %s: max|dq|=%d -> %s\n", fd >= 0 ? "HW glue" : "CPU-oracle glue",
           maxd, maxd == 0 ? "PASS" : "FAIL");
    return maxd == 0 ? 0 : 1;
}

/* ==========================================================================
 * NATIVE int8 DIRECT CONV_2D — the EXACT int8 datapath, not the fp16 approx.
 * Glue: rocket_in_i8_to_nchw_pad (pad = in_zp) -> rocket_filter_i8_to_oihw (raw int8) ->
 * rocket_eff_bias_per_axis (fold bias + input-zp correction) -> rocket_conv2d_int8 (int8
 * x int8 -> int32 on the NPU, or the int64 oracle at fd<0) -> rocket_out_nchw_to_nhwc_q_
 * per_axis. Compared to an independent oracle doing the EXACT integer math
 * (sum (in_q-in_zp)*w_q + bias_q, then scale/act/requant with the same lrintf). int8
 * (signed) only. The int32 accumulate is exact, so values can span the full range; the
 * only float step is the shared requant -> max|dq| must be 0 (exact vs the same-math
 * oracle). On HW (fd>=0) the same compare gates the real NPU int32 accumulate.
 * ========================================================================== */
static int run_ni8_shape(int fd, const QShape &s)
{
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);
    printf("%-26s IC=%d %dx%d -> OC=%d K=%dx%d s=%dx%d d=%dx%d %s bias=%d act=%d "
           "in(%.2g,%d) out(%.2g,%d) (OH=%d OW=%d)\n",
           s.name, s.IC, s.IH, s.IW, s.OC, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.use_bias, s.act,
           s.in_scale, s.in_zp, s.out_scale, s.out_zp, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }
    if (s.is_unsigned) { printf("  native int8 path is signed-only — SKIP\n"); return 0; }

    /* signed int8 tensors. Values span a moderate range (exact accumulate handles it);
     * scales keep most outputs inside int8 so the compare is discriminating. */
    std::vector<signed char> in((size_t)s.IH * s.IW * s.IC);
    std::vector<signed char> flt((size_t)s.OC * s.KH * s.KW * s.IC);
    std::vector<int32_t>     bias(s.OC);
    std::vector<float>       w_scale(s.OC);
    for (int oc = 0; oc < s.OC; oc++) w_scale[oc] = qshape_w_scale(oc);
    for (size_t i = 0; i < in.size();  i++)  in[i]  = (signed char)(s.in_zp + ((int)(i % 5) - 2));
    for (size_t i = 0; i < flt.size(); i++)  flt[i] = (signed char)(((int)(i % 7) - 3));
    for (int oc = 0; oc < s.OC; oc++) bias[oc] = s.use_bias ? ((oc % 9) - 4) * 3 : 0;
    const int32_t *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly the delegate's native_int8 Eval) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<int8_t>  in_nchw((size_t)s.IC * IHp * IWp);
    std::vector<int8_t>  w_oihw((size_t)s.OC * s.IC * s.KH * s.KW);
    std::vector<int32_t> out_nchw((size_t)s.OC * OH * OW);
    std::vector<int32_t> eff_bias(s.OC);
    std::vector<unsigned char> got((size_t)OH * OW * s.OC);

    rocket_in_i8_to_nchw_pad(in.data(), in_nchw.data(), s.IC, s.IH, s.IW, s.in_zp,
                             pad_top, pad_left, IHp, IWp);
    rocket_filter_i8_to_oihw(flt.data(), w_oihw.data(), s.OC, s.KH, s.KW, s.IC);
    rocket_eff_bias_per_axis(w_oihw.data(), biasp, s.in_zp, eff_bias.data(),
                             s.OC, s.IC, s.KH, s.KW);

    rocket_conv2d_desc d = {};
    d.ic = s.IC; d.ih = IHp; d.iw = IWp; d.oc = s.OC;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 0;
    if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) {
        printf("  driver OH/OW mismatch — FAIL\n"); return 1;
    }
    int r = rocket_conv2d_int8(fd, &d, in_nchw.data(), w_oihw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_int8 = %d — FAIL\n", r); return 1; }

    rocket_out_nchw_to_nhwc_q_per_axis(out_nchw.data(), got.data(), 0, s.OC, OH, OW,
                                       eff_bias.data(), s.act, s.in_scale, w_scale.data(),
                                       s.out_scale, s.out_zp);

    /* --- independent EXACT-int8 oracle (sum (in_q-in_zp)*w_q + bias_q, scale/act/requant) --- */
    std::vector<unsigned char> ref((size_t)OH * OW * s.OC);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < s.OC; oc++) {
                int32_t acc = 0;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        int in_range = (ih >= 0 && ih < s.IH && iw >= 0 && iw < s.IW);
                        for (int ic = 0; ic < s.IC; ic++) {
                            int q = in_range
                                ? (int)in[((size_t)ih * s.IW + iw) * s.IC + ic] : s.in_zp;
                            int wq = (int)flt[(((size_t)oc * s.KH + kh) * s.KW + kw) * s.IC + ic];
                            acc += (q - s.in_zp) * wq;
                        }
                    }
                }
                if (biasp) acc += bias[oc];
                float v = (s.in_scale * w_scale[oc]) * (float)acc;
                v = rocket_apply_act(v, s.act);
                long qo = (long)lrintf(v / s.out_scale) + s.out_zp;
                if (qo < -128) qo = -128;
                if (qo > 127) qo = 127;
                ref[((size_t)oh * OW + ow) * s.OC + oc] = (unsigned char)(signed char)qo;
            }

    int bad = 0, maxd = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int g = (int)(signed char)got[i];
        int e = (int)(signed char)ref[i];
        int ad = abs(g - e);
        if (ad > maxd) maxd = ad;
        if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, e, g); bad++; }
    }
    printf("  %s: max|dq|=%d -> %s\n", fd >= 0 ? "HW native-i8" : "CPU-oracle native-i8",
           maxd, maxd == 0 ? "PASS" : "FAIL");
    return maxd == 0 ? 0 : 1;
}

/* ==========================================================================
 * NATIVE uint8 DIRECT CONV_2D (uint8 recenter path) — the EXACT uint8 datapath. Stock
 * detectors (coral MobileDet, MediaPipe) are uint8 with an asymmetric weight zero-point,
 * so they cannot use the signed-int8 native path. Glue: rocket_in_u8_to_nchw_pad (recenter
 * x=in_q-128, pad=in_zp) -> rocket_filter_u8_to_oihw (recenter y=w_q-128) ->
 * rocket_eff_bias_u8_per_axis (fold bias + alpha*Wy + N*alpha*beta) -> rocket_conv2d_int8
 * (raw int8xint8->int32 on the recentered operands; int64 oracle at fd<0) ->
 * rocket_in_window_sum_i8 (box-sum Sx, skipped when w_zp==128) ->
 * rocket_out_nchw_to_nhwc_q_per_axis_u8 (acc + eff_bias + beta*Sx, requant to uint8).
 * Compared to an independent EXACT integer oracle (sum (in_q-in_zp)*(w_q-w_zp) + bias_q,
 * scale/act/requant with the same lrintf). The int32 accumulate is exact -> max|dq| must
 * be 0. On HW (fd>=0) the same compare gates the real NPU int32 accumulate.
 * ========================================================================== */
static int run_nu8_shape(int fd, const QShape &s)
{
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);
    printf("%-30s IC=%d %dx%d -> OC=%d K=%dx%d s=%dx%d d=%dx%d %s bias=%d act=%d "
           "in(%.2g,%d) w_zp=%d out(%.2g,%d) (OH=%d OW=%d)\n",
           s.name, s.IC, s.IH, s.IW, s.OC, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.use_bias, s.act,
           s.in_scale, s.in_zp, s.w_zp, s.out_scale, s.out_zp, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }

    /* uint8 tensors. (in_q-in_zp) in {-2..2}, (w_q-w_zp) in {-3..3}, clamped to [0,255];
     * in_zp/w_zp asymmetric (incl. in_zp=0, the post-depthwise case) so the recenter +
     * box-sum correction is genuinely exercised. Exact int32 accumulate, so the only float
     * step is the shared requant -> max|dq| must be 0 vs the same-math oracle. */
    auto clampu8 = [](int v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    std::vector<unsigned char> in((size_t)s.IH * s.IW * s.IC);
    std::vector<unsigned char> flt((size_t)s.OC * s.KH * s.KW * s.IC);
    std::vector<int32_t>       bias(s.OC);
    std::vector<float>         w_scale(s.OC);
    std::vector<int>           w_zp(s.OC, s.w_zp);
    for (int oc = 0; oc < s.OC; oc++) w_scale[oc] = qshape_w_scale(oc);
    for (size_t i = 0; i < in.size();  i++)  in[i]  = clampu8(s.in_zp + ((int)(i % 5) - 2));
    for (size_t i = 0; i < flt.size(); i++)  flt[i] = clampu8(s.w_zp  + ((int)(i % 7) - 3));
    for (int oc = 0; oc < s.OC; oc++) bias[oc] = s.use_bias ? ((oc % 9) - 4) * 3 : 0;
    const int32_t *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly the delegate's native_u8 Eval) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<int8_t>  in_nchw((size_t)s.IC * IHp * IWp);
    std::vector<int8_t>  w_oihw((size_t)s.OC * s.IC * s.KH * s.KW);
    std::vector<int32_t> out_nchw((size_t)s.OC * OH * OW);
    std::vector<int32_t> eff_bias(s.OC);
    std::vector<unsigned char> got((size_t)OH * OW * s.OC);

    rocket_in_u8_to_nchw_pad(in.data(), in_nchw.data(), s.IC, s.IH, s.IW, s.in_zp,
                             pad_top, pad_left, IHp, IWp);
    rocket_filter_u8_to_oihw(flt.data(), w_oihw.data(), s.OC, s.KH, s.KW, s.IC);
    rocket_eff_bias_u8_per_axis(w_oihw.data(), biasp, s.in_zp, w_zp.data(), eff_bias.data(),
                                s.OC, s.IC, s.KH, s.KW);

    rocket_conv2d_desc d = {};
    d.ic = s.IC; d.ih = IHp; d.iw = IWp; d.oc = s.OC;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 0;
    if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) {
        printf("  driver OH/OW mismatch — FAIL\n"); return 1;
    }
    int r = rocket_conv2d_int8(fd, &d, in_nchw.data(), w_oihw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_int8 = %d — FAIL\n", r); return 1; }

    /* box-sum only when the weight zero-point is asymmetric (beta!=0) */
    const bool need_box = (s.w_zp != 128);
    std::vector<int32_t> Sx(need_box ? (size_t)OH * OW : 0);
    if (need_box)
        rocket_in_window_sum_i8(in_nchw.data(), Sx.data(), s.IC, IHp, IWp, OH, OW,
                                s.KH, s.KW, s.sy, s.sx, s.dy, s.dx);
    rocket_out_nchw_to_nhwc_q_per_axis_u8(out_nchw.data(), got.data(), s.OC, OH, OW,
                                          eff_bias.data(), w_zp.data(),
                                          need_box ? Sx.data() : nullptr, s.act,
                                          s.in_scale, w_scale.data(), s.out_scale, s.out_zp);

    /* --- independent EXACT-uint8 oracle (sum (in_q-in_zp)*(w_q-w_zp) + bias_q, requant) --- */
    std::vector<unsigned char> ref((size_t)OH * OW * s.OC);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < s.OC; oc++) {
                int32_t acc = 0;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        int in_range = (ih >= 0 && ih < s.IH && iw >= 0 && iw < s.IW);
                        for (int ic = 0; ic < s.IC; ic++) {
                            int q = in_range
                                ? (int)in[((size_t)ih * s.IW + iw) * s.IC + ic] : s.in_zp;
                            int wq = (int)flt[(((size_t)oc * s.KH + kh) * s.KW + kw) * s.IC + ic];
                            acc += (q - s.in_zp) * (wq - s.w_zp);
                        }
                    }
                }
                if (biasp) acc += bias[oc];
                float v = (s.in_scale * w_scale[oc]) * (float)acc;
                v = rocket_apply_act(v, s.act);
                long qo = (long)lrintf(v / s.out_scale) + s.out_zp;
                if (qo < 0)   qo = 0;
                if (qo > 255) qo = 255;
                ref[((size_t)oh * OW + ow) * s.OC + oc] = (unsigned char)qo;
            }

    int bad = 0, maxd = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int ad = abs((int)got[i] - (int)ref[i]);
        if (ad > maxd) maxd = ad;
        if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, (int)ref[i], (int)got[i]); bad++; }
    }
    printf("  %s%s: max|dq|=%d -> %s\n", fd >= 0 ? "HW native-u8" : "CPU-oracle native-u8",
           need_box ? " (box-sum)" : " (sym)", maxd, maxd == 0 ? "PASS" : "FAIL");
    return maxd == 0 ? 0 : 1;
}

/* ==========================================================================
 * NATIVE int8/uint8 1x1 pointwise via the resident MATMUL (perf Step 1). A 1x1 stride-1
 * conv IS a matmul C[M,OC]=A[M,IC]*B[OC,IC]^T over the M=H*W positions, so the delegate
 * routes it to rocket_matmul_int8_prepacked (int32-raw, output columns fanned across the
 * 3 cores) — NO transpose either side (NHWC in == A, C == NHWC out). This gate validates
 * the GLUE around the matmul: the M-major requant (rocket_out_mn_to_nhwc_q_per_axis[_u8]),
 * the uint8 recenter + per-row box-sum (rocket_recenter_u8_mk), and the K/N %32 zero-pad.
 * The resident matmul needs a device, so off-HW we substitute a host int8 matmul STAND-IN
 * (C[M,Np]=A[M,Kp]*B[Np,Kp]^T — the EXACT integer product the NPU computes) and prove the
 * surrounding glue matches an independent EXACT-int oracle. The real NPU matmul is bit-exact
 * (separately validated) + the end-to-end matmul-on/off byte-identical A/B on RK3588 hardware. The
 * stand-in is pure host, so this runs identically at fd<0 and fd>=0. max|dq| must be 0.
 * ========================================================================== */
static int run_mm1x1_shape(int fd, const QShape &s)
{
    const int M = s.IH * s.IW, IC = s.IC, OC = s.OC;
    const int Kp = (IC + 31) / 32 * 32, Np = (OC + 31) / 32 * 32;
    printf("%-34s M=%d IC=%d->Kp=%d OC=%d->Np=%d %s bias=%d act=%d in(%.2g,%d) w_zp=%d out(%.2g,%d)\n",
           s.name, M, IC, Kp, OC, Np, s.is_unsigned ? "u8" : "i8", s.use_bias, s.act,
           s.in_scale, s.in_zp, s.w_zp, s.out_scale, s.out_zp);

    std::vector<float>   w_scale(OC);
    std::vector<int>     w_zp(OC, s.w_zp);
    std::vector<int32_t> bias(OC);
    for (int oc = 0; oc < OC; oc++) {
        w_scale[oc] = qshape_w_scale(oc);
        bias[oc] = s.use_bias ? ((oc % 9) - 4) * 3 : 0;
    }
    const int32_t *biasp = s.use_bias ? bias.data() : nullptr;

    /* operands (same generators as run_ni8/run_nu8), kept as int values for the oracle. */
    auto clampu8 = [](int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); };
    std::vector<int> in_v((size_t)M * IC), w_v((size_t)OC * IC);
    for (size_t i = 0; i < in_v.size(); i++)
        in_v[i] = s.is_unsigned ? clampu8(s.in_zp + ((int)(i % 5) - 2)) : (s.in_zp + ((int)(i % 5) - 2));
    for (size_t i = 0; i < w_v.size(); i++)
        w_v[i] = s.is_unsigned ? clampu8(s.w_zp + ((int)(i % 7) - 3)) : (((int)(i % 7) - 3));

    /* --- the GLUE: A[M,Kp] (+box-sum), eff_bias, padded B[Np,Kp], host matmul, MN requant --- */
    std::vector<int8_t>  A((size_t)M * Kp);
    std::vector<int8_t>  w_oihw((size_t)OC * IC);      // recentered (u8) / raw (i8) [OC][IC]
    std::vector<int8_t>  Bpad((size_t)Np * Kp, 0);
    std::vector<int32_t> C32((size_t)M * Np, 0);
    std::vector<int32_t> eff_bias(OC);
    std::vector<int32_t> Sx;
    int32_t *Sxp = nullptr;
    std::vector<unsigned char> got((size_t)M * OC);

    if (s.is_unsigned) {
        std::vector<unsigned char> in_u8(in_v.begin(), in_v.end()), w_u8(w_v.begin(), w_v.end());
        rocket_filter_u8_to_oihw(w_u8.data(), w_oihw.data(), OC, 1, 1, IC);   // y=w_q-128
        rocket_eff_bias_u8_per_axis(w_oihw.data(), biasp, s.in_zp, w_zp.data(),
                                    eff_bias.data(), OC, IC, 1, 1);
        if (s.w_zp != 128) { Sx.resize(M); Sxp = Sx.data(); }   // box-sum only when asymmetric
        rocket_recenter_u8_mk(in_u8.data(), A.data(), Sxp, M, IC, Kp);
    } else {
        std::vector<signed char> in_i8(in_v.begin(), in_v.end()), w_i8(w_v.begin(), w_v.end());
        rocket_filter_i8_to_oihw(w_i8.data(), w_oihw.data(), OC, 1, 1, IC);
        rocket_eff_bias_per_axis(w_oihw.data(), biasp, s.in_zp, eff_bias.data(), OC, IC, 1, 1);
        rocket_pad_i8_mk_band(in_i8.data(), A.data(), M, IC, Kp, 0, M);
    }
    for (int oc = 0; oc < OC; oc++)                    // padded B[Np][Kp] from w_oihw[OC][IC]
        memcpy(&Bpad[(size_t)oc * Kp], &w_oihw[(size_t)oc * IC], (size_t)IC);
    for (int m = 0; m < M; m++)                        // host int8 matmul STAND-IN (= NPU output)
        for (int n = 0; n < Np; n++) {
            int32_t acc = 0;
            const int8_t *a = &A[(size_t)m * Kp], *b = &Bpad[(size_t)n * Kp];
            for (int k = 0; k < Kp; k++) acc += (int32_t)a[k] * (int32_t)b[k];
            C32[(size_t)m * Np + n] = acc;
        }
    if (s.is_unsigned)
        rocket_out_mn_to_nhwc_q_per_axis_u8(C32.data(), got.data(), M, OC, Np, eff_bias.data(),
                w_zp.data(), Sxp, s.act, s.in_scale, w_scale.data(), s.out_scale, s.out_zp);
    else
        rocket_out_mn_to_nhwc_q_per_axis(C32.data(), got.data(), 0, M, OC, Np, eff_bias.data(),
                s.act, s.in_scale, w_scale.data(), s.out_scale, s.out_zp);

    /* --- independent EXACT-int oracle (1x1: out[m,oc] = sum_c (in-in_zp)*(w-w_zp) + bias) --- */
    std::vector<unsigned char> ref((size_t)M * OC);
    const int qmin = s.is_unsigned ? 0 : -128, qmax = s.is_unsigned ? 255 : 127;
    for (int m = 0; m < M; m++)
        for (int oc = 0; oc < OC; oc++) {
            int32_t acc = 0;
            for (int c = 0; c < IC; c++)
                acc += (in_v[(size_t)m * IC + c] - s.in_zp) * (w_v[(size_t)oc * IC + c] - s.w_zp);
            if (biasp) acc += bias[oc];
            float v = (s.in_scale * w_scale[oc]) * (float)acc;
            v = rocket_apply_act(v, s.act);
            long qo = (long)lrintf(v / s.out_scale) + s.out_zp;
            if (qo < qmin) qo = qmin;
            if (qo > qmax) qo = qmax;
            ref[(size_t)m * OC + oc] = (unsigned char)(s.is_unsigned ? qo : (signed char)qo);
        }

    int maxd = 0, bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int g = s.is_unsigned ? (int)got[i] : (int)(signed char)got[i];
        int e = s.is_unsigned ? (int)ref[i] : (int)(signed char)ref[i];
        int ad = abs(g - e);
        if (ad > maxd) maxd = ad;
        if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, e, g); bad++; }
    }
    (void)fd;   // pure host stand-in (the real NPU matmul is HW-gated end-to-end)
    printf("  mm-1x1 %s glue (host stand-in): max|dq|=%d -> %s\n",
           s.is_unsigned ? "u8" : "i8", maxd, maxd == 0 ? "PASS" : "FAIL");
    return maxd == 0 ? 0 : 1;
}

/* ==========================================================================
 * Depthwise CONV_2D — same glue/oracle discipline, for the native depthwise driver
 * path (desc.depthwise=1). The ONLY new glue vs. the direct conv is the filter
 * reorder: TFLite [1][KH][KW][C] -> driver [C][KH][KW] (rocket_dw_filter_*_to_chw).
 * Input pack, output transpose, SAME/VALID pad, bias and activation are byte-identical
 * to the direct path. Each output channel reduces ONLY its own input channel
 * (OC==IC==C), so partial sums stay tiny (<= KH*KW) and fp16-exact on {-1,0,1} data —
 * a PASS isolates the reorder + the depthwise driver path. fd<0 -> the driver's
 * depthwise CPU oracle (x86); fd>=0 -> the native DW NPU job (the RK3588 HW gate).
 * ========================================================================== */
struct DWShape {
    int C, IH, IW, KH, KW, sy, sx, dy, dx;
    int same, use_bias, act;
    const char *name;
};

/* Independent NHWC depthwise oracle. out[oh,ow,c] = act(bias[c] +
 * sum_{kh,kw} in[ih,iw,c] * dwflt[kh,kw,c]); fp32 accumulate, fp16 narrow like the
 * NPU output, then +bias, then act. dwflt is TFLite [1][KH][KW][C]. */
static void nhwc_dw_oracle(const DWShape &s, const float *in, const float *flt,
                           const float *bias, int OH, int OW,
                           int pad_top, int pad_left, float *out)
{
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < s.C; c++) {
                float acc = 0.f;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    if (ih < 0 || ih >= s.IH) continue;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        if (iw < 0 || iw >= s.IW) continue;
                        acc += in[((size_t)ih * s.IW + iw) * s.C + c] *
                               flt[((size_t)kh * s.KW + kw) * s.C + c];
                    }
                }
                float v = (float)(_Float16)acc;          /* narrow like the NPU fp16 out */
                if (bias) v += bias[c];
                out[((size_t)oh * OW + ow) * s.C + c] = rocket_apply_act(v, s.act);
            }
}

static int run_dw_shape(int fd, const DWShape &s)
{
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);

    printf("%-24s C=%d %dx%d K=%dx%d s=%dx%d d=%dx%d %s bias=%d act=%d (OH=%d OW=%d)\n",
           s.name, s.C, s.IH, s.IW, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.use_bias, s.act, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }

    /* TFLite NHWC buffers: input [1][IH][IW][C], depthwise filter [1][KH][KW][C]. */
    std::vector<float> in((size_t)s.IH * s.IW * s.C);
    std::vector<float> flt((size_t)s.KH * s.KW * s.C);
    std::vector<float> bias(s.C);
    for (size_t i = 0; i < in.size();  i++) in[i]  = (float)((int)(i % 3) - 1);
    for (size_t i = 0; i < flt.size(); i++) flt[i] = (float)((int)(i % 3) - 1);
    for (int i = 0; i < s.C; i++) bias[i] = s.use_bias ? (float)((i % 5) - 2) : 0.f;
    const float *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly what the delegate's depthwise Eval does) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<_Float16> in_nchw((size_t)s.C * IHp * IWp);
    std::vector<_Float16> w_chw((size_t)s.C * s.KH * s.KW);
    std::vector<_Float16> out_nchw((size_t)s.C * OH * OW);
    std::vector<float>    got((size_t)OH * OW * s.C);

    rocket_in_nhwc_to_nchw_pad(in.data(), in_nchw.data(), s.C, s.IH, s.IW,
                               pad_top, pad_left, IHp, IWp);
    rocket_dw_filter_hwc_to_chw(flt.data(), w_chw.data(), s.C, s.KH, s.KW);

    rocket_conv2d_desc d = {};
    d.ic = s.C; d.ih = IHp; d.iw = IWp; d.oc = s.C;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 1;

    int drv_oh = rocket_conv2d_oh(&d), drv_ow = rocket_conv2d_ow(&d);
    if (drv_oh != OH || drv_ow != OW) {
        printf("  driver OH/OW (%d,%d) != TFLite (%d,%d) — FAIL\n", drv_oh, drv_ow, OH, OW);
        return 1;
    }
    int plan = rocket_conv2d_plan(&d);
    if (plan) { printf("  rocket_conv2d_plan = %d — FAIL\n", plan); return 1; }

    int r = rocket_conv2d_fp16(fd, &d, in_nchw.data(), w_chw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_fp16 = %d — FAIL\n", r); return 1; }

    rocket_out_nchw_to_nhwc_bias_act(out_nchw.data(), got.data(), s.C, OH, OW, biasp, s.act);

    /* --- independent oracle + compare --- */
    std::vector<float> ref((size_t)OH * OW * s.C);
    nhwc_dw_oracle(s, in.data(), flt.data(), biasp, OH, OW, pad_top, pad_left, ref.data());

    double max_abs = 0; int bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double ad = std::fabs(got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0 && bad < 6) { printf("    [%zu] ref=%.3f got=%.3f\n", i, ref[i], got[i]); bad++; }
    }
    printf("  %s: max_abs=%.4f -> %s\n", fd >= 0 ? "HW glue" : "CPU-oracle glue",
           max_abs, max_abs == 0.0 ? "PASS" : "FAIL");
    return max_abs == 0.0 ? 0 : 1;
}

/* ---- quantized (int8 / uint8) depthwise: dequant->fp16-DW->requant boundary ---- */
struct DWQShape {
    int C, IH, IW, KH, KW, sy, sx, dy, dx;
    int same, use_bias, act;
    int is_unsigned;
    float in_scale; int in_zp;
    float out_scale; int out_zp;
    int w_zp;                   /* shared filter zero point (0 = symmetric) */
    const char *name;
};

/* per-channel filter scale: alternate 1.0 / 2.0 across channels (both integer-exact),
 * so the test exercises per-channel dequant along the depthwise C axis. */
static float dwshape_w_scale(int c) { return (c & 1) ? 2.f : 1.f; }

/* Independent NHWC quantized depthwise oracle: mirrors the glue's numeric path
 * (dequant operands + narrow to fp16, fp32 accumulate, fp16 narrow, +bias, act,
 * requantize with the same lrintf). dwflt is TFLite [1][KH][KW][C]. */
static void nhwc_dw_q_oracle(const DWQShape &s, const unsigned char *in,
                             const unsigned char *flt, const float *w_scale,
                             const int32_t *bias_q, int OH, int OW,
                             int pad_top, int pad_left, unsigned char *out)
{
    const int qmin = s.is_unsigned ? 0 : -128;
    const int qmax = s.is_unsigned ? 255 : 127;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < s.C; c++) {
                float acc = 0.f;
                for (int kh = 0; kh < s.KH; kh++) {
                    int ih = oh * s.sy + kh * s.dy - pad_top;
                    if (ih < 0 || ih >= s.IH) continue;
                    for (int kw = 0; kw < s.KW; kw++) {
                        int iw = ow * s.sx + kw * s.dx - pad_left;
                        if (iw < 0 || iw >= s.IW) continue;
                        int q  = rocket_qread(in,  ((size_t)ih * s.IW + iw) * s.C + c, s.is_unsigned);
                        int wq = rocket_qread(flt, ((size_t)kh * s.KW + kw) * s.C + c, s.is_unsigned);
                        float din = (float)(_Float16)(s.in_scale * (float)(q - s.in_zp));
                        float dw  = (float)(_Float16)(w_scale[c] * (float)(wq - s.w_zp));
                        acc += din * dw;
                    }
                }
                float v = (float)(_Float16)acc;
                if (s.use_bias) v += (float)bias_q[c] * s.in_scale * w_scale[c];
                v = rocket_apply_act(v, s.act);
                long qo = (long)lrintf(v / s.out_scale) + s.out_zp;
                if (qo < qmin) qo = qmin;
                if (qo > qmax) qo = qmax;
                set_q(out, ((size_t)oh * OW + ow) * s.C + c, (int)qo, s.is_unsigned);
            }
}

static int run_dw_q_shape(int fd, const DWQShape &s)
{
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, s.dy, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, s.dx, s.same);

    printf("%-26s C=%d %dx%d K=%dx%d s=%dx%d d=%dx%d %s %s bias=%d act=%d "
           "in(%.2g,%d) out(%.2g,%d) (OH=%d OW=%d)\n",
           s.name, s.C, s.IH, s.IW, s.KH, s.KW, s.sy, s.sx, s.dy, s.dx,
           s.same ? "SAME" : "VALID", s.is_unsigned ? "u8" : "i8", s.use_bias, s.act,
           s.in_scale, s.in_zp, s.out_scale, s.out_zp, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad output dims — SKIP\n"); return 0; }

    /* TFLite NHWC quantized buffers: input [1][IH][IW][C], filter [1][KH][KW][C]. */
    std::vector<unsigned char> in((size_t)s.IH * s.IW * s.C);
    std::vector<unsigned char> flt((size_t)s.KH * s.KW * s.C);
    std::vector<int32_t>       bias(s.C);
    std::vector<float>         w_scale(s.C);
    std::vector<int>           w_zp(s.C, s.w_zp);
    for (int c = 0; c < s.C; c++) w_scale[c] = dwshape_w_scale(c);
    for (size_t i = 0; i < in.size();  i++)
        set_q(in.data(),  i, s.in_zp + ((int)(i % 3) - 1), s.is_unsigned);
    for (size_t i = 0; i < flt.size(); i++)
        set_q(flt.data(), i, s.w_zp  + ((int)(i % 3) - 1), s.is_unsigned);
    for (int c = 0; c < s.C; c++) bias[c] = s.use_bias ? ((c % 5) - 2) : 0;
    const int32_t *biasp = s.use_bias ? bias.data() : nullptr;

    /* --- the GLUE (exactly what the delegate's quant depthwise Eval does) --- */
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, s.dy, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, s.dx, OW);
    const int pad_top = tot_h / 2, pad_left = tot_w / 2;
    const int IHp = s.IH + tot_h, IWp = s.IW + tot_w;

    std::vector<_Float16> in_nchw((size_t)s.C * IHp * IWp);
    std::vector<_Float16> w_chw((size_t)s.C * s.KH * s.KW);
    std::vector<_Float16> out_nchw((size_t)s.C * OH * OW);
    std::vector<float>    bias_f(s.C);
    std::vector<unsigned char> got((size_t)OH * OW * s.C);

    rocket_in_q_to_nchw_pad(in.data(), s.is_unsigned, s.in_scale, s.in_zp,
                            in_nchw.data(), s.C, s.IH, s.IW, pad_top, pad_left, IHp, IWp);
    rocket_dw_filter_q_to_chw(flt.data(), s.is_unsigned, w_scale.data(), w_zp.data(),
                              w_chw.data(), s.C, s.KH, s.KW);
    if (biasp)
        rocket_dequant_bias(biasp, s.in_scale, w_scale.data(), bias_f.data(), s.C);

    rocket_conv2d_desc d = {};
    d.ic = s.C; d.ih = IHp; d.iw = IWp; d.oc = s.C;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = 1;
    if (rocket_conv2d_oh(&d) != OH || rocket_conv2d_ow(&d) != OW) {
        printf("  driver OH/OW mismatch — FAIL\n"); return 1;
    }
    if (rocket_conv2d_plan(&d)) { printf("  rocket_conv2d_plan != 0 — FAIL\n"); return 1; }

    int r = rocket_conv2d_fp16(fd, &d, in_nchw.data(), w_chw.data(), out_nchw.data());
    if (r) { printf("  rocket_conv2d_fp16 = %d — FAIL\n", r); return 1; }

    rocket_out_nchw_to_nhwc_q(out_nchw.data(), got.data(), s.is_unsigned, s.C, OH, OW,
                              biasp ? bias_f.data() : nullptr, s.act, s.out_scale, s.out_zp);

    /* --- independent quant oracle + compare (exact int output) --- */
    std::vector<unsigned char> ref((size_t)OH * OW * s.C);
    nhwc_dw_q_oracle(s, in.data(), flt.data(), w_scale.data(), biasp, OH, OW,
                     pad_top, pad_left, ref.data());

    int bad = 0, maxd = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int g = rocket_qread(got.data(), i, s.is_unsigned);
        int e = rocket_qread(ref.data(), i, s.is_unsigned);
        int ad = abs(g - e);
        if (ad > maxd) maxd = ad;
        if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, e, g); bad++; }
    }
    printf("  %s: max|dq|=%d -> %s\n", fd >= 0 ? "HW glue" : "CPU-oracle glue",
           maxd, maxd == 0 ? "PASS" : "FAIL");
    return maxd == 0 ? 0 : 1;
}

/* ==========================================================================
 * Aux host ops (ADD / POOL / CONCAT / RESHAPE) — the elementwise / pooling / join
 * ops a detector graph carries around its convs (rocket_ops.h). These are pure HOST
 * kernels (no NPU, no NCHW transpose), so the proof is device-independent: each runs
 * the REAL delegate kernel and compares to an INDEPENDENT NHWC oracle that performs
 * the same reference semantics by a different traversal. Quant operands are kept to
 * small integers (exact dequant) and the oracle mirrors the kernel's lrintf requant,
 * so a PASS (max_abs=0 / max|dq|=0) is a clean statement about the indexing, axis /
 * window handling, and per-tensor scale/zp plumbing — not bit-equality with TFLite's
 * fixed-point int8 kernels (these are an fp32-host approximation, like the conv path).
 * ========================================================================== */

/* ---- UNARY ACTIVATION (HARD_SWISH / LOGISTIC / hardsigmoid) ---- */
struct ActT {
    int H, W, C, kind, is_q, u;
    float in_s; int in_z; float out_s; int out_z;
    const char *name;
    float param;            /* LEAKY_RELU slope; 0 (unused) for every other kind */
};
/* Independent oracle: the same float ops rocket_unary_eval uses, re-written here so a
 * PASS (max_abs=0 / max|dq|=0) is a clean statement about the f/q loop + dequant/requant
 * plumbing (the math primitive is trivial + matched to the driver's rocket_activation.c). */
static float act_ref_f(int kind, float x, float param) {
    switch (kind) {
    case ROCKET_UNARY_HARDSWISH:  { float r = x/6.0f + 0.5f; r = r<0.f?0.f:(r>1.f?1.f:r); return x*r; }
    case ROCKET_UNARY_SIGMOID:    return 1.0f/(1.0f+expf(-x));
    case ROCKET_UNARY_HARDSIGMOID:{ float r = x/6.0f + 0.5f; return r<0.f?0.f:(r>1.f?1.f:r); }
    case ROCKET_UNARY_TANH:       return tanhf(x);
    case ROCKET_UNARY_ELU:        return x >= 0.f ? x : (expf(x) - 1.0f);
    case ROCKET_UNARY_LOG:        return logf(x);
    case ROCKET_UNARY_RELU:       return x > 0.f ? x : 0.f;
    case ROCKET_UNARY_RELU6:      return x < 0.f ? 0.f : (x > 6.f ? 6.f : x);
    case ROCKET_UNARY_RELU_N1_1:  return x < -1.f ? -1.f : (x > 1.f ? 1.f : x);
    case ROCKET_UNARY_LEAKY_RELU: return x >= 0.f ? x : param * x;
    case ROCKET_UNARY_EXP:        return expf(x);
    case ROCKET_UNARY_SQRT:       return sqrtf(x);
    case ROCKET_UNARY_RSQRT:      return 1.0f/sqrtf(x);
    case ROCKET_UNARY_ABS:        return fabsf(x);
    case ROCKET_UNARY_NEG:        return -x;
    case ROCKET_UNARY_SQUARE:     return x * x;
    case ROCKET_UNARY_FLOOR:      return floorf(x);
    default: return x;
    }
}
static int run_act(const ActT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    const char *kn = s.kind == ROCKET_UNARY_HARDSWISH ? "hardswish"
                   : s.kind == ROCKET_UNARY_SIGMOID   ? "sigmoid"
                   : s.kind == ROCKET_UNARY_TANH      ? "tanh"
                   : s.kind == ROCKET_UNARY_ELU       ? "elu"
                   : s.kind == ROCKET_UNARY_LOG       ? "log"
                   : s.kind == ROCKET_UNARY_RELU      ? "relu"
                   : s.kind == ROCKET_UNARY_RELU6     ? "relu6"
                   : s.kind == ROCKET_UNARY_RELU_N1_1 ? "relu_n1_1"
                   : s.kind == ROCKET_UNARY_LEAKY_RELU? "leaky_relu"
                   : s.kind == ROCKET_UNARY_EXP       ? "exp"
                   : s.kind == ROCKET_UNARY_SQRT      ? "sqrt"
                   : s.kind == ROCKET_UNARY_RSQRT     ? "rsqrt"
                   : s.kind == ROCKET_UNARY_ABS       ? "abs"
                   : s.kind == ROCKET_UNARY_NEG       ? "neg"
                   : s.kind == ROCKET_UNARY_SQUARE    ? "square"
                   : s.kind == ROCKET_UNARY_FLOOR     ? "floor" : "hardsigmoid";
    /* sqrt/rsqrt/log need x>0 (sqrt(neg)=nan, rsqrt(0)=inf) -> a positive input sweep */
    const bool pos_only = s.kind == ROCKET_UNARY_LOG || s.kind == ROCKET_UNARY_SQRT
                       || s.kind == ROCKET_UNARY_RSQRT;
    printf("%-26s %dx%dx%d %s %s\n", s.name, s.H, s.W, s.C, kn,
           s.is_q ? (s.u ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> in(n), got(n);
        for (size_t i = 0; i < n; i++) in[i] = pos_only ? 0.1f + (float)(i % 25) * 0.5f       /* (0,12.1] */
                                                        : (float)((int)(i % 25) - 12) * 0.5f; /* [-6,6]  */
        rocket_unary_f(in.data(), got.data(), n, s.kind, s.param);
        double md = 0;
        for (size_t i = 0; i < n; i++)
            md = std::fmax(md, std::fabs((double)got[i] - (double)act_ref_f(s.kind, in[i], s.param)));
        printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> in(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) {
        int q = s.in_z + ((int)(i % 21) - 10);
        if (pos_only && (s.in_s * (float)(q - s.in_z)) <= 0.f) q = s.in_z + 1 + (int)(i % 10);  /* keep dequant > 0 */
        set_q(in.data(), i, q, s.u);
    }
    rocket_unary_q(in.data(), got.data(), n, s.kind, s.u, s.u, s.in_s, s.in_z, s.out_s, s.out_z, s.param);
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv = 1.f / s.out_s;
    for (size_t i = 0; i < n; i++) {            /* independent dequant->f->requant oracle */
        float v = act_ref_f(s.kind, s.in_s * (float)(rocket_qread(in.data(), i, s.u) - s.in_z), s.param);
        long q = (long)lrintf(v * inv) + s.out_z; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), i, (int)q, s.u);
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.u) - rocket_qread(ref.data(), i, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- FULLY_CONNECTED (float host kernel rocket_fc_f) ---- */
struct FCT { int M, K, N, act, bias; const char *name; };
static int run_fc(const FCT &s) {
    const int M = s.M, K = s.K, N = s.N;
    printf("%-26s M=%d K=%d N=%d act=%d %s\n", s.name, M, K, N, s.act, s.bias ? "+bias" : "no-bias");
    std::vector<float> A((size_t)M*K), B((size_t)N*K), bias(N), got((size_t)M*N), ref((size_t)M*N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (float)((int)(i % 7) - 3) * 0.5f;
    for (size_t i = 0; i < B.size(); i++) B[i] = (float)((int)(i % 5) - 2) * 0.25f;
    for (int n = 0; n < N; n++) bias[n] = (float)((n % 3) - 1);
    rocket_fc_f(A.data(), B.data(), s.bias ? bias.data() : nullptr, got.data(), M, K, N, s.act);
    for (int m = 0; m < M; m++)              /* independent same-precision oracle */
        for (int n = 0; n < N; n++) {
            float acc = s.bias ? bias[n] : 0.f;
            for (int k = 0; k < K; k++) acc += A[(size_t)m*K+k] * B[(size_t)n*K+k];
            ref[(size_t)m*N+n] = rocket_apply_act(acc, s.act);
        }
    double md = 0; for (size_t i = 0; i < got.size(); i++) md = std::fmax(md, std::fabs(got[i]-ref[i]));
    printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
    return md == 0.0 ? 0 : 1;
}

/* ---- ADD (elementwise residual, same shape, fused act) ---- */
struct AddT {
    int H, W, C, act, is_q;
    int au, bu, ou;                 /* uint8 storage flags (quant) */
    float as; int az; float bs; int bz; float os; int oz;
    const char *name;
};
static int run_add(const AddT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    printf("%-26s %dx%dx%d act=%d %s\n", s.name, s.H, s.W, s.C, s.act,
           s.is_q ? (s.ou ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> a(n), b(n), got(n), ref(n);
        for (size_t i = 0; i < n; i++) { a[i] = (float)((int)(i % 5) - 2);
                                         b[i] = (float)((int)((i / 3) % 5) - 2); }
        rocket_add_f(a.data(), b.data(), got.data(), n, s.act);
        for (size_t i = 0; i < n; i++) ref[i] = rocket_apply_act(a[i] + b[i], s.act);
        double md = 0; for (size_t i = 0; i < n; i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  max_abs=%.4f -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> a(n), b(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) { set_q(a.data(), i, s.az + ((int)(i % 3) - 1), s.au);
                                     set_q(b.data(), i, s.bz + ((int)((i / 2) % 3) - 1), s.bu); }
    rocket_add_q(a.data(), b.data(), got.data(), n, s.au, s.bu, s.ou,
                 s.as, s.az, s.bs, s.bz, s.os, s.oz, s.act);
    const int qmin = s.ou ? 0 : -128, qmax = s.ou ? 255 : 127; const float inv = 1.f / s.os;
    for (size_t i = 0; i < n; i++) {            /* independent oracle */
        float va = s.as * (float)(rocket_qread(a.data(), i, s.au) - s.az);
        float vb = s.bs * (float)(rocket_qread(b.data(), i, s.bu) - s.bz);
        float v = rocket_apply_act(va + vb, s.act);
        long q = (long)lrintf(v * inv) + s.oz; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), i, (int)q, s.ou);
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.ou) - rocket_qread(ref.data(), i, s.ou)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- BINARY MAXIMUM / MINIMUM (elementwise two-tensor, same shape, no act) ---- */
struct BinT {
    int H, W, C, op, is_q;          /* op = ROCKET_BINOP_MAX / _MIN */
    int au, bu, ou;
    float as; int az; float bs; int bz; float os; int oz;
    const char *name;
};
static int run_binary(const BinT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    printf("%-26s %dx%dx%d %s %s\n", s.name, s.H, s.W, s.C,
           s.op == ROCKET_BINOP_MIN ? "min" : "max",
           s.is_q ? (s.ou ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> a(n), b(n), got(n), ref(n);
        for (size_t i = 0; i < n; i++) { a[i] = (float)((int)(i % 7) - 3);
                                         b[i] = (float)((int)((i / 3) % 7) - 3); }
        rocket_binary_f(a.data(), b.data(), got.data(), n, s.op);
        for (size_t i = 0; i < n; i++)
            ref[i] = s.op == ROCKET_BINOP_MIN ? std::fmin(a[i], b[i]) : std::fmax(a[i], b[i]);
        double md = 0; for (size_t i = 0; i < n; i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  max_abs=%.4f -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> a(n), b(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) { set_q(a.data(), i, s.az + ((int)(i % 5) - 2), s.au);
                                     set_q(b.data(), i, s.bz + ((int)((i / 2) % 5) - 2), s.bu); }
    rocket_binary_q(a.data(), b.data(), got.data(), n, s.op, s.au, s.bu, s.ou,
                    s.as, s.az, s.bs, s.bz, s.os, s.oz);
    const int qmin = s.ou ? 0 : -128, qmax = s.ou ? 255 : 127; const float inv = 1.f / s.os;
    for (size_t i = 0; i < n; i++) {            /* independent dequant->op->requant oracle */
        float va = s.as * (float)(rocket_qread(a.data(), i, s.au) - s.az);
        float vb = s.bs * (float)(rocket_qread(b.data(), i, s.bu) - s.bz);
        float v = s.op == ROCKET_BINOP_MIN ? std::fmin(va, vb) : std::fmax(va, vb);
        long q = (long)lrintf(v * inv) + s.oz; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), i, (int)q, s.ou);
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.ou) - rocket_qread(ref.data(), i, s.ou)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- MUL / SUB / DIV (elementwise two-tensor, same shape, FUSED act) ---- */
struct ArithT {
    int H, W, C, op, act, is_q;     /* op = ROCKET_ARITH_MUL/SUB/DIV */
    int au, bu, ou;
    float as; int az; float bs; int bz; float os; int oz;
    const char *name;
};
static int run_arith(const ArithT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    const char *on = s.op == ROCKET_ARITH_MUL ? "mul" : s.op == ROCKET_ARITH_SUB ? "sub" : "div";
    printf("%-26s %dx%dx%d %s act=%d %s\n", s.name, s.H, s.W, s.C, on, s.act,
           s.is_q ? (s.ou ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> a(n), b(n), got(n), ref(n);
        for (size_t i = 0; i < n; i++) { a[i] = (float)((int)(i % 7) - 3);
                                         b[i] = (float)((int)(i % 4) + 1); }   /* b != 0 (div) */
        rocket_arith_f(a.data(), b.data(), got.data(), n, s.op, s.act);
        for (size_t i = 0; i < n; i++) {            /* independent oracle */
            float v = s.op == ROCKET_ARITH_MUL ? a[i] * b[i]
                    : s.op == ROCKET_ARITH_SUB ? a[i] - b[i] : a[i] / b[i];
            ref[i] = rocket_apply_act(v, s.act);
        }
        double md = 0; for (size_t i = 0; i < n; i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> a(n), b(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) { set_q(a.data(), i, s.az + ((int)(i % 5) - 2), s.au);
                                     set_q(b.data(), i, s.bz + ((int)(i % 4) + 1), s.bu); }  /* vb != 0 */
    rocket_arith_q(a.data(), b.data(), got.data(), n, s.op, s.au, s.bu, s.ou,
                   s.as, s.az, s.bs, s.bz, s.os, s.oz, s.act);
    const int qmin = s.ou ? 0 : -128, qmax = s.ou ? 255 : 127; const float inv = 1.f / s.os;
    for (size_t i = 0; i < n; i++) {            /* independent dequant->op->act->requant oracle */
        float va = s.as * (float)(rocket_qread(a.data(), i, s.au) - s.az);
        float vb = s.bs * (float)(rocket_qread(b.data(), i, s.bu) - s.bz);
        float v = s.op == ROCKET_ARITH_MUL ? va * vb : s.op == ROCKET_ARITH_SUB ? va - vb : va / vb;
        v = rocket_apply_act(v, s.act);
        long q = (long)lrintf(v * inv) + s.oz; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), i, (int)q, s.ou);
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.ou) - rocket_qread(ref.data(), i, s.ou)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- PRELU (per-channel parametric ReLU, NHWC channel-innermost) ---- */
struct PreluT {
    int H, W, C, is_q, u;
    float in_s; int in_z; float out_s; int out_z;
    const char *name;
};
static int run_prelu(const PreluT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    printf("%-26s %dx%dx%d prelu %s\n", s.name, s.H, s.W, s.C,
           s.is_q ? (s.u ? "u8" : "i8") : "float");
    std::vector<float> alpha((size_t)s.C);
    for (int c = 0; c < s.C; c++) alpha[c] = 0.05f + 0.1f * (float)(c % 5);   /* per-channel slopes */
    if (!s.is_q) {
        std::vector<float> in(n), got(n), ref(n);
        for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 13) - 6) * 0.5f;
        rocket_prelu_f(in.data(), got.data(), n, s.C, alpha.data());
        for (size_t i = 0; i < n; i++) { float x = in[i]; ref[i] = x >= 0.f ? x : alpha[i % s.C] * x; }
        double md = 0; for (size_t i = 0; i < n; i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> in(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) set_q(in.data(), i, s.in_z + ((int)(i % 21) - 10), s.u);
    rocket_prelu_q(in.data(), got.data(), n, s.C, alpha.data(), s.u, s.u,
                   s.in_s, s.in_z, s.out_s, s.out_z);
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv = 1.f / s.out_s;
    for (size_t i = 0; i < n; i++) {            /* independent dequant->op->requant oracle */
        float x = s.in_s * (float)(rocket_qread(in.data(), i, s.u) - s.in_z);
        float v = x >= 0.f ? x : alpha[i % s.C] * x;
        long q = (long)lrintf(v * inv) + s.out_z; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), i, (int)q, s.u);
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.u) - rocket_qread(ref.data(), i, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- SPATIAL REDUCE (mean / max / min over [H,W], per channel) ---- */
struct ReduceT {
    int H, W, C, op, is_q, u;
    float in_s; int in_z; float out_s; int out_z;
    const char *name;
};
static int run_reduce(const ReduceT &s) {
    const size_t n = (size_t)s.H * s.W * s.C;
    const char *on = s.op == ROCKET_REDUCE_MEAN ? "mean" : s.op == ROCKET_REDUCE_MAX ? "max" : "min";
    printf("%-26s %dx%dx%d reduce_%s %s\n", s.name, s.H, s.W, s.C, on,
           s.is_q ? (s.u ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> in(n), got((size_t)s.C), ref((size_t)s.C);
        for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 17) - 8) * 0.25f;
        rocket_reduce_spatial_f(in.data(), got.data(), s.H, s.W, s.C, s.op);
        const size_t HW = (size_t)s.H * s.W;
        for (int c = 0; c < s.C; c++) {
            double acc = s.op == ROCKET_REDUCE_MEAN ? 0.0
                       : s.op == ROCKET_REDUCE_MIN ? 1e30 : -1e30;
            for (size_t p = 0; p < HW; p++) { double v = in[p*(size_t)s.C + c];
                if (s.op==ROCKET_REDUCE_MEAN) acc+=v; else if (s.op==ROCKET_REDUCE_MIN){if(v<acc)acc=v;} else {if(v>acc)acc=v;} }
            ref[c] = s.op == ROCKET_REDUCE_MEAN ? (float)(acc/(double)HW) : (float)acc;
        }
        double md = 0; for (int c = 0; c < s.C; c++) md = std::fmax(md, std::fabs((double)got[c]-(double)ref[c]));
        printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<unsigned char> in(n), got((size_t)s.C), ref((size_t)s.C);
    for (size_t i = 0; i < n; i++) set_q(in.data(), i, s.in_z + ((int)(i % 23) - 11), s.u);
    rocket_reduce_spatial_q(in.data(), got.data(), s.H, s.W, s.C, s.op, s.u, s.u,
                            s.in_s, s.in_z, s.out_s, s.out_z);
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv = 1.f / s.out_s;
    const size_t HW = (size_t)s.H * s.W;
    for (int c = 0; c < s.C; c++) {            /* independent dequant->reduce->requant oracle */
        float acc = s.op == ROCKET_REDUCE_MEAN ? 0.f : s.op == ROCKET_REDUCE_MIN ? INFINITY : -INFINITY;
        for (size_t p = 0; p < HW; p++) { float v = s.in_s * (float)(rocket_qread(in.data(), p*(size_t)s.C+c, s.u) - s.in_z);
            if (s.op==ROCKET_REDUCE_MEAN) acc+=v; else if (s.op==ROCKET_REDUCE_MIN){if(v<acc)acc=v;} else {if(v>acc)acc=v;} }
        float r = s.op == ROCKET_REDUCE_MEAN ? acc/(float)HW : acc;
        long q = (long)lrintf(r * inv) + s.out_z; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), c, (int)q, s.u);
    }
    int md = 0; for (int c = 0; c < s.C; c++)
        md = std::max(md, abs(rocket_qread(got.data(), c, s.u) - rocket_qread(ref.data(), c, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- LOG_SOFTMAX (x - logsumexp over the last axis) ---- */
struct LSMT { int M, N; const char *name; };
static int run_logsoftmax(const LSMT &s) {
    const size_t n = (size_t)s.M * s.N;
    printf("%-26s M=%d N=%d float\n", s.name, s.M, s.N);
    std::vector<float> in(n), got(n);
    for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 13) - 6) * 0.5f;
    rocket_logsoftmax_f(in.data(), got.data(), s.M, s.N);
    double md = 0;
    for (int m = 0; m < s.M; m++) {
        double mx = -1e30; for (int j = 0; j < s.N; j++) if (in[(size_t)m*s.N+j] > mx) mx = in[(size_t)m*s.N+j];
        double ss = 0; for (int j = 0; j < s.N; j++) ss += std::exp(in[(size_t)m*s.N+j]-mx);
        double lse = mx + std::log(ss);
        for (int j = 0; j < s.N; j++) md = std::fmax(md, std::fabs((double)got[(size_t)m*s.N+j] - (in[(size_t)m*s.N+j]-lse)));
    }
    printf("  max_abs=%.6g -> %s\n", md, md < 1e-5 ? "PASS" : "FAIL");
    return md < 1e-5 ? 0 : 1;
}

/* ---- SOFTMAX (exp(beta*x)/sum over the last axis) ---- */
struct SMT { int M, N; float beta; const char *name; };
static int run_softmax(const SMT &s) {
    const size_t n = (size_t)s.M * s.N;
    printf("%-26s M=%d N=%d beta=%.3g float\n", s.name, s.M, s.N, s.beta);
    std::vector<float> in(n), got(n);
    for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 13) - 6) * 0.5f;
    rocket_softmax_f(in.data(), got.data(), s.M, s.N, s.beta);
    double md = 0, rowsum_err = 0;
    for (int m = 0; m < s.M; m++) {
        double mx = -1e30; for (int j = 0; j < s.N; j++) if (in[(size_t)m*s.N+j] > mx) mx = in[(size_t)m*s.N+j];
        double ss = 0; for (int j = 0; j < s.N; j++) ss += std::exp((double)s.beta*(in[(size_t)m*s.N+j]-mx));
        double rs = 0;
        for (int j = 0; j < s.N; j++) {
            double r = std::exp((double)s.beta*(in[(size_t)m*s.N+j]-mx)) / ss;
            md = std::fmax(md, std::fabs((double)got[(size_t)m*s.N+j] - r));
            rs += got[(size_t)m*s.N+j];
        }
        rowsum_err = std::fmax(rowsum_err, std::fabs(rs - 1.0));   /* probabilities sum to 1 */
    }
    printf("  max_abs=%.6g rowsum_err=%.6g -> %s\n", md, rowsum_err,
           (md < 1e-5 && rowsum_err < 1e-5) ? "PASS" : "FAIL");
    return (md < 1e-5 && rowsum_err < 1e-5) ? 0 : 1;
}

/* ---- CUMSUM (prefix sum along the last axis, 4 variants) ---- */
struct CSMT { int M, N, excl, rev; const char *name; };
static int run_cumsum(const CSMT &s) {
    const size_t n = (size_t)s.M * s.N;
    printf("%-26s M=%d N=%d excl=%d rev=%d float\n", s.name, s.M, s.N, s.excl, s.rev);
    std::vector<float> in(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 9) - 4);
    rocket_cumsum_f(in.data(), got.data(), s.M, s.N, s.excl, s.rev);
    for (int m = 0; m < s.M; m++) {
        double acc = 0;
        for (int i = 0; i < s.N; i++) { int j = s.rev ? s.N-1-i : i;
            if (s.excl) { ref[(size_t)m*s.N+j] = (float)acc; acc += in[(size_t)m*s.N+j]; }
            else        { acc += in[(size_t)m*s.N+j]; ref[(size_t)m*s.N+j] = (float)acc; } }
    }
    double md = 0; for (size_t i = 0; i < n; i++) md = std::fmax(md, std::fabs((double)got[i]-(double)ref[i]));
    printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
    return md == 0.0 ? 0 : 1;
}

/* ---- L2_NORMALIZATION (normalize over the channel axis) ---- */
struct L2T { int M, C, is_q, u; float in_s; int in_z; float out_s; int out_z; const char *name; };
static int run_l2norm(const L2T &s) {
    const size_t n = (size_t)s.M * s.C;
    printf("%-26s M=%d C=%d %s\n", s.name, s.M, s.C, s.is_q ? (s.u ? "u8" : "i8") : "float");
    if (!s.is_q) {
        std::vector<float> in(n), got(n);
        for (size_t i = 0; i < n; i++) in[i] = (float)((int)(i % 11) - 5) * 0.5f;
        rocket_l2norm_f(in.data(), got.data(), s.M, s.C);
        double md = 0;
        for (int m = 0; m < s.M; m++) {
            double ss = 0; for (int c = 0; c < s.C; c++){ double v=in[(size_t)m*s.C+c]; ss+=v*v; }
            double inv = ss > 0 ? 1.0/std::sqrt(ss) : 0.0;
            for (int c = 0; c < s.C; c++) md = std::fmax(md, std::fabs((double)got[(size_t)m*s.C+c] - in[(size_t)m*s.C+c]*inv));
        }
        printf("  max_abs=%.6g -> %s\n", md, md < 1e-6 ? "PASS" : "FAIL");
        return md < 1e-6 ? 0 : 1;
    }
    std::vector<unsigned char> in(n), got(n), ref(n);
    for (size_t i = 0; i < n; i++) set_q(in.data(), i, s.in_z + ((int)(i % 15) - 7), s.u);
    rocket_l2norm_q(in.data(), got.data(), s.M, s.C, s.u, s.u, s.in_s, s.in_z, s.out_s, s.out_z);
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv_os = 1.f / s.out_s;
    for (int m = 0; m < s.M; m++) {            /* independent dequant->l2norm->requant oracle */
        float ss = 0; for (int c = 0; c < s.C; c++){ float v=s.in_s*(float)(rocket_qread(in.data(),(size_t)m*s.C+c,s.u)-s.in_z); ss+=v*v; }
        float inv = ss > 0 ? 1.f/std::sqrt(ss) : 0.f;
        for (int c = 0; c < s.C; c++) {
            float v = s.in_s*(float)(rocket_qread(in.data(),(size_t)m*s.C+c,s.u)-s.in_z) * inv;
            long q = (long)lrintf(v*inv_os) + s.out_z; if(q<qmin)q=qmin; if(q>qmax)q=qmax;
            set_q(ref.data(), (size_t)m*s.C+c, (int)q, s.u);
        }
    }
    int md = 0; for (size_t i = 0; i < n; i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.u) - rocket_qread(ref.data(), i, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- TRANSPOSE_CONV (float scatter, vs an independent oracle) ---- */
struct TConvT { int IH, IW, IC, OC, KH, KW, sy, sx, pad_h, pad_w, OH, OW, bias, act; const char *name; };
static int run_tconv(const TConvT &s) {
    printf("%-26s %dx%dx%d->%dx%dx%d K=%dx%d s=%dx%d pad=%d,%d %s act=%d\n", s.name,
           s.IH, s.IW, s.IC, s.OH, s.OW, s.OC, s.KH, s.KW, s.sy, s.sx, s.pad_h, s.pad_w,
           s.bias ? "+bias" : "no-bias", s.act);
    std::vector<float> in((size_t)s.IH*s.IW*s.IC), W((size_t)s.OC*s.KH*s.KW*s.IC),
                       bias((size_t)s.OC), got((size_t)s.OH*s.OW*s.OC), ref((size_t)s.OH*s.OW*s.OC);
    for (size_t i = 0; i < in.size(); i++) in[i] = (float)((int)(i % 7) - 3) * 0.5f;
    for (size_t i = 0; i < W.size(); i++)  W[i]  = (float)((int)(i % 5) - 2) * 0.25f;
    for (int c = 0; c < s.OC; c++) bias[c] = (float)((c % 3) - 1);
    rocket_transpose_conv_f(in.data(), W.data(), s.bias ? bias.data() : nullptr, got.data(),
                            s.IH, s.IW, s.IC, s.OC, s.KH, s.KW, s.sy, s.sx,
                            s.pad_h, s.pad_w, s.OH, s.OW, s.act);
    /* independent scatter oracle (TFLite convention) */
    for (size_t i = 0; i < ref.size(); i++) ref[i] = 0.f;
    for (int ih = 0; ih < s.IH; ih++) for (int iw = 0; iw < s.IW; iw++)
        for (int kh = 0; kh < s.KH; kh++) { int oh = ih*s.sy + kh - s.pad_h; if (oh<0||oh>=s.OH) continue;
            for (int kw = 0; kw < s.KW; kw++) { int ow = iw*s.sx + kw - s.pad_w; if (ow<0||ow>=s.OW) continue;
                for (int oc = 0; oc < s.OC; oc++) for (int ic = 0; ic < s.IC; ic++)
                    ref[((size_t)oh*s.OW+ow)*s.OC+oc] +=
                        in[((size_t)ih*s.IW+iw)*s.IC+ic] *
                        W[(((size_t)oc*s.KH+kh)*s.KW+kw)*s.IC+ic];
            }
        }
    if (s.bias) for (size_t p = 0; p < (size_t)s.OH*s.OW; p++) for (int oc = 0; oc < s.OC; oc++)
        ref[p*s.OC+oc] += bias[oc];
    for (size_t i = 0; i < ref.size(); i++) ref[i] = rocket_apply_act(ref[i], s.act);
    double md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::fmax(md, std::fabs((double)got[i]-(double)ref[i]));
    printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
    return md == 0.0 ? 0 : 1;
}

/* ---- RESIZE (nearest / bilinear, align_corners x half_pixel) ---- */
struct ResizeT { int IH, IW, C, OH, OW, bilinear, ac, hp; const char *name; };
static int run_resize(const ResizeT &s) {
    const size_t in_n = (size_t)s.IH * s.IW * s.C, out_n = (size_t)s.OH * s.OW * s.C;
    printf("%-26s %dx%d->%dx%d C=%d %s ac=%d hp=%d\n", s.name, s.IH, s.IW, s.OH, s.OW, s.C,
           s.bilinear ? "bilinear" : "nearest", s.ac, s.hp);
    std::vector<float> in(in_n), got(out_n), ref(out_n);
    for (size_t i = 0; i < in_n; i++) in[i] = (float)((int)(i % 19) - 9) * 0.5f;
    if (s.bilinear) rocket_resize_bilinear_f(in.data(), got.data(), s.IH, s.IW, s.C, s.OH, s.OW, s.ac, s.hp);
    else            rocket_resize_nearest_f (in.data(), got.data(), s.IH, s.IW, s.C, s.OH, s.OW, s.ac, s.hp);
    /* independent oracle: same TFLite coordinate transform, recomputed inline */
    for (int oh = 0; oh < s.OH; oh++)
        for (int ow = 0; ow < s.OW; ow++)
            for (int c = 0; c < s.C; c++) {
                float v;
                if (!s.bilinear) {
                    int ih = rocket_nn_src(oh, s.IH, s.OH, s.ac, s.hp);
                    int iw = rocket_nn_src(ow, s.IW, s.OW, s.ac, s.hp);
                    v = in[((size_t)ih*s.IW + iw)*s.C + c];
                } else {
                    float fy = rocket_bilin_src(oh, s.IH, s.OH, s.ac, s.hp);
                    if (fy<0) fy=0; if (fy>s.IH-1) fy=s.IH-1;
                    int y0=(int)floorf(fy), y1=y0<s.IH-1?y0+1:y0; float wy=fy-y0;
                    float fx = rocket_bilin_src(ow, s.IW, s.OW, s.ac, s.hp);
                    if (fx<0) fx=0; if (fx>s.IW-1) fx=s.IW-1;
                    int x0=(int)floorf(fx), x1=x0<s.IW-1?x0+1:x0; float wx=fx-x0;
                    float a=in[((size_t)y0*s.IW+x0)*s.C+c], b=in[((size_t)y0*s.IW+x1)*s.C+c];
                    float cc=in[((size_t)y1*s.IW+x0)*s.C+c], d=in[((size_t)y1*s.IW+x1)*s.C+c];
                    float top=a+(b-a)*wx, bot=cc+(d-cc)*wx; v=top+(bot-top)*wy;
                }
                ref[((size_t)oh*s.OW+ow)*s.C+c] = v;
            }
    double md = 0; for (size_t i = 0; i < out_n; i++) md = std::fmax(md, std::fabs((double)got[i]-(double)ref[i]));
    printf("  max_abs=%.6g -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
    return md == 0.0 ? 0 : 1;
}

/* ---- POOL (average / max, SAME|VALID, fused act) ---- */
struct PoolT {
    int IH, IW, C, KH, KW, sy, sx, same, is_avg, act, is_q, u;
    float in_s; int in_z; float out_s; int out_z;
    const char *name;
};
static int run_pool(int fd, const PoolT &s) {
    const int OH = rocket_out_dim(s.IH, s.KH, s.sy, 1, s.same);
    const int OW = rocket_out_dim(s.IW, s.KW, s.sx, 1, s.same);
    printf("%-26s C=%d %dx%d K=%dx%d s=%dx%d %s %s act=%d %s (OH=%d OW=%d)\n",
           s.name, s.C, s.IH, s.IW, s.KH, s.KW, s.sy, s.sx, s.same ? "SAME" : "VALID",
           s.is_avg ? "avg" : "max", s.act, s.is_q ? (s.u ? "u8" : "i8") : "float", OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  bad dims — SKIP\n"); return 0; }
    const int tot_h = rocket_total_pad(s.IH, s.KH, s.sy, 1, OH);
    const int tot_w = rocket_total_pad(s.IW, s.KW, s.sx, 1, OW);
    const int pt = tot_h / 2, pl = tot_w / 2;
    if (!s.is_q) {
        std::vector<float> in((size_t)s.IH * s.IW * s.C), got((size_t)OH * OW * s.C),
                           ref((size_t)OH * OW * s.C);
        for (size_t i = 0; i < in.size(); i++) in[i] = (float)((int)(i % 7) - 3);
        rocket_pool_f(in.data(), got.data(), s.IH, s.IW, s.C, OH, OW, s.KH, s.KW, s.sy, s.sx,
                      pt, pl, s.is_avg, s.act);
        for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) for (int c = 0; c < s.C; c++) {
            float v = s.is_avg ? 0.f : -INFINITY; int cnt = 0;
            for (int kh = 0; kh < s.KH; kh++) { int ih = oh * s.sy + kh - pt; if (ih < 0 || ih >= s.IH) continue;
              for (int kw = 0; kw < s.KW; kw++) { int iw = ow * s.sx + kw - pl; if (iw < 0 || iw >= s.IW) continue;
                float t = in[((size_t)ih * s.IW + iw) * s.C + c]; if (s.is_avg) v += t; else if (t > v) v = t; cnt++; } }
            float r = s.is_avg ? (cnt ? v / (float)cnt : 0.f) : (cnt ? v : 0.f);
            ref[((size_t)oh * OW + ow) * s.C + c] = rocket_apply_act(r, s.act);
        }
        double md = 0; for (size_t i = 0; i < got.size(); i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  host max_abs=%.4f -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        int fail = (md == 0.0) ? 0 : 1;
        // NPU PPU path — exactly what the delegate's eval_pool does under pool_npu: float,
        // and AVERAGE only when VALID (the PPU divides by KH*KW; padded avg diverges from the
        // oracle's valid-count — MAX with pad is fine). Compared to the same independent oracle.
        if (fd >= 0 && (!s.is_avg || s.same == 0)) {
            rocket_pool_desc d;
            d.c = s.C; d.ih = s.IH; d.iw = s.IW; d.kh = s.KH; d.kw = s.KW;
            d.stride_y = s.sy; d.stride_x = s.sx;
            d.pad_top = pt; d.pad_left = pl; d.pad_bottom = tot_h - pt; d.pad_right = tot_w - pl;
            d.method = s.is_avg ? POOL_METHOD_AVG : POOL_METHOD_MAX;
            if (rocket_pool_fp16_plan(&d) == 0) {
                std::vector<_Float16> a((size_t)s.C * s.IH * s.IW), b((size_t)s.C * OH * OW);
                for (int h = 0; h < s.IH; h++) for (int w = 0; w < s.IW; w++) for (int c = 0; c < s.C; c++)
                    a[((size_t)c * s.IH + h) * s.IW + w] = (_Float16)in[((size_t)h * s.IW + w) * s.C + c];
                int r = rocket_pool_fp16(fd, &d, a.data(), b.data());
                if (r != 0) { printf("  NPU PPU: rocket_pool_fp16=%d -> FAIL\n", r); fail = 1; }
                else {
                    double nd = 0;
                    for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) for (int c = 0; c < s.C; c++) {
                        float v = rocket_apply_act((float)b[((size_t)c * OH + oh) * OW + ow], s.act);
                        nd = std::fmax(nd, std::fabs(v - ref[((size_t)oh * OW + ow) * s.C + c]));
                    }
                    const double tol = s.is_avg ? 0.06 : 0.0;
                    printf("  NPU PPU: max_abs=%.4f (tol=%.3f) -> %s\n", nd, tol, nd <= tol ? "PASS" : "FAIL");
                    if (nd > tol) fail = 1;
                }
            }
        }
        return fail;
    }
    std::vector<unsigned char> in((size_t)s.IH * s.IW * s.C), got((size_t)OH * OW * s.C),
                               ref((size_t)OH * OW * s.C);
    for (size_t i = 0; i < in.size(); i++) set_q(in.data(), i, s.in_z + ((int)(i % 5) - 2), s.u);
    rocket_pool_q(in.data(), got.data(), s.IH, s.IW, s.C, OH, OW, s.KH, s.KW, s.sy, s.sx,
                  pt, pl, s.is_avg, s.act, s.u, s.u, s.in_s, s.in_z, s.out_s, s.out_z);
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv = 1.f / s.out_s;
    for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) for (int c = 0; c < s.C; c++) {
        float v = s.is_avg ? 0.f : -INFINITY; int cnt = 0;
        for (int kh = 0; kh < s.KH; kh++) { int ih = oh * s.sy + kh - pt; if (ih < 0 || ih >= s.IH) continue;
          for (int kw = 0; kw < s.KW; kw++) { int iw = ow * s.sx + kw - pl; if (iw < 0 || iw >= s.IW) continue;
            float t = s.in_s * (float)(rocket_qread(in.data(), ((size_t)ih * s.IW + iw) * s.C + c, s.u) - s.in_z);
            if (s.is_avg) v += t; else if (t > v) v = t; cnt++; } }
        float r = s.is_avg ? (cnt ? v / (float)cnt : 0.f) : (cnt ? v : 0.f);
        r = rocket_apply_act(r, s.act);
        long q = (long)lrintf(r * inv) + s.out_z; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), ((size_t)oh * OW + ow) * s.C + c, (int)q, s.u);
    }
    int md = 0; for (size_t i = 0; i < got.size(); i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.u) - rocket_qread(ref.data(), i, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- CONCATENATION (N inputs along axis; requant per input to the output scale) ---- */
struct ConcatT {
    int D0, D1, D2, D3;             /* OUTPUT dims (NHWC, rank 4) */
    int axis, nin, part[4];         /* per-input extent along axis (sum == D[axis]) */
    int is_q, u, act;
    float in_s[4]; int in_z[4]; float out_s; int out_z;
    const char *name;
};
static int run_concat(const ConcatT &s) {
    const int od[4] = { s.D0, s.D1, s.D2, s.D3 };
    int outer = 1; for (int d = 0; d < s.axis; d++) outer *= od[d];
    int inner = 1; for (int d = s.axis + 1; d < 4; d++) inner *= od[d];
    const int out_axis = od[s.axis];
    printf("%-26s out=%dx%dx%dx%d axis=%d nin=%d %s act=%d\n", s.name,
           s.D0, s.D1, s.D2, s.D3, s.axis, s.nin, s.is_q ? (s.u ? "u8" : "i8") : "float", s.act);
    /* reverse map: output axis index -> (input, local index) */
    std::vector<int> which(out_axis), local(out_axis);
    { int a = 0; for (int i = 0; i < s.nin; i++) for (int j = 0; j < s.part[i]; j++) { which[a] = i; local[a] = j; a++; } }

    if (!s.is_q) {
        std::vector<std::vector<float>> ins(s.nin);
        std::vector<float> got((size_t)outer * out_axis * inner), ref(got.size());
        for (int i = 0; i < s.nin; i++) {
            ins[i].resize((size_t)outer * s.part[i] * inner);
            for (size_t k = 0; k < ins[i].size(); k++) ins[i][k] = (float)((int)((k + i) % 7) - 3);
        }
        int off = 0;                              /* kernel: forward scatter */
        for (int i = 0; i < s.nin; i++) {
            rocket_concat_in_f(ins[i].data(), got.data(), outer, s.part[i], out_axis, inner, off, s.act);
            off += s.part[i];
        }
        for (int o = 0; o < outer; o++) for (int a = 0; a < out_axis; a++) for (int k = 0; k < inner; k++) {
            int i = which[a], j = local[a];       /* oracle: output-driven reverse map */
            float v = rocket_apply_act(ins[i][((size_t)o * s.part[i] + j) * inner + k], s.act);
            ref[((size_t)o * out_axis + a) * inner + k] = v;
        }
        double md = 0; for (size_t i = 0; i < got.size(); i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  max_abs=%.4f -> %s\n", md, md == 0.0 ? "PASS" : "FAIL");
        return md == 0.0 ? 0 : 1;
    }
    std::vector<std::vector<unsigned char>> ins(s.nin);
    std::vector<unsigned char> got((size_t)outer * out_axis * inner), ref(got.size());
    for (int i = 0; i < s.nin; i++) {
        ins[i].resize((size_t)outer * s.part[i] * inner);
        for (size_t k = 0; k < ins[i].size(); k++) set_q(ins[i].data(), k, s.in_z[i] + ((int)((k + i) % 3) - 1), s.u);
    }
    int off = 0;
    for (int i = 0; i < s.nin; i++) {
        rocket_concat_in_q(ins[i].data(), got.data(), outer, s.part[i], out_axis, inner, off, s.act,
                           s.u, s.u, s.in_s[i], s.in_z[i], s.out_s, s.out_z);
        off += s.part[i];
    }
    const int qmin = s.u ? 0 : -128, qmax = s.u ? 255 : 127; const float inv = 1.f / s.out_s;
    for (int o = 0; o < outer; o++) for (int a = 0; a < out_axis; a++) for (int k = 0; k < inner; k++) {
        int i = which[a], j = local[a];
        float v = s.in_s[i] * (float)(rocket_qread(ins[i].data(), ((size_t)o * s.part[i] + j) * inner + k, s.u) - s.in_z[i]);
        v = rocket_apply_act(v, s.act);
        long q = (long)lrintf(v * inv) + s.out_z; if (q < qmin) q = qmin; if (q > qmax) q = qmax;
        set_q(ref.data(), ((size_t)o * out_axis + a) * inner + k, (int)q, s.u);
    }
    int md = 0; for (size_t i = 0; i < got.size(); i++)
        md = std::max(md, abs(rocket_qread(got.data(), i, s.u) - rocket_qread(ref.data(), i, s.u)));
    printf("  max|dq|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ---- RESHAPE (shape-only byte copy; no arithmetic to approximate) ---- */
static int run_reshape(int n, int elem_size, const char *name) {
    printf("%-26s n=%d elem=%d\n", name, n, elem_size);
    std::vector<unsigned char> in((size_t)n * elem_size), out((size_t)n * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (unsigned char)((i * 37 + 11) & 0xff);
    rocket_reshape_copy(out.data(), in.data(), (size_t)n, elem_size);
    int md = 0; for (size_t i = 0; i < in.size(); i++) md = std::max(md, abs((int)out[i] - (int)in[i]));
    printf("  max|byte|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ==========================================================================
 * LAYOUT OPS — TRANSPOSE / PAD / SLICE / SPLIT (byte-level, exact -> max|byte|=0).
 * The kernels walk an ODOMETER + precomputed strides; these oracles decode the
 * linear index by integer div/mod into explicit per-axis coordinates — a
 * deliberately DIFFERENT idiom, so a stride/perm/offset bug doesn't hide by
 * appearing in both. Random byte fill exercises every lane; elem_size 4 (float)
 * and 1 (int8/uint8) share one path (these ops never touch values). The end-to-end
 * cross-check vs TFLite's own kernels is tools/run_delegate.py --compare on the RK1.
 * ========================================================================== */
static void decode_coord(long k, int rank, const int *dims, int *coord) {
    for (int d = rank - 1; d >= 0; d--) { coord[d] = (int)(k % dims[d]); k /= dims[d]; }
}
static long encode_coord(int rank, const int *dims, const int *coord) {
    long k = 0; for (int d = 0; d < rank; d++) k = k * dims[d] + coord[d]; return k;
}

static int run_transpose(int rank, const int *in_dims, const int *perm, int elem_size,
                         const char *name) {
    long n = 1; int out_dims[ROCKET_MAX_RANK];
    for (int d = 0; d < rank; d++) { out_dims[d] = in_dims[perm[d]]; n *= in_dims[d]; }
    printf("%-30s rank=%d elem=%d\n", name, rank, elem_size);
    std::vector<uint8_t> in((size_t)n * elem_size), got((size_t)n * elem_size), ref((size_t)n * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)((i * 37 + 11) & 0xff);
    rocket_transpose_bytes(got.data(), in.data(), rank, in_dims, perm, elem_size);
    int oc[ROCKET_MAX_RANK], ic[ROCKET_MAX_RANK];
    for (long k = 0; k < n; k++) {                          /* oracle: oc -> ic[perm[d]]=oc[d] */
        decode_coord(k, rank, out_dims, oc);
        for (int d = 0; d < rank; d++) ic[perm[d]] = oc[d];
        long src = encode_coord(rank, in_dims, ic);
        memcpy(&ref[(size_t)k * elem_size], &in[(size_t)src * elem_size], elem_size);
    }
    int md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::max(md, abs((int)got[i] - (int)ref[i]));
    printf("  max|byte|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

static int run_slice(int rank, const int *in_dims, const int *begin, const int *size,
                     int elem_size, const char *name) {
    long ni = 1, no = 1;
    for (int d = 0; d < rank; d++) { ni *= in_dims[d]; no *= size[d]; }
    printf("%-30s rank=%d elem=%d\n", name, rank, elem_size);
    std::vector<uint8_t> in((size_t)ni * elem_size), got((size_t)no * elem_size), ref((size_t)no * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)((i * 53 + 7) & 0xff);
    rocket_slice_bytes(got.data(), in.data(), rank, in_dims, begin, size, elem_size);
    int oc[ROCKET_MAX_RANK], ic[ROCKET_MAX_RANK];
    for (long k = 0; k < no; k++) {                         /* oracle: out[oc] = in[oc+begin] */
        decode_coord(k, rank, size, oc);
        for (int d = 0; d < rank; d++) ic[d] = oc[d] + begin[d];
        long src = encode_coord(rank, in_dims, ic);
        memcpy(&ref[(size_t)k * elem_size], &in[(size_t)src * elem_size], elem_size);
    }
    int md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::max(md, abs((int)got[i] - (int)ref[i]));
    printf("  max|byte|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* SPLIT = N equal slices along `axis`; verify each output slice via the SLICE kernel
 * (exactly what eval_split does), so this also gates the split lowering. */
static int run_split(int rank, const int *in_dims, int axis, int nsplit, int elem_size,
                     const char *name) {
    printf("%-30s rank=%d axis=%d nsplit=%d elem=%d\n", name, rank, axis, nsplit, elem_size);
    long ni = 1; for (int d = 0; d < rank; d++) ni *= in_dims[d];
    std::vector<uint8_t> in((size_t)ni * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)((i * 29 + 3) & 0xff);
    const int part = in_dims[axis] / nsplit;
    int out_dims[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) out_dims[d] = in_dims[d];
    out_dims[axis] = part;
    long no = 1; for (int d = 0; d < rank; d++) no *= out_dims[d];
    int worst = 0;
    int oc[ROCKET_MAX_RANK], ic[ROCKET_MAX_RANK];
    for (int s = 0; s < nsplit; s++) {
        int begin[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) begin[d] = 0;
        begin[axis] = s * part;
        std::vector<uint8_t> got((size_t)no * elem_size), ref((size_t)no * elem_size);
        rocket_slice_bytes(got.data(), in.data(), rank, in_dims, begin, out_dims, elem_size);
        for (long k = 0; k < no; k++) {
            decode_coord(k, rank, out_dims, oc);
            for (int d = 0; d < rank; d++) ic[d] = oc[d] + begin[d];
            long src = encode_coord(rank, in_dims, ic);
            memcpy(&ref[(size_t)k * elem_size], &in[(size_t)src * elem_size], elem_size);
        }
        int md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::max(md, abs((int)got[i] - (int)ref[i]));
        worst = std::max(worst, md);
    }
    printf("  max|byte|=%d -> %s\n", worst, worst == 0 ? "PASS" : "FAIL");
    return worst == 0 ? 0 : 1;
}

/* STRIDED_SLICE with a positive per-axis stride: out[oc] = in[begin + oc*stride].
 * Gates rocket_strided_slice_bytes against an independent strided-gather oracle. */
static int run_strided_slice(int rank, const int *in_dims, const int *begin,
                             const int *stride, const int *size, int elem_size, const char *name) {
    long ni = 1, no = 1;
    for (int d = 0; d < rank; d++) { ni *= in_dims[d]; no *= size[d]; }
    printf("%-30s rank=%d elem=%d (strided)\n", name, rank, elem_size);
    std::vector<uint8_t> in((size_t)ni * elem_size), got((size_t)no * elem_size), ref((size_t)no * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)((i * 53 + 7) & 0xff);
    rocket_strided_slice_bytes(got.data(), in.data(), rank, in_dims, begin, stride, size, elem_size);
    int oc[ROCKET_MAX_RANK], ic[ROCKET_MAX_RANK];
    for (long k = 0; k < no; k++) {                        /* oracle: out[oc] = in[begin + oc*stride] */
        decode_coord(k, rank, size, oc);
        for (int d = 0; d < rank; d++) ic[d] = begin[d] + oc[d] * stride[d];
        long src = encode_coord(rank, in_dims, ic);
        memcpy(&ref[(size_t)k * elem_size], &in[(size_t)src * elem_size], elem_size);
    }
    int md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::max(md, abs((int)got[i] - (int)ref[i]));
    printf("  max|byte|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* BROADCAST elementwise: the eval_add lowering — materialize each operand to the output
 * shape (rocket_broadcast_copy) then run the same-shape host kernel. Gates the per-channel
 * [C] and scalar [1] broadcasts (the SE-block / bias cases) against NumPy-broadcast oracles. */
static int run_bcast(const char *name) {
    const int N = 2, H = 3, W = 4, C = 5;
    const int out_dims[4] = {N, H, W, C};
    const long no = (long)N * H * W * C;
    int fails = 0;
    printf("%-30s [N,H,W,C]=2,3,4,5\n", name);
    /* (1) per-channel [C] broadcast over [N,H,W,C] (SE-block scale) */
    {
        std::vector<float> c(C), full(no), ref(no);
        for (int i = 0; i < C; i++) c[i] = (float)(i - 2);
        const int cdims[1] = {C};
        rocket_broadcast_copy(c.data(), full.data(), cdims, 1, out_dims, 4, (int)sizeof(float));
        long k = 0; for (int n = 0; n < N; n++) for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) for (int ch = 0; ch < C; ch++) ref[k++] = c[ch];
        double md = 0; for (long i = 0; i < no; i++) md = std::fmax(md, std::fabs(full[i] - ref[i]));
        printf("  per-channel [C]      max_abs=%.4f %s\n", md, md == 0.0 ? "" : "FAIL");
        fails |= (md != 0.0);
    }
    /* (2) scalar [1] broadcast + a full broadcast ADD (x[N,H,W,C] + s[1]) via the eval path */
    {
        std::vector<float> x(no), s(1), xb(no), sb(no), got(no), ref(no);
        for (long i = 0; i < no; i++) x[i] = (float)((i % 7) - 3);
        s[0] = 2.5f;
        const int xdims[4] = {N, H, W, C}, sdims[1] = {1};
        rocket_broadcast_copy(x.data(), xb.data(), xdims, 4, out_dims, 4, (int)sizeof(float));
        rocket_broadcast_copy(s.data(), sb.data(), sdims, 1, out_dims, 4, (int)sizeof(float));
        rocket_add_f(xb.data(), sb.data(), got.data(), (size_t)no, ROCKET_ACT_NONE);
        for (long i = 0; i < no; i++) ref[i] = x[i] + s[0];
        double md = 0; for (long i = 0; i < no; i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
        printf("  scalar [1] + ADD     max_abs=%.4f %s\n", md, md == 0.0 ? "" : "FAIL");
        fails |= (md != 0.0);
    }
    /* (3) int8 per-channel broadcast (byte copy) — quant operands broadcast their bytes */
    {
        std::vector<uint8_t> c(C), full(no), ref(no);
        for (int i = 0; i < C; i++) c[i] = (uint8_t)(i * 17 + 3);
        const int cdims[1] = {C};
        rocket_broadcast_copy(c.data(), full.data(), cdims, 1, out_dims, 4, 1);
        long k = 0; for (int n = 0; n < N; n++) for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) for (int ch = 0; ch < C; ch++) ref[k++] = c[ch];
        int md = 0; for (long i = 0; i < no; i++) md = std::max(md, abs((int)full[i] - (int)ref[i]));
        printf("  per-channel [C] i8   max|byte|=%d %s\n", md, md == 0 ? "" : "FAIL");
        fails |= (md != 0);
    }
    printf("  -> %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

/* PAD: float fill value (0.0f here) for elem_size 4; a zero_point byte for int8.
 * pad_byte is the repeated border byte; for float pass the 4 bytes of the value. */
static int run_pad(int rank, const int *in_dims, const int *pad_before, const int *pad_after,
                   int elem_size, const void *pad_elem, const char *name) {
    int out_dims[ROCKET_MAX_RANK]; long no = 1, ni = 1;
    for (int d = 0; d < rank; d++) {
        out_dims[d] = in_dims[d] + pad_before[d] + pad_after[d];
        no *= out_dims[d]; ni *= in_dims[d];
    }
    printf("%-30s rank=%d elem=%d\n", name, rank, elem_size);
    std::vector<uint8_t> in((size_t)ni * elem_size), got((size_t)no * elem_size), ref((size_t)no * elem_size);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)((i * 41 + 17) & 0xff);
    rocket_pad_bytes(got.data(), in.data(), rank, in_dims, pad_before, out_dims, elem_size, pad_elem);
    for (long k = 0; k < no; k++) memcpy(&ref[(size_t)k * elem_size], pad_elem, elem_size);
    int oc[ROCKET_MAX_RANK];
    for (long k = 0; k < ni; k++) {                        /* oracle: scatter input to interior */
        decode_coord(k, rank, in_dims, oc);
        int o[ROCKET_MAX_RANK]; for (int d = 0; d < rank; d++) o[d] = oc[d] + pad_before[d];
        long dst = encode_coord(rank, out_dims, o);
        memcpy(&ref[(size_t)dst * elem_size], &in[(size_t)k * elem_size], elem_size);
    }
    int md = 0; for (size_t i = 0; i < ref.size(); i++) md = std::max(md, abs((int)got[i] - (int)ref[i]));
    printf("  max|byte|=%d -> %s\n", md, md == 0 ? "PASS" : "FAIL");
    return md == 0 ? 0 : 1;
}

/* ==========================================================================
 * Resident-BO conv context (rocket_conv2d_fp16_ctx): must be BIT-IDENTICAL to the
 * per-call rocket_conv2d_fp16 for every shape. On x86 (fd<0) this checks that the
 * ctx threads correctly through every tiling branch to the same CPU oracle; on the
 * target RK3588 device (fd>=0) it checks the RESIDENT BO reuse + pool growth (ONE ctx driven across a
 * small->large->small sequence + the direct / OC-pad / depthwise / spatial-tile
 * branches) against fresh per-call allocations. The driver owns NCHW<->cube
 * correctness (the shape tests above); here we only assert ctx output == non-ctx.
 * ========================================================================== */
struct CtxShape { int IC, IH, IW, OC, KH, KW, sy, sx, pt, pl, dy, dx, dw; const char *name; };

static int run_ctx_shape(rocket_conv_ctx *ctx, int fd, const CtxShape &s)
{
    rocket_conv2d_desc d = {};
    d.ic = s.IC; d.ih = s.IH; d.iw = s.IW; d.oc = s.OC;
    d.kh = s.KH; d.kw = s.KW; d.stride_y = s.sy; d.stride_x = s.sx;
    d.pad_top = s.pt; d.pad_left = s.pl; d.dil_y = s.dy; d.dil_x = s.dx; d.depthwise = s.dw;
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    printf("%-34s IC=%d %dx%d OC=%d K=%dx%d s=%dx%d dw=%d (OH=%d OW=%d)\n",
           s.name, s.IC, s.IH, s.IW, s.OC, s.KH, s.KW, s.sy, s.sx, s.dw, OH, OW);
    if (OH <= 0 || OW <= 0 || rocket_conv2d_plan(&d)) { printf("  unsupported — SKIP\n"); return 0; }

    const size_t in_n = (size_t)s.IC * s.IH * s.IW;
    const size_t w_n  = s.dw ? (size_t)s.OC * s.KH * s.KW : (size_t)s.OC * s.IC * s.KH * s.KW;
    const size_t o_n  = (size_t)s.OC * OH * OW;
    std::vector<_Float16> in(in_n), w(w_n), o_ref(o_n, (_Float16)7), o_ctx(o_n, (_Float16)9);
    for (size_t i = 0; i < in_n; i++) in[i] = (_Float16)((int)(i % 3) - 1);
    for (size_t i = 0; i < w_n;  i++) w[i]  = (_Float16)((int)(i % 3) - 1);

    int r1 = rocket_conv2d_fp16(fd, &d, in.data(), w.data(), o_ref.data());      // per-call BOs
    int r2 = rocket_conv2d_fp16_ctx(ctx, &d, in.data(), w.data(), o_ctx.data()); // resident pool
    if (r1 || r2) { printf("  conv ret r1=%d r2=%d — FAIL\n", r1, r2); return 1; }
    int differ = memcmp(o_ref.data(), o_ctx.data(), o_n * sizeof(_Float16)) != 0;
    printf("  %s: ctx vs per-call -> %s\n", fd >= 0 ? "HW resident-BO" : "oracle thread",
           differ ? "FAIL (differ)" : "PASS (identical)");
    return differ ? 1 : 0;
}

/* ==========================================================================
 * NCHW-resident inter-op helpers: rocket_nchw_pad + rocket_nchw_bias_act.
 * Proven CONSISTENT with the already-validated NHWC glue (no NPU needed):
 *   - padding an already-NCHW buffer (rocket_nchw_pad) == transposing NHWC straight
 *     into the padded NCHW (rocket_in_nhwc_to_nchw_pad), and
 *   - applying bias+act in NCHW then a plain transpose == the fused NHWC epilogue.
 * memcmp/max=0 then says the resident path carries the same values as the path it
 * replaces (minus the int8 requant, which is the intended precision change).
 * ========================================================================== */
static int run_nchw_pad_test(int C, int IH, int IW, int pt, int pl, const char *name) {
    const int IHp = IH + 2 * pt, IWp = IW + 2 * pl;            // symmetric pad for the check
    std::vector<float> nhwc((size_t)IH * IW * C);
    for (size_t i = 0; i < nhwc.size(); i++) nhwc[i] = (float)((int)(i % 7) - 3);
    std::vector<_Float16> nchw_unpad((size_t)C * IH * IW),
                          via_pad((size_t)C * IHp * IWp), direct((size_t)C * IHp * IWp);
    rocket_in_nhwc_to_nchw_pad(nhwc.data(), nchw_unpad.data(), C, IH, IW, 0, 0, IH, IW);
    rocket_nchw_pad(nchw_unpad.data(), via_pad.data(), C, IH, IW, pt, pl, IHp, IWp);
    rocket_in_nhwc_to_nchw_pad(nhwc.data(), direct.data(), C, IH, IW, pt, pl, IHp, IWp);
    int differ = memcmp(via_pad.data(), direct.data(), via_pad.size() * sizeof(_Float16)) != 0;
    printf("%-26s C=%d %dx%d pad=%d,%d -> %s\n", name, C, IH, IW, pt, pl,
           differ ? "FAIL" : "PASS (identical)");
    return differ ? 1 : 0;
}
static int run_nchw_bias_act_test(int OC, int OH, int OW, int act, const char *name) {
    std::vector<_Float16> a((size_t)OC * OH * OW), b((size_t)OC * OH * OW);
    std::vector<float> bias(OC), got((size_t)OH * OW * OC), ref((size_t)OH * OW * OC);
    for (size_t i = 0; i < a.size(); i++) { a[i] = (_Float16)((int)(i % 9) - 4); b[i] = a[i]; }
    for (int oc = 0; oc < OC; oc++) bias[oc] = (float)((oc % 5) - 2);
    rocket_nchw_bias_act(a.data(), OC, OH, OW, bias.data(), act);                       // A: NCHW epilogue
    rocket_out_nchw_to_nhwc_bias_act(a.data(), got.data(), OC, OH, OW, nullptr, ROCKET_ACT_NONE);
    rocket_out_nchw_to_nhwc_bias_act(b.data(), ref.data(), OC, OH, OW, bias.data(), act); // B: fused NHWC
    double md = 0; for (size_t i = 0; i < got.size(); i++) md = std::fmax(md, std::fabs(got[i] - ref[i]));
    printf("%-26s OC=%d %dx%d act=%d -> %s (max=%.4f)\n", name, OC, OH, OW, act,
           md == 0.0 ? "PASS" : "FAIL", md);
    return md == 0.0 ? 0 : 1;
}

/* ==========================================================================
 * Native int8/uint8 DEPTHWISE host glue (transpose/reorder only):
 *   rocket_in_i8_to_nchw_pad  — NHWC int8 -> NCHW int8 with in_zp halo
 *   rocket_dw_filter_i8_to_chw — TFLite [1,KH,KW,C] -> driver [C,KH,KW]
 *   rocket_out_nchw_to_nhwc_i8 — NCHW int8 -> NHWC int8 (pure transpose)
 * The int8-out DW *compute* is HW-only (replay_dw_mesa / conv_dw_int8 gates), but
 * these pure host transforms are the plumbing the native_dw Eval path rests on; a
 * regression here would silently corrupt the path off-HW. Device-independent
 * (no fd): each transform is checked against an exact integer reference, plus a
 * NHWC->NCHW->NHWC round-trip (the no-pad case the native_dw path actually uses).
 * ========================================================================== */
static int run_dw_i8_glue(int C, int IH, int IW, int KH, int KW, int in_zp,
                          int pad_top, int pad_left, const char *name)
{
    printf("%-30s C=%d %dx%d K=%dx%d zp=%d pad=%d,%d\n",
           name, C, IH, IW, KH, KW, in_zp, pad_top, pad_left);
    int fails = 0;

    /* 1) input NHWC -> NCHW with the in_zp halo materialized. */
    const int IHp = IH + 2 * pad_top, IWp = IW + 2 * pad_left;
    std::vector<signed char> in((size_t)IH * IW * C);
    for (size_t i = 0; i < in.size(); i++) in[i] = (signed char)((int)((i * 31 + 7) % 251) - 125);
    std::vector<int8_t> nchw((size_t)C * IHp * IWp);
    rocket_in_i8_to_nchw_pad(in.data(), nchw.data(), C, IH, IW, in_zp, pad_top, pad_left, IHp, IWp);
    int md_in = 0;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < IHp; y++)
            for (int x = 0; x < IWp; x++) {
                int got = nchw[((size_t)c * IHp + y) * IWp + x];
                int want = (y >= pad_top && y < pad_top + IH && x >= pad_left && x < pad_left + IW)
                    ? (int)in[(((size_t)(y - pad_top)) * IW + (x - pad_left)) * C + c]
                    : (signed char)in_zp;
                md_in = std::max(md_in, abs(got - want));
            }
    printf("  in_i8_to_nchw_pad    max|d|=%d -> %s\n", md_in, md_in == 0 ? "PASS" : "FAIL");
    fails |= (md_in != 0);

    /* 2) DW filter TFLite [1,KH,KW,C] -> driver [C,KH,KW]. */
    std::vector<signed char> flt((size_t)KH * KW * C);
    for (size_t i = 0; i < flt.size(); i++) flt[i] = (signed char)((int)((i * 17 + 3) % 200) - 100);
    std::vector<int8_t> chw((size_t)C * KH * KW);
    rocket_dw_filter_i8_to_chw(flt.data(), chw.data(), C, KH, KW);
    int md_f = 0;
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int got = chw[((size_t)c * KH + kh) * KW + kw];
                int want = flt[((size_t)kh * KW + kw) * C + c];
                md_f = std::max(md_f, abs(got - want));
            }
    printf("  dw_filter_i8_to_chw  max|d|=%d -> %s\n", md_f, md_f == 0 ? "PASS" : "FAIL");
    fails |= (md_f != 0);

    /* 3) output NCHW [C,OH,OW] -> NHWC (pure transpose). */
    const int OH = IH, OW = IW;
    std::vector<int8_t> outnchw((size_t)C * OH * OW);
    for (size_t i = 0; i < outnchw.size(); i++) outnchw[i] = (int8_t)((int)((i * 13 + 5) % 240) - 120);
    std::vector<signed char> nhwc((size_t)OH * OW * C);
    rocket_out_nchw_to_nhwc_i8(outnchw.data(), nhwc.data(), C, OH, OW);
    int md_o = 0;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < OH; y++)
            for (int x = 0; x < OW; x++) {
                int got = nhwc[((size_t)y * OW + x) * C + c];
                int want = outnchw[((size_t)c * OH + y) * OW + x];
                md_o = std::max(md_o, abs(got - want));
            }
    printf("  out_nchw_to_nhwc_i8  max|d|=%d -> %s\n", md_o, md_o == 0 ? "PASS" : "FAIL");
    fails |= (md_o != 0);

    /* 4) round-trip: NHWC --(in, no pad)--> NCHW --(out)--> NHWC recovers the input.
     *    This is the exact no-pad transpose pair the native_dw Eval path uses. */
    std::vector<int8_t> rt_nchw((size_t)C * IH * IW);
    rocket_in_i8_to_nchw_pad(in.data(), rt_nchw.data(), C, IH, IW, in_zp, 0, 0, IH, IW);
    std::vector<signed char> rt_nhwc((size_t)IH * IW * C);
    rocket_out_nchw_to_nhwc_i8(rt_nchw.data(), rt_nhwc.data(), C, IH, IW);
    int md_rt = 0;
    for (size_t i = 0; i < in.size(); i++) md_rt = std::max(md_rt, abs((int)rt_nhwc[i] - (int)in[i]));
    printf("  roundtrip NHWC->NCHW->NHWC max|d|=%d -> %s\n", md_rt, md_rt == 0 ? "PASS" : "FAIL");
    fails |= (md_rt != 0);

    return fails ? 1 : 0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no NPU (%d) — validating the layout/pad/act glue via the "
               "driver's CPU oracle (fd=-1)\n\n", fd);

    const Shape shapes[] = {
        /* name                  IC IH IW  OC  KH KW sy sx dy dx same bias act */
        { 64, 8, 8,   64, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "1x1 pointwise" },
        { 32,10,10,   16, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "3x3 SAME +bias relu" },
        { 32, 8, 8,   32, 3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,   "3x3 s2 SAME (asym pad)" },
        { 48, 9, 9,   16, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU6,  "3x3 VALID +bias relu6" },
        { 32, 9, 9,   16, 3,3, 1,1, 2,2, 1, 0, ROCKET_ACT_NONE,   "3x3 dil2 SAME" },
        { 32, 7, 7,   16, 5,5, 1,1, 1,1, 1, 1, ROCKET_ACT_RELUN1, "5x5 SAME +bias reluN1" },
        { 64, 7,11,   48, 1,5, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "1x5 asym SAME" },
        {  3,16,16,   16, 3,3, 2,2, 1,1, 1, 1, ROCKET_ACT_RELU,   "RGB stem 3x3 s2 +bias relu" },
        { 16, 8, 8,   16, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "1x1 IC=16 (conv path)" },
        { 64,11,11,   32, 3,3, 2,2, 1,1, 0, 0, ROCKET_ACT_NONE,   "3x3 s2 VALID" },
        /* large: forces the driver's internal OH/OC/OW tiling under the glue */
        { 64,64,64,  128, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "64x64->128 3x3 SAME (tiled)" },
        /* OC % 16 != 0 (e.g. an SSD box/class head): the driver pads OC up to 16 and
         * slices the extra channels off the output. OC=24 -> pad 32; OC=40 -> pad 48. */
        { 32, 8, 8,   24, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,   "1x1 OC=24 (OC pad)" },
        { 16,10,10,   40, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "3x3 SAME OC=40 +bias (OC pad)" },
    };

    int fail = 0;
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
        fail |= run_shape(fd, shapes[i]);
        printf("\n");
    }

    printf("---- quantized (int8 / uint8) CONV_2D ----\n\n");
    const QShape qshapes[] = {
        /* IC IH IW  OC  KH KW sy sx dy dx same bias act  u8  in_s in_zp out_s out_zp w_zp name */
        { 32, 8, 8,  16, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  0, 1.f,  0, 1.f,  0, 0,
          "i8 1x1 pointwise" },
        { 32,10,10,  16, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, 1.f, -3, 1.f,  5, 0,
          "i8 3x3 SAME +bias relu zp" },
        { 16, 8, 8,  16, 3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,  0, 2.f,  7, 1.f,  0, 0,
          "i8 3x3 s2 SAME (asym pad)" },
        { 32, 9, 9,  16, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU6, 0, 1.f,  0, 0.5f, 0, 0,
          "i8 3x3 VALID +bias relu6" },
        { 16, 9, 9,  16, 3,3, 1,1, 2,2, 1, 0, ROCKET_ACT_NONE,  0, 1.f,  0, 2.f,  0, 0,
          "i8 3x3 dil2 SAME (per-axis w)" },
        {  3,16,16,  16, 3,3, 2,2, 1,1, 1, 1, ROCKET_ACT_RELU,  0, 1.f,  0, 1.f,  0, 0,
          "i8 RGB stem 3x3 s2 +bias" },
        { 32, 8, 8,  16, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU6, 1, 1.f,128, 1.f,128,128,
          "u8 1x1 +bias relu6 zp128" },
        { 16,10,10,  16, 3,3, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  1, 1.f,128, 2.f,128,128,
          "u8 3x3 SAME zp128" },
        { 32, 8, 8,  24, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, 1.f,  0, 1.f,  0, 0,
          "i8 1x1 OC=24 +bias (OC pad)" },
    };
    for (size_t i = 0; i < sizeof(qshapes)/sizeof(qshapes[0]); i++) {
        fail |= run_q_shape(fd, qshapes[i]);
        printf("\n");
    }

    printf("---- NATIVE int8 DIRECT CONV_2D (exact int32 accumulate) ----\n\n");
    /* int8 (signed) only; scales chosen so outputs land mostly inside int8 (in/out 0.5,
     * per-axis w 1.0/2.0). Mirrors the quant-conv shape coverage: 1x1, 3x3 SAME +bias zp,
     * stride-2 asym pad, VALID +bias relu6, dil2 (per-axis w), RGB stem, OC%32 pad. */
    const QShape ni8shapes[] = {
        /* IC IH IW  OC  KH KW sy sx dy dx same bias act  u8  in_s in_zp out_s out_zp w_zp name */
        { 32, 8, 8,  32, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  0, .5f,  0, .5f,  0, 0,
          "ni8 1x1 pointwise" },
        { 32,10,10,  32, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, .5f, -3, .5f,  5, 0,
          "ni8 3x3 SAME +bias relu zp" },
        { 16, 8, 8,  32, 3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,  0, .5f,  7, .5f,  0, 0,
          "ni8 3x3 s2 SAME (asym pad)" },
        { 32, 9, 9,  32, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU6, 0, .5f,  0, .25f, 0, 0,
          "ni8 3x3 VALID +bias relu6" },
        { 16, 9, 9,  32, 3,3, 1,1, 2,2, 1, 0, ROCKET_ACT_NONE,  0, .5f,  4, .5f,  0, 0,
          "ni8 3x3 dil2 SAME (per-axis w)" },
        {  3,16,16,  32, 3,3, 2,2, 1,1, 1, 1, ROCKET_ACT_RELU,  0, .5f,  2, .5f,  0, 0,
          "ni8 RGB stem 3x3 s2 +bias" },
        { 32, 8, 8,  48, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  0, .5f, -3, .5f,  0, 0,
          "ni8 1x1 OC=48 +bias (OC%32 pad)" },
        { 64,16,16,  64, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, .25f, 1, .5f,  0, 0,
          "ni8 3x3 SAME C=64 16x16 +bias" },
        /* large-feature-cube regression gates for large-IC int8 tiling (measured 2026-06-21);
         * the smaller IC<=64 cases above don't exercise these tiling paths:
         *   1x1 IC=768 OC=96 20x20  forces OH-band tiling + a short remainder;
         *   3x3 IC=768 OC=64 18x18  forces a >4-bank weight cube vs a shrunk feature;
         *   1x1 IC=256 OC=546 3x3   is a small SSD head, OH<4 -> the datain_height floor pad. */
        { 768,20,20,  96, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  0, .25f, 3, 8.f,  0, 0,
          "ni8 1x1 IC768 OC96 20x20 (tile+remainder)" },
        { 768,18,18,  64, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU,  0, .25f, 1, 32.f, 0, 0,
          "ni8 3x3 IC768 OC64 18x18 VALID (weight banks)" },
        { 256, 3, 3, 546, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  0, .25f, 0, 8.f,  0, 0,
          "ni8 1x1 IC256 OC546 3x3 (small-head floor pad)" },
    };
    for (size_t i = 0; i < sizeof(ni8shapes)/sizeof(ni8shapes[0]); i++) {
        fail |= run_ni8_shape(fd, ni8shapes[i]);
        printf("\n");
    }

    printf("---- NATIVE uint8 DIRECT CONV_2D (uint8 recenter path: recenter + box-sum) ----\n\n");
    /* uint8 with ASYMMETRIC weight zero-points (the real coral/MediaPipe case: MobileDet
     * conv weights span w_zp 91..177). in_zp varied incl. 0 (post-depthwise). One symmetric
     * w_zp=128 shape exercises the box-sum-skipped branch. Same shape coverage as ni8. */
    const QShape nu8shapes[] = {
        /* IC IH IW  OC  KH KW sy sx dy dx same bias act  u8  in_s in_zp out_s out_zp w_zp name */
        { 32, 8, 8,  32, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  1, .5f,128, .5f,128,110,
          "nu8 1x1 pointwise" },
        { 32,10,10,  32, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  1, .5f,120, .5f,130,140,
          "nu8 3x3 SAME +bias relu" },
        { 16, 8, 8,  32, 3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,  1, .5f,  0, .5f,127,100,
          "nu8 3x3 s2 SAME (in_zp=0)" },
        { 32, 9, 9,  32, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU6, 1, .5f,128, .25f,0, 128,
          "nu8 3x3 VALID +bias relu6 (sym)" },
        { 16, 9, 9,  32, 3,3, 1,1, 2,2, 1, 0, ROCKET_ACT_NONE,  1, .5f,132, .5f,124,126,
          "nu8 3x3 dil2 SAME (per-axis w)" },
        {  3,16,16,  32, 3,3, 2,2, 1,1, 1, 1, ROCKET_ACT_RELU,  1, .5f,128, .5f,118,119,
          "nu8 RGB stem 3x3 s2 +bias" },
        { 32, 8, 8,  48, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  1, .5f,147, .5f,140,135,
          "nu8 1x1 OC=48 +bias (OC%32 pad)" },
        { 64,16,16,  64, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  1, .25f,125,.5f,131,131,
          "nu8 3x3 SAME C=64 16x16 +bias" },
        { 48,16,16,  32, 5,5, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  1, .25f,122,.5f,128,121,
          "nu8 5x5 SAME C=48 16x16 +bias (separable box-sum)" },
        { 32,17,17,  32, 7,7, 1,1, 1,1, 0, 0, ROCKET_ACT_NONE,  1, .5f,119, .5f,127,133,
          "nu8 7x7 VALID C=32 17x17 (separable box-sum, odd OW)" },
        /* uint8 (asymmetric w_zp) large-feature-cube regression gates — same three shapes
         * as ni8 (the recenter+box-sum path through the same fixed tiler/floor-pad). */
        { 768,20,20,  96, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  1, .25f,130, 8.f,128,135,
          "nu8 1x1 IC768 OC96 20x20 (tile+remainder)" },
        { 768,18,18,  64, 3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU,  1, .25f,120, 32.f,118,119,
          "nu8 3x3 IC768 OC64 18x18 VALID (weight banks)" },
        { 256, 3, 3, 546, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  1, .25f,128, 8.f,140,135,
          "nu8 1x1 IC256 OC546 3x3 (small-head floor pad)" },
    };
    for (size_t i = 0; i < sizeof(nu8shapes)/sizeof(nu8shapes[0]); i++) {
        fail |= run_nu8_shape(fd, nu8shapes[i]);
        printf("\n");
    }

    printf("---- NATIVE int8/uint8 1x1 via MATMUL (perf Step 1: recenter+box-sum, K/N pad, MN requant) ----\n\n");
    /* 1x1 stride-1 (M=IH*IW). Covers: no pad (IC/OC%32==0), K-pad, N-pad, both; uint8
     * asymmetric (box-sum) + symmetric (box skipped) + in_zp=0; signed int8; M==1 (GEMV);
     * with/without bias; the activations. Validates the MN-requant + recenter + zero-pad glue
     * against the same exact-int oracle the conv path uses (a host matmul stand-in; the NPU
     * matmul is HW-gated by the end-to-end matmul-on/off byte-identical run). */
    const QShape mm1x1shapes[] = {
        /* IC IH IW  OC  KH KW sy sx dy dx same bias act  u8  in_s in_zp out_s out_zp w_zp name */
        { 32, 8, 8,  32, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU6, 1, .5f,128, .5f,128,110,
          "mm u8 1x1 IC32 OC32 (no pad)" },
        { 48,10,10,  24, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  1, .5f,120, .5f,130,140,
          "mm u8 1x1 IC48 OC24 (K+N pad)" },
        { 40, 8, 8,  64, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  1, .5f,  0, .5f,127,100,
          "mm u8 1x1 IC40 OC64 in_zp0 (K pad)" },
        { 64, 4, 4,  96, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  1, .5f,128, .25f, 0,128,
          "mm u8 1x1 IC64 OC96 sym (no box)" },
        {128, 1, 1, 546, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  1, .5f,147, .5f,140,135,
          "mm u8 1x1 IC128 OC546 M=1 (N pad)" },
        { 32, 8, 8,  32, 1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  0, .5f,  0, .5f,  0, 0,
          "mm i8 1x1 IC32 OC32 (no pad)" },
        { 48, 8, 8,  48, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, .5f, -3, .5f,  0, 0,
          "mm i8 1x1 IC48 OC48 (K+N pad)" },
        { 64,10,10,  24, 1,1, 1,1, 1,1, 1, 1, ROCKET_ACT_NONE,  0, .5f,  4, .5f,  0, 0,
          "mm i8 1x1 IC64 OC24 (N pad)" },
    };
    for (size_t i = 0; i < sizeof(mm1x1shapes)/sizeof(mm1x1shapes[0]); i++) {
        fail |= run_mm1x1_shape(fd, mm1x1shapes[i]);
        printf("\n");
    }

    printf("---- depthwise (float) DEPTHWISE_CONV_2D ----\n\n");
    const DWShape dwshapes[] = {
        /* C  IH IW  KH KW sy sx dy dx same bias act               name */
        { 32, 8, 8,  3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "dw 3x3 SAME +bias relu" },
        { 64, 8, 8,  3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,   "dw 3x3 s2 SAME (asym pad)" },
        { 32, 9, 9,  5,5, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU6,  "dw 5x5 SAME +bias relu6" },
        { 64, 9, 9,  3,3, 1,1, 1,1, 0, 1, ROCKET_ACT_RELUN1, "dw 3x3 VALID +bias reluN1" },
        { 32, 9, 9,  3,3, 1,1, 2,2, 1, 0, ROCKET_ACT_NONE,   "dw 3x3 dil2 SAME" },
        { 64, 8, 8,  1,1, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "dw 1x1 (per-channel scale)" },
        { 96, 7,11,  3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "dw 3x3 SAME C=96 +bias" },
        /* channel tiling: whole layer overflows one CBUF pass -> split over channels.
         * 64x64 overflows on FEATURE bytes (tiled by G=32); C=2048 overflows on the
         * WEIGHT cube (tiled to ~1792-channel chunks). Each chunk is an independent DW
         * job; the off-HW oracle proves the channel decomposition is bit-exact. */
        { 64,64,64,  3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,   "dw 3x3 SAME C=64 64x64 (feat tile)" },
        {2048, 8, 8, 3,3, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "dw 3x3 SAME C=2048 (weight tile)" },
        /* the MobileNetV2 block depthwise (C=192 32x32): channel-tiled into 8-bank chunks
         * (exercises the DW feature-budget bound that an unbounded 12-bank feature exceeds). */
        { 192,32,32, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU6,  "dw 3x3 SAME C=192 32x32 (mbv2 block)" },
        /* single-channel SPATIAL overflow: one channel's feature alone exceeds the budget,
         * so channel tiling can't help -> dw_spatial bands it over output rows/cols. */
        { 32,96,96,  3,3, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,   "dw 3x3 SAME C=32 96x96 (spatial tile)" },
    };
    for (size_t i = 0; i < sizeof(dwshapes)/sizeof(dwshapes[0]); i++) {
        fail |= run_dw_shape(fd, dwshapes[i]);
        printf("\n");
    }

    printf("---- depthwise (int8 / uint8) DEPTHWISE_CONV_2D ----\n\n");
    const DWQShape dwqshapes[] = {
        /* C  IH IW  KH KW sy sx dy dx same bias act  u8  in_s in_zp out_s out_zp w_zp name */
        { 32,10,10,  3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, 1.f, -3, 1.f,  5, 0,
          "i8 dw 3x3 SAME +bias relu zp" },
        { 64, 8, 8,  3,3, 2,2, 1,1, 1, 0, ROCKET_ACT_NONE,  0, 2.f,  7, 1.f,  0, 0,
          "i8 dw 3x3 s2 SAME (asym pad)" },
        { 32, 9, 9,  5,5, 1,1, 1,1, 0, 1, ROCKET_ACT_RELU6, 0, 1.f,  0, 0.5f, 0, 0,
          "i8 dw 5x5 VALID +bias (per-axis w)" },
        { 32,10,10,  3,3, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  1, 1.f,128, 2.f,128,128,
          "u8 dw 3x3 SAME zp128" },
        { 64,64,64,  3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU,  0, 1.f,  0, 1.f,  0, 0,
          "i8 dw 3x3 SAME C=64 64x64 (feat tile)" },
        /* the actual MobileNetV2-block depthwise that failed on HW (int8, C=192 32x32). */
        { 192,32,32, 3,3, 1,1, 1,1, 1, 1, ROCKET_ACT_RELU6, 0, 1.f, -3, 1.f,  5, 0,
          "i8 dw 3x3 SAME C=192 32x32 (mbv2)" },
        /* single-channel spatial overflow (int8). */
        { 32,96,96,  3,3, 1,1, 1,1, 1, 0, ROCKET_ACT_NONE,  0, 1.f,  0, 1.f,  0, 0,
          "i8 dw 3x3 SAME C=32 96x96 (spatial)" },
    };
    for (size_t i = 0; i < sizeof(dwqshapes)/sizeof(dwqshapes[0]); i++) {
        fail |= run_dw_q_shape(fd, dwqshapes[i]);
        printf("\n");
    }

    /* The dwqshapes above exercise the dequant->fp16-DW->requant path; the native
     * int8-out DW glue (the transpose/reorder the native_dw Eval path uses) has no
     * software compute oracle, so unit-test its pure host transforms directly. */
    printf("---- native DW int8/uint8 host glue (transpose/reorder, device-independent) ----\n\n");
    fail |= run_dw_i8_glue(32,  8, 8, 3,3,  -3, 1, 1, "glue C=32 3x3 SAME pad zp-3");  printf("\n");
    fail |= run_dw_i8_glue(64, 10, 6, 5,5,   0, 0, 0, "glue C=64 5x5 nopad zp0");      printf("\n");
    fail |= run_dw_i8_glue(16,  7, 9, 3,3, 128, 2, 1, "glue C=16 u8-domain zp128 pad"); printf("\n");

    /* ---- resident-BO conv context: rocket_conv2d_fp16_ctx vs the per-call path ----
     * ONE ctx driven across the sequence below exercises pool reuse + growth
     * (small -> large -> small) and threads through every branch (direct fast-path,
     * OC-tiled, OC%16 pad, depthwise channel-tile, dw_spatial). Each must match the
     * per-call path bit-for-bit. On HW (fd>=0) this is the resident-BO gate. */
    printf("---- resident-BO conv ctx (rocket_conv2d_fp16_ctx == per-call) ----\n\n");
    const CtxShape ctxshapes[] = {
        /* IC  IH IW  OC  KH KW sy sx pt pl dy dx dw  name */
        {  64, 8, 8,  64, 1,1, 1,1, 0,0, 1,1, 0, "ctx 1x1 small (direct)" },
        {  32,10,10,  16, 3,3, 1,1, 1,1, 1,1, 0, "ctx 3x3 SAME (direct)" },
        {  64,64,64, 128, 3,3, 1,1, 1,1, 1,1, 0, "ctx 64x64->128 (grows pool, OC+OH tiled)" },
        {  64, 8, 8,  64, 1,1, 1,1, 0,0, 1,1, 0, "ctx 1x1 small again (reuse big pool)" },
        {  32, 8, 8,  24, 1,1, 1,1, 0,0, 1,1, 0, "ctx OC=24 (OC%16 pad recursion)" },
        {  32, 8, 8,  32, 3,3, 1,1, 1,1, 1,1, 1, "ctx dw 3x3 (channel)" },
        { 192,32,32, 192, 3,3, 1,1, 1,1, 1,1, 1, "ctx dw C=192 32x32 (channel-tiled)" },
        {  32,96,96,  32, 3,3, 1,1, 1,1, 1,1, 1, "ctx dw C=32 96x96 (spatial-tiled)" },
    };
    {
        rocket_conv_ctx *cctx = rocket_conv_ctx_create(fd);
        if (!cctx) { printf("  rocket_conv_ctx_create failed — FAIL\n"); fail |= 1; }
        else {
            for (size_t i = 0; i < sizeof(ctxshapes)/sizeof(ctxshapes[0]); i++) {
                fail |= run_ctx_shape(cctx, fd, ctxshapes[i]);
                printf("\n");
            }
            rocket_conv_ctx_free(cctx);
        }
    }

    /* ---- aux host ops: ACT / ADD / POOL / CONCAT / RESHAPE (CPU, device-independent) ---- */
    printf("---- aux host ops (ACT / ADD / POOL / CONCAT / RESHAPE) ----\n\n");

    const ActT acts[] = {
        /* H  W  C   kind                    q  u  in(s,z)   out(s,z)  name */
        { 8, 8,16, ROCKET_UNARY_HARDSWISH,   0, 0, 1.f,0,   1.f,0,   "hardswish float" },
        { 6, 6,16, ROCKET_UNARY_SIGMOID,     0, 0, 1.f,0,   1.f,0,   "sigmoid float" },
        { 5, 5,16, ROCKET_UNARY_HARDSIGMOID, 0, 0, 1.f,0,   1.f,0,   "hardsigmoid float" },
        { 7, 7,16, ROCKET_UNARY_TANH,        0, 0, 1.f,0,   1.f,0,   "tanh float" },
        { 6, 6,16, ROCKET_UNARY_TANH,        1, 0, .25f,4,  0.0078125f,0, "i8 tanh (out [-1,1])" },
        /* int8: hardswish out spans +/- (signed out); LOGISTIC out in [0,1] (scale 1/256, zp -128). */
        { 8, 8,16, ROCKET_UNARY_HARDSWISH,   1, 0, .25f,-3, .25f,-5, "i8 hardswish +zp" },
        { 6, 6,16, ROCKET_UNARY_SIGMOID,     1, 0, .25f, 4, 0.00390625f,-128, "i8 sigmoid (logistic out)" },
        /* uint8 (coral/MediaPipe): hardswish + sigmoid, zp128 / asymmetric. */
        { 8, 8,16, ROCKET_UNARY_HARDSWISH,   1, 1, .25f,128,.25f,128, "u8 hardswish zp128" },
        { 6, 6,16, ROCKET_UNARY_SIGMOID,     1, 1, .25f,120,0.00390625f,0, "u8 sigmoid (logistic out)" },
        /* ELU (TFLite Elu, alpha=1; signed output spans +/-). */
        { 7, 7,16, ROCKET_UNARY_ELU,         0, 0, 1.f,0,   1.f,0,   "elu float" },
        { 6, 6,16, ROCKET_UNARY_ELU,         1, 0, .25f,-3, .25f,-5, "i8 elu +zp" },
        { 6, 6,16, ROCKET_UNARY_ELU,         1, 1, .25f,128,.25f,128,"u8 elu zp128" },
        { 7, 7,16, ROCKET_UNARY_LOG,         0, 0, 1.f,0,   1.f,0,   "log float (x>0)" },
        /* Exact-arithmetic siblings: RELU / RELU6 / RELU_N1_TO_1 / NEG / SQUARE / FLOOR / ABS
         * are exact float ops (max_abs must be 0). EXP / SQRT / RSQRT / LEAKY_RELU likewise
         * (libm float matches the oracle exactly). Each gets a float + an int8 + a uint8 case. */
        { 8, 8,16, ROCKET_UNARY_RELU,        0, 0, 1.f,0,   1.f,0,   "relu float" },
        { 6, 6,16, ROCKET_UNARY_RELU,        1, 0, .25f,-3, .25f,-3, "i8 relu" },
        { 6, 6,16, ROCKET_UNARY_RELU,        1, 1, .25f,128,.25f,128,"u8 relu zp128" },
        { 8, 8,16, ROCKET_UNARY_RELU6,       0, 0, 1.f,0,   1.f,0,   "relu6 float" },
        { 6, 6,16, ROCKET_UNARY_RELU6,       1, 0, .0625f,-128,.0625f,-128,"i8 relu6 (out [0,6])" },
        { 6, 6,16, ROCKET_UNARY_RELU6,       1, 1, .0625f,0,.0625f,0, "u8 relu6 (out [0,6])" },
        { 8, 8,16, ROCKET_UNARY_RELU_N1_1,   0, 0, 1.f,0,   1.f,0,   "relu_n1_1 float" },
        { 6, 6,16, ROCKET_UNARY_RELU_N1_1,   1, 0, .25f,0,  0.0078125f,0, "i8 relu_n1_1 (out [-1,1])" },
        { 8, 8,16, ROCKET_UNARY_LEAKY_RELU,  0, 0, 1.f,0,   1.f,0,   "leaky_relu a=0.1 float", 0.1f },
        { 6, 6,16, ROCKET_UNARY_LEAKY_RELU,  1, 0, .25f,-3, .25f,-3, "i8 leaky_relu a=0.2", 0.2f },
        { 6, 6,16, ROCKET_UNARY_LEAKY_RELU,  1, 1, .25f,128,.25f,128,"u8 leaky_relu a=0.1", 0.1f },
        { 7, 7,16, ROCKET_UNARY_EXP,         0, 0, 1.f,0,   1.f,0,   "exp float" },
        { 6, 6,16, ROCKET_UNARY_EXP,         1, 0, .125f,0, .25f,-128,"i8 exp (out>0)" },
        { 7, 7,16, ROCKET_UNARY_SQRT,        0, 0, 1.f,0,   1.f,0,   "sqrt float (x>0)" },
        { 6, 6,16, ROCKET_UNARY_SQRT,        1, 1, .25f,0,  .03125f,0,"u8 sqrt (x>0)" },
        { 7, 7,16, ROCKET_UNARY_RSQRT,       0, 0, 1.f,0,   1.f,0,   "rsqrt float (x>0)" },
        { 8, 8,16, ROCKET_UNARY_ABS,         0, 0, 1.f,0,   1.f,0,   "abs float" },
        { 6, 6,16, ROCKET_UNARY_ABS,         1, 0, .25f,0,  .25f,-128,"i8 abs (out>=0)" },
        { 8, 8,16, ROCKET_UNARY_NEG,         0, 0, 1.f,0,   1.f,0,   "neg float" },
        { 6, 6,16, ROCKET_UNARY_NEG,         1, 0, .25f,-3, .25f,3,  "i8 neg" },
        { 8, 8,16, ROCKET_UNARY_SQUARE,      0, 0, 1.f,0,   1.f,0,   "square float" },
        { 6, 6,16, ROCKET_UNARY_SQUARE,      1, 1, .125f,0, .25f,0,  "u8 square (out>=0)" },
        { 8, 8,16, ROCKET_UNARY_FLOOR,       0, 0, 1.f,0,   1.f,0,   "floor float" },
    };
    for (size_t i = 0; i < sizeof(acts)/sizeof(acts[0]); i++) { fail |= run_act(acts[i]); printf("\n"); }

    const BinT bins[] = {
        /* H  W  C  op                q  au bu ou  in0(s,z)  in1(s,z)  out(s,z)  name */
        { 8, 8,16, ROCKET_BINOP_MAX,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "max float" },
        { 6, 6,16, ROCKET_BINOP_MIN,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "min float" },
        { 8, 8,16, ROCKET_BINOP_MAX,  1, 0,0,0, 1.f,0, 2.f,0, 1.f,0, "i8 max (diff scales)" },
        { 5, 5,16, ROCKET_BINOP_MIN,  1, 0,0,0, 1.f,-3,1.f,2, 1.f,5, "i8 min +zp" },
        { 6, 6,16, ROCKET_BINOP_MAX,  1, 1,1,1, 1.f,128,2.f,128,1.f,128,"u8 max zp128" },
    };
    for (size_t i = 0; i < sizeof(bins)/sizeof(bins[0]); i++) { fail |= run_binary(bins[i]); printf("\n"); }

    const ArithT ariths[] = {
        /* H  W  C  op                act              q  au bu ou  in0(s,z) in1(s,z) out(s,z) name */
        { 8, 8,16, ROCKET_ARITH_MUL, ROCKET_ACT_NONE,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "mul float" },
        { 6, 6,16, ROCKET_ARITH_SUB, ROCKET_ACT_NONE,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "sub float" },
        { 6, 6,16, ROCKET_ARITH_DIV, ROCKET_ACT_NONE,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "div float" },
        { 5, 5,16, ROCKET_ARITH_MUL, ROCKET_ACT_RELU,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "mul float relu" },
        { 8, 8,16, ROCKET_ARITH_SUB, ROCKET_ACT_NONE,  1, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "i8 sub" },
        { 8, 8,16, ROCKET_ARITH_MUL, ROCKET_ACT_NONE,  1, 0,0,0, 1.f,-2,2.f,0, 4.f,0, "i8 mul (diff scales)" },
        { 6, 6,16, ROCKET_ARITH_DIV, ROCKET_ACT_NONE,  1, 1,1,1, 1.f,128,1.f,128,1.f,128,"u8 div zp128" },
    };
    for (size_t i = 0; i < sizeof(ariths)/sizeof(ariths[0]); i++) { fail |= run_arith(ariths[i]); printf("\n"); }

    const PreluT prelus[] = {
        /* H  W  C   q  u  in(s,z)   out(s,z)  name */
        { 8, 8,32, 0, 0, 1.f,0,   1.f,0,   "prelu float" },
        { 6, 6,32, 1, 0, .25f,-3, .25f,-5, "i8 prelu +zp" },
        { 6, 6,32, 1, 1, .25f,128,.25f,128,"u8 prelu zp128" },
    };
    for (size_t i = 0; i < sizeof(prelus)/sizeof(prelus[0]); i++) { fail |= run_prelu(prelus[i]); printf("\n"); }

    const ReduceT reduces[] = {
        /* H  W  C   op                 q  u  in(s,z)   out(s,z)  name */
        { 7, 7,32, ROCKET_REDUCE_MEAN, 0, 0, 1.f,0,   1.f,0,   "reduce mean float (GAP)" },
        {14,14,16, ROCKET_REDUCE_MAX,  0, 0, 1.f,0,   1.f,0,   "reduce max float" },
        { 8, 8,32, ROCKET_REDUCE_MIN,  0, 0, 1.f,0,   1.f,0,   "reduce min float" },
        { 7, 7,32, ROCKET_REDUCE_MEAN, 1, 0, .25f,-3, .5f,5,   "i8 reduce mean +zp" },
        { 7, 7,32, ROCKET_REDUCE_MAX,  1, 1, .25f,128,.25f,128,"u8 reduce max zp128" },
    };
    for (size_t i = 0; i < sizeof(reduces)/sizeof(reduces[0]); i++) { fail |= run_reduce(reduces[i]); printf("\n"); }

    const ResizeT resizes[] = {
        /* IH IW  C  OH OW bil ac hp  name */
        {  8, 8,32,16,16, 0, 0, 0, "nearest 2x asym" },
        {  8, 8,32,16,16, 0, 0, 1, "nearest 2x half-pixel" },
        {  7, 7,16,14,14, 1, 0, 1, "bilinear 2x half-pixel" },
        {  7, 7,16,13,13, 1, 1, 0, "bilinear align_corners" },
        {  8, 8,32,24,24, 1, 0, 1, "bilinear 3x half-pixel" },
    };
    for (size_t i = 0; i < sizeof(resizes)/sizeof(resizes[0]); i++) { fail |= run_resize(resizes[i]); printf("\n"); }

    const TConvT tconvs[] = {
        /* IH IW IC OC KH KW sy sx ph pw OH OW bias act               name */
        {  8, 8,16,32, 2,2, 2,2, 0,0,16,16, 1, ROCKET_ACT_NONE,  "tconv 2x2 s2 VALID +bias" },
        {  8, 8,32,16, 3,3, 2,2, 1,1,16,16, 0, ROCKET_ACT_RELU,  "tconv 3x3 s2 SAME relu" },
        {  6, 6,16,16, 4,4, 2,2, 1,1,12,12, 1, ROCKET_ACT_NONE,  "tconv 4x4 s2 +bias" },
        {  5, 5, 8,24, 3,3, 1,1, 0,0, 7, 7, 1, ROCKET_ACT_NONE,  "tconv 3x3 s1 VALID +bias" },
    };
    for (size_t i = 0; i < sizeof(tconvs)/sizeof(tconvs[0]); i++) { fail |= run_tconv(tconvs[i]); printf("\n"); }

    const L2T l2norms[] = {
        /* M    C   q  u  in(s,z)   out(s,z)        name */
        { 64,  32, 0, 0, 1.f,0,   1.f,0,          "l2norm float [HW,C]" },
        {  1, 256, 0, 0, 1.f,0,   1.f,0,          "l2norm float row" },
        { 49,  32, 1, 0, .25f,-3, 0.0078125f,0,   "i8 l2norm (out 1/128)" },
        { 49,  32, 1, 1, .25f,128,0.0078125f,128, "u8 l2norm zp128" },
    };
    for (size_t i = 0; i < sizeof(l2norms)/sizeof(l2norms[0]); i++) { fail |= run_l2norm(l2norms[i]); printf("\n"); }

    const LSMT lsms[] = {
        { 16, 32, "logsoftmax [16,32]" },
        {  1, 91, "logsoftmax det-classes row" },
    };
    for (size_t i = 0; i < sizeof(lsms)/sizeof(lsms[0]); i++) { fail |= run_logsoftmax(lsms[i]); printf("\n"); }

    const SMT sms[] = {
        { 16, 32, 1.0f, "softmax [16,32]" },
        {  1, 91, 1.0f, "softmax det-classes row" },
        {  8, 10, 2.0f, "softmax beta=2" },
    };
    for (size_t i = 0; i < sizeof(sms)/sizeof(sms[0]); i++) { fail |= run_softmax(sms[i]); printf("\n"); }

    const CSMT csms[] = {
        { 8, 16, 0, 0, "cumsum incl fwd" },
        { 8, 16, 1, 0, "cumsum excl fwd" },
        { 8, 16, 0, 1, "cumsum incl rev" },
        { 8, 16, 1, 1, "cumsum excl rev" },
    };
    for (size_t i = 0; i < sizeof(csms)/sizeof(csms[0]); i++) { fail |= run_cumsum(csms[i]); printf("\n"); }

    const FCT fcs[] = {
        /* M   K    N    act               bias  name */
        {  1, 1024, 1001, ROCKET_ACT_NONE,  1, "fc classifier 1x1024->1001 (N,K pad)" },
        {  1,  256,   90, ROCKET_ACT_NONE,  1, "fc head 1x256->90 (det classes)" },
        {  4,   64,   32, ROCKET_ACT_RELU,  1, "fc M=4 +bias relu" },
        {  1,   32,   16, ROCKET_ACT_NONE,  0, "fc no-bias aligned" },
        {  1,  100,   10, ROCKET_ACT_RELU6, 1, "fc 1x100->10 +bias relu6 (K,N pad)" },
    };
    for (size_t i = 0; i < sizeof(fcs)/sizeof(fcs[0]); i++) { fail |= run_fc(fcs[i]); printf("\n"); }

    const AddT adds[] = {
        /* H  W  C  act              q  au bu ou  in0(s,z)  in1(s,z)  out(s,z)  name */
        { 8, 8,16, ROCKET_ACT_NONE,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "add float residual" },
        { 6, 6,16, ROCKET_ACT_RELU,  0, 0,0,0, 1.f,0, 1.f,0, 1.f,0, "add float relu" },
        { 8, 8,16, ROCKET_ACT_NONE,  1, 0,0,0, 1.f,0, 2.f,0, 1.f,0, "i8 add (diff scales)" },
        { 5, 5,16, ROCKET_ACT_RELU6, 1, 0,0,0, 1.f,-3,1.f,2, 1.f,5, "i8 add +zp relu6" },
        { 6, 6,16, ROCKET_ACT_NONE,  1, 1,1,1, 1.f,128,2.f,128,1.f,128,"u8 add zp128" },
    };
    for (size_t i = 0; i < sizeof(adds)/sizeof(adds[0]); i++) { fail |= run_add(adds[i]); printf("\n"); }

    const PoolT pools[] = {
        /* IH IW  C  KH KW sy sx same avg act              q  u  in(s,z)  out(s,z) name */
        { 8, 8, 16, 2,2, 2,2, 0, 1, ROCKET_ACT_NONE,  0, 0, 1.f,0, 1.f,0, "avgpool 2x2 s2 VALID f" },
        { 7, 7, 16, 3,3, 2,2, 1, 1, ROCKET_ACT_NONE,  0, 0, 1.f,0, 1.f,0, "avgpool 3x3 s2 SAME f" },
        { 8, 8, 16, 2,2, 2,2, 0, 0, ROCKET_ACT_NONE,  0, 0, 1.f,0, 1.f,0, "maxpool 2x2 s2 VALID f" },
        { 7, 7, 16, 3,3, 1,1, 1, 0, ROCKET_ACT_RELU,  0, 0, 1.f,0, 1.f,0, "maxpool 3x3 SAME relu f" },
        { 8, 8, 16, 2,2, 2,2, 0, 1, ROCKET_ACT_NONE,  1, 0, 1.f,-2,1.f,-2,"i8 avgpool 2x2 s2" },
        { 7, 7, 16, 3,3, 2,2, 1, 1, ROCKET_ACT_NONE,  1, 0, 2.f,0, 1.f,0, "i8 avgpool 3x3 s2 SAME" },
        { 8, 8, 16, 2,2, 2,2, 0, 0, ROCKET_ACT_NONE,  1, 1, 1.f,128,1.f,128,"u8 maxpool 2x2 s2" },
    };
    for (size_t i = 0; i < sizeof(pools)/sizeof(pools[0]); i++) { fail |= run_pool(fd, pools[i]); printf("\n"); }

    const ConcatT concats[] = {
        /* D0 D1 D2 D3  axis nin {parts}   q  u  act              {in scales}   {in zp}     out(s,z) name */
        { 1, 4, 4, 48,  3, 3, {16,16,16},  0, 0, ROCKET_ACT_NONE, {1,1,1,1},{0,0,0,0}, 1.f,0, "concat ch float (SSD head)" },
        { 1, 4, 4, 32,  3, 2, {16,16,0,0}, 0, 0, ROCKET_ACT_RELU, {1,1,1,1},{0,0,0,0}, 1.f,0, "concat ch float relu" },
        { 1, 8, 4, 16,  1, 2, {3,5,0,0},   0, 0, ROCKET_ACT_NONE, {1,1,1,1},{0,0,0,0}, 1.f,0, "concat axis1 float (inner>1)" },
        { 1, 4, 4, 48,  3, 3, {16,16,16},  1, 0, ROCKET_ACT_NONE, {1,2,1,1},{0,0,3,0}, 1.f,0, "i8 concat ch (diff scales)" },
        { 1, 4, 4, 32,  3, 2, {16,16,0,0}, 1, 1, ROCKET_ACT_NONE, {1,2,1,1},{128,128,0,0},1.f,128,"u8 concat ch zp128" },
    };
    for (size_t i = 0; i < sizeof(concats)/sizeof(concats[0]); i++) { fail |= run_concat(concats[i]); printf("\n"); }

    fail |= run_reshape(1 * 4 * 4 * 16, 4, "reshape float [1,4,4,16]->flat"); printf("\n");
    fail |= run_reshape(1 * 8 * 8 * 16, 1, "reshape i8 [1,8,8,16]->flat");    printf("\n");

    printf("---- LAYOUT OPS (transpose / pad / slice / split, byte-exact) ----\n\n");
    {   /* TRANSPOSE: NHWC->NCHW, a 2D swap, a 3D rotation, and an identity perm */
        const int d_nhwc[4] = {1, 5, 7, 16}; const int p_nchw[4] = {0, 3, 1, 2};
        fail |= run_transpose(4, d_nhwc, p_nchw, 4, "transpose NHWC->NCHW f"); printf("\n");
        fail |= run_transpose(4, d_nhwc, p_nchw, 1, "transpose NHWC->NCHW i8"); printf("\n");
        const int d2[2] = {6, 11}; const int p2[2] = {1, 0};
        fail |= run_transpose(2, d2, p2, 4, "transpose 2D [6,11]->[11,6] f"); printf("\n");
        const int d3[3] = {3, 5, 7}; const int p3[3] = {2, 0, 1};
        fail |= run_transpose(3, d3, p3, 1, "transpose 3D rotate i8"); printf("\n");
        const int dI[4] = {1, 4, 4, 8}; const int pI[4] = {0, 1, 2, 3};
        fail |= run_transpose(4, dI, pI, 4, "transpose identity perm f"); printf("\n");
    }
    {   /* SLICE: crop spatial + channel, begin>0, and a degenerate size-1 axis */
        const int din[4] = {1, 8, 8, 16};
        const int b1[4] = {0, 2, 1, 0};  const int s1[4] = {1, 4, 6, 16};
        fail |= run_slice(4, din, b1, s1, 4, "slice spatial f"); printf("\n");
        const int b2[4] = {0, 0, 0, 4};  const int s2[4] = {1, 8, 8, 8};
        fail |= run_slice(4, din, b2, s2, 1, "slice channel i8"); printf("\n");
        const int b3[4] = {0, 7, 0, 0};  const int s3[4] = {1, 1, 8, 16};
        fail |= run_slice(4, din, b3, s3, 4, "slice size-1 row f"); printf("\n");
    }
    {   /* STRIDED_SLICE: positive stride != 1 (downsample), per-axis, float + int8 */
        const int din[4] = {1, 8, 9, 16};
        const int b1[4] = {0, 0, 0, 0}; const int st1[4] = {1, 2, 3, 1}; const int sz1[4] = {1, 4, 3, 16};
        fail |= run_strided_slice(4, din, b1, st1, sz1, 4, "strided HxW /2,/3 f"); printf("\n");
        const int b2[4] = {0, 1, 0, 2}; const int st2[4] = {1, 2, 1, 2}; const int sz2[4] = {1, 4, 9, 7};
        fail |= run_strided_slice(4, din, b2, st2, sz2, 1, "strided begin+ch /2 i8"); printf("\n");
        /* NEGATIVE stride: begin[] is the HIGH index, the walk descends. Mirrors the
         * delegate's signed begin/stride (rocket_strided_slice_bytes is sign-agnostic). */
        /* H fully reversed (begin=7,stride=-1,size=8), W reversed /2 (begin=8,stride=-2,size=5) */
        const int b3[4] = {0, 7, 8, 0}; const int st3[4] = {1, -1, -2, 1}; const int sz3[4] = {1, 8, 5, 16};
        fail |= run_strided_slice(4, din, b3, st3, sz3, 4, "strided H rev, W rev/2 f"); printf("\n");
        /* mixed signs + channel reverse: H +/2 (begin=1,size=4), C reversed (begin=15,stride=-1) */
        const int b4[4] = {0, 1, 0, 15}; const int st4[4] = {1, 2, 1, -1}; const int sz4[4] = {1, 4, 9, 16};
        fail |= run_strided_slice(4, din, b4, st4, sz4, 1, "strided H/2, C rev i8"); printf("\n");
    }
    {   /* BROADCAST elementwise: per-channel [C] + scalar [1] (eval_add lowering) */
        fail |= run_bcast("broadcast EW"); printf("\n");
    }
    {   /* SPLIT: equal split along channel (the inverse of concat) and along width */
        const int din[4] = {1, 4, 4, 48};
        fail |= run_split(4, din, 3, 3, 4, "split ch /3 f"); printf("\n");
        fail |= run_split(4, din, 3, 3, 1, "split ch /3 i8"); printf("\n");
        const int dw[4] = {1, 6, 8, 16};
        fail |= run_split(4, dw, 2, 2, 4, "split width /2 f"); printf("\n");
    }
    {   /* PAD: float zero-fill, PADV2 float constant, int8 zero_point border */
        const int din[4] = {1, 5, 7, 16};
        const int pb[4] = {0, 1, 2, 0}; const int pa[4] = {0, 1, 1, 0};
        const float zf = 0.0f;          fail |= run_pad(4, din, pb, pa, 4, &zf, "pad spatial zero f"); printf("\n");
        const float cf = 1.5f;          fail |= run_pad(4, din, pb, pa, 4, &cf, "padv2 spatial const f"); printf("\n");
        const signed char zp = -3;      fail |= run_pad(4, din, pb, pa, 1, &zp, "pad spatial zp i8"); printf("\n");
        const int pc[4] = {0, 0, 0, 8}; const int za[4] = {0, 0, 0, 0};
        fail |= run_pad(4, din, za, pc, 4, &zf, "pad channel-after zero f"); printf("\n");
    }

    printf("---- NCHW-resident inter-op helpers ----\n\n");
    fail |= run_nchw_pad_test(16, 8, 8, 1, 1, "nchw_pad 3x3-pad");      printf("\n");
    fail |= run_nchw_pad_test(192, 32, 32, 1, 1, "nchw_pad C=192 32x32"); printf("\n");
    fail |= run_nchw_pad_test(8, 10, 6, 2, 0, "nchw_pad asym 5x5/1x");  printf("\n");
    fail |= run_nchw_bias_act_test(16, 8, 8, ROCKET_ACT_NONE,  "nchw_bias_act none");  printf("\n");
    fail |= run_nchw_bias_act_test(24, 6, 6, ROCKET_ACT_RELU6, "nchw_bias_act relu6"); printf("\n");
    fail |= run_nchw_bias_act_test(192, 32, 32, ROCKET_ACT_RELU, "nchw_bias_act C=192");printf("\n");

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
