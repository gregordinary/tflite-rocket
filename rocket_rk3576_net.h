// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
//
// rocket_rk3576_net.h — the RK3576 GRAPH path: a delegated partition run as one
// planned network rather than as a sequence of independent ops.
//
// WHY THIS IS A SEPARATE PATH AND NOT AN OP MAPPING. On the RK3588 the delegate claims
// ops and runs them one at a time, and that is the right shape there. On the RK3576 it is
// not: the same MobileNetV1-224 costs ~115 ms as per-op calls with transient weights,
// ~21 ms with the weights resident, 10.4-10.5 ms with the tensors kept in the part's cube
// layout between layers, and 5.0 ms when the whole run goes out as ONE hardware kick
// [HW sweep, H96 MAX M9]. A frontend that only calls the op entries therefore gets about a
// twentieth of the part's demonstrated throughput. The three levers between those numbers
// are all properties of the GRAPH — which tensors never have to be transposed, which
// producers write slices of one buffer, and which runs of layers are one submit — so none
// of them can be expressed one op at a time.
//
// WHAT IS HERE AND WHAT IS NOT. The placement and linking rules are NOT here: they are
// rocketnpu's rocket_graph_plan_new() (the `rocketgraph` component), because a second copy
// of them would fork silently — every refusal class closed on this part so far was a
// host-buffer rule rather than the hardware's, and a wrong one computes a full, correctly
// sized, entirely plausible surface. What is here is what a CALLER owns:
//
//   * the TFLite lowering — which nodes are claimed, and the geometry, quantization and
//     operand graph each one turns into. That includes TFLite's asymmetric SAME padding as
//     an output EXTENT (the CNA takes no configured trailing pad; it derives the pad its
//     last window consumes from the extent and the leading pad) and the uint8 -> int8
//     rebase (the part is int8; subtracting 128 from a tensor's data and its zero point
//     together is exact and cancels, so `q - zp` is untouched);
//   * the weights, transposed once into the layouts the library entries take;
//   * the tensors, their buffers and their lifetimes;
//   * NHWC <-> CHW at the partition boundary, and nowhere else.
//
// LAYOUT. Everything inside a partition is CHW int8, which is what the RK3576 entries
// take. The two transposes at the boundary are the only ones a fully-linked graph pays.

#ifndef ROCKET_RK3576_NET_H
#define ROCKET_RK3576_NET_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "rocket_graph_rk3576.h"

namespace rocket_rk3576 {

// ---------------------------------------------------------------------------
// Is this an RK3576? The library detects the part from the rocket-bound platform device's
// DT `compatible` and falls back to the RK3588 profile, so the profile name is the
// question to ask. A wrong answer here is not a slow path but a wrong one: the RK3588
// generators refuse on this part by construction, and the RK3576 entries refuse there.
// ---------------------------------------------------------------------------
inline bool is_rk3576() {
    const struct rocket_hw_profile *hw = rocket_hw_current();
    return hw && hw->name && std::strcmp(hw->name, "rk3576") == 0;
}

// ---------------------------------------------------------------------------
// Small tensor helpers
// ---------------------------------------------------------------------------
inline bool q8(TfLiteType t) { return t == kTfLiteInt8 || t == kTfLiteUInt8; }

// The rebase a tensor's storage type implies. The part is int8; a uint8 tensor's data and
// its zero point are both shifted by -128, so `q - zp` — the only thing the arithmetic
// reads — is unchanged. Per TENSOR rather than per model: the shift cancels within a
// tensor, so a model mixing the two needs no special case.
inline int shift_of(TfLiteType t) { return t == kTfLiteUInt8 ? 128 : 0; }

inline int tdim(const TfLiteTensor &t, int i) {
    return (t.dims && i < t.dims->size) ? t.dims->data[i] : 1;
}

// A tensor's single (scale, zero point). False for a missing or per-axis one.
inline bool quant1(const TfLiteTensor &t, float *scale, int *zp) {
    if (t.quantization.type == kTfLiteAffineQuantization && t.quantization.params) {
        const auto *aq =
            reinterpret_cast<const TfLiteAffineQuantization *>(t.quantization.params);
        if (!aq->scale || aq->scale->size != 1) return false;
        *scale = aq->scale->data[0];
        *zp = (aq->zero_point && aq->zero_point->size == 1) ? aq->zero_point->data[0] : 0;
        return *scale > 0.f;
    }
    if (t.params.scale <= 0.f) return false;
    *scale = t.params.scale;
    *zp = t.params.zero_point;
    return true;
}

// TFLite [OC,KH,KW,IC] -> the library's [OC,IC,KH,KW]; depthwise [1,KH,KW,C] -> [C,KH,KW].
// Rebased into int8 in the same pass. At namespace scope because the CLAIM gate needs it as
// well as the packer: a per-axis layer's accuracy bound is a question about the weights, and
// a depthwise filter's channel is STRIDED in TFLite's layout, so the per-channel sums the
// bound is built from are wrong unless the transpose has happened first.
inline bool transpose_weights(const TfLiteTensor &f, bool dw, std::vector<int8_t> *w) {
    const int KH = f.dims->data[1], KW = f.dims->data[2];
    const int sh = shift_of(f.type);
    const uint8_t *src = reinterpret_cast<const uint8_t *>(f.data.data);
    if (!src) return false;
    if (dw) {
        const int C = f.dims->data[3];
        w->assign((size_t)C * KH * KW, 0);
        for (int c = 0; c < C; c++)
            for (int y = 0; y < KH; y++)
                for (int x = 0; x < KW; x++)
                    (*w)[((size_t)c * KH + y) * KW + x] =
                        (int8_t)((int)src[((size_t)y * KW + x) * C + c] - sh);
        return true;
    }
    const int OC = f.dims->data[0], IC = f.dims->data[3];
    w->assign((size_t)OC * IC * KH * KW, 0);
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int y = 0; y < KH; y++)
                for (int x = 0; x < KW; x++)
                    (*w)[(((size_t)oc * IC + ic) * KH + y) * KW + x] =
                        (int8_t)((int)src[(((size_t)oc * KH + y) * KW + x) * IC + ic] - sh);
    return true;
}

// WHETHER A PER-AXIS CONVOLUTION'S SCALE SPREAD HAS AN EXPRESSION ON THIS PART, asked of
// the library rather than decided here. A tile is one task and a task carries one OUT_CVT
// shift, so a channel reaches its own scale only through the int16 C in its coefficient
// group; past what that ramp spans the layer computes a full, correctly sized, entirely
// plausible surface at a gain that is wrong by a factor. It has to be asked HERE because
// the library's own refusal at Prepare would fail the whole model, where an unclaimed node
// is one TFLite runs itself.
//
// Per-tensor filters are not this question and pass straight through. The cost is one
// weight transpose per candidate per-axis convolution, at load.
inline bool peraxis_ok(const TfLiteTensor &f, const TfLiteTensor &b, bool dw, int IC,
                       float in_scale, int in_zp, float out_scale, int out_zp) {
    if (f.quantization.type != kTfLiteAffineQuantization || !f.quantization.params)
        return true;
    const auto *aq =
        reinterpret_cast<const TfLiteAffineQuantization *>(f.quantization.params);
    if (!aq->scale || aq->scale->size <= 1) return true;
    const int OC = dw ? f.dims->data[3] : f.dims->data[0];
    if (aq->scale->size != OC || !b.data.i32) return false;
    std::vector<int8_t> w;
    if (!transpose_weights(f, dw, &w)) return false;
    rocket_conv2d_desc d;
    std::memset(&d, 0, sizeof d);
    d.ic = dw ? OC : IC;
    d.oc = OC;
    d.kh = f.dims->data[1];
    d.kw = f.dims->data[2];
    d.depthwise = dw;
    // A narrow per-axis convolution can only be the ordinary encoding: the packed-image
    // first conv refuses a per-axis quantization, so pack() will hand it the same flag.
    d.direct_datapath = (!dw && IC <= 4) ? 1 : 0;
    // Only the ACCURACY refusal is a claim answer. ROCKET_E_SHAPE says this path does not
    // take the descriptor at all, which is a question the rest of the gate owns.
    //
    // The OUTPUT quantization is part of the question, not bookkeeping: a clamped channel
    // whose filter is all zero reaches one accumulator, and whether its wrong gain changes
    // a byte depends on where that value sits against the saturation rails.
    return rocket_conv2d_int8_perchannel_plan_rk3576(&d, w.data(), b.data.i32, in_scale,
                                                     aq->scale->data, out_scale, in_zp,
                                                     out_zp, nullptr, nullptr)
           != ROCKET_E_UNSUPPORTED;
}

// TFLite's own padding resolution, as an explicit (out, lead, trail). `trail` is the pad
// the LAST WINDOW CONSUMES — what the CNA derives from the output extent and the leading
// pad — and it can be smaller than the pad the model declares: a stride-2 3x3 padding one
// row at each end of an even plane never reaches the trailing one.
inline void resolve_pad(int in_dim, int k, int stride, TfLitePadding p, int ex_lead,
                        int ex_trail, int *out, int *lead, int *trail) {
    if (ex_lead >= 0) {                       // an explicit PAD op folded into this node
        *out = (in_dim + ex_lead + ex_trail - k) / stride + 1;
        *lead = ex_lead;
        const int reach = (*out - 1) * stride + k;
        *trail = reach > in_dim + ex_lead ? reach - (in_dim + ex_lead) : 0;
        return;
    }
    if (p == kTfLitePaddingSame) {
        *out = (in_dim + stride - 1) / stride;
        const int total = (*out - 1) * stride + k - in_dim;
        *lead = total > 0 ? total / 2 : 0;
        *trail = total > 0 ? total - total / 2 : 0;
        return;
    }
    *out = (in_dim - k + stride) / stride;
    *lead = *trail = 0;
}

// A fused activation is FREE while its clamp IS the whole storage range — a quantizer that
// sees a ReLU puts the zero point at the bottom of the range, and a ReLU6's output scale
// IS 6/range. Where it is not, the node needs a host clamp nothing here applies, so it is
// refused rather than computed without one.
inline bool act_is_free(TfLiteFusedActivation a, float out_scale, int out_zp,
                        TfLiteType out_type) {
    if (a == kTfLiteActNone) return true;
    if (a != kTfLiteActRelu && a != kTfLiteActRelu6) return false;
    const int qmin = out_type == kTfLiteInt8 ? -128 : 0;
    const int qmax = out_type == kTfLiteInt8 ? 127 : 255;
    const int lo = out_zp > qmin ? out_zp : qmin;
    int hi = qmax;
    if (a == kTfLiteActRelu6) {
        const int cap = out_zp + (int)(6.0f / out_scale + 0.5f);
        if (cap < hi) hi = cap;
    }
    return lo == qmin && hi == qmax;
}

// ---------------------------------------------------------------------------
// The claim: which nodes of the MODEL this path takes.
//
// Computed over the whole execution plan rather than per node, and the delegate's Prepare
// and the kernel's Init both call it, so the two cannot disagree about the partition. One
// rule needs the whole graph: a PAD is claimed only where it FOLDS. It is not a layer — it
// is the lead/trail pair of whichever convolution or pool consumes it — so a PAD whose
// output escapes the claimed set would have to be materialised and is left to the CPU.
// ---------------------------------------------------------------------------

// The per-node KIND gate: everything decidable from one node.
inline bool kind_supported(const TfLiteRegistration *reg, const TfLiteNode *node,
                           TfLiteContext *ctx) {
    if (!reg || !node || !node->inputs || !node->outputs) return false;
    if (node->inputs->size < 1 || node->outputs->size != 1) return false;
    const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];
    if (!q8(o.type)) return false;
    float osc; int ozp;
    if (!quant1(o, &osc, &ozp)) return false;

    switch (reg->builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d: {
        const bool dw = reg->builtin_code == kTfLiteBuiltinDepthwiseConv2d;
        if (node->inputs->size < 3) return false;          // a real int32 bias is required
        const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
        const TfLiteTensor &f = ctx->tensors[node->inputs->data[1]];
        const TfLiteTensor &b = ctx->tensors[node->inputs->data[2]];
        if (!q8(in.type) || !q8(f.type) || b.type != kTfLiteInt32) return false;
        if (!in.dims || in.dims->size != 4 || in.dims->data[0] != 1) return false;
        if (!o.dims || o.dims->size != 4) return false;
        if (!f.dims || f.dims->size != 4 || !f.data.data || !b.data.data) return false;
        if (f.dims->data[3] != in.dims->data[3]) return false;   // grouped conv / bad shape
        float isc; int izp;
        if (!quant1(in, &isc, &izp)) return false;
        int sy, sx, dy, dx;
        TfLiteFusedActivation act;
        TfLitePadding pad;
        if (dw) {
            const auto *p =
                reinterpret_cast<const TfLiteDepthwiseConvParams *>(node->builtin_data);
            if (!p || p->depth_multiplier != 1) return false;
            sy = p->stride_height; sx = p->stride_width;
            dy = p->dilation_height_factor; dx = p->dilation_width_factor;
            act = p->activation;
            pad = p->padding;
        } else {
            const auto *p = reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);
            if (!p) return false;
            sy = p->stride_height; sx = p->stride_width;
            dy = p->dilation_height_factor; dx = p->dilation_width_factor;
            act = p->activation;
            pad = p->padding;
        }
        if (sy < 1 || sx < 1) return false;
        // DILATION is refused in the library's own RK3576 check and has never been driven
        // on the part, so there is no map here to be wrong about.
        if (dy != 1 || dx != 1) return false;
        // A DEPTHWISE CONVOLUTION OF FOUR OR FEWER CHANNELS HAS NO FORM ON THIS PART. The
        // packed-image first conv folds the kernel's columns into the channel axis, so
        // there is nothing left to be depthwise over, and the direct datapath's flag is
        // refused on a depthwise descriptor rather than ignored. It has to be refused HERE
        // — a layer the library will not pack is a Prepare failure, which fails the whole
        // model, where an unclaimed node is just one TFLite runs itself. No classifier in
        // the corpus has one; an SSD head does.
        if (dw && in.dims->data[3] <= 4) return false;
        if (!act_is_free(act, osc, ozp, o.type)) return false;
        // AND THE PART'S GEOMETRY BOUNDS, ASKED RATHER THAN ASSUMED — the resident weight
        // slice and the row window, neither of which any inspection of this node could
        // predict and both of which the pack reaches. Same reason as the pool's plan call
        // below: a refusal here costs one node TFLite runs itself, where the same refusal
        // at Prepare fails the whole model.
        //
        // The geometry is the node's OWN, before an explicit PAD is folded into it. That
        // is sound rather than merely conservative: a fold makes the plane taller, and the
        // row window's refusal ("no window fits at all") is a function of the plane WIDTH,
        // the channel counts and the kernel — not of its height, which only decides how
        // many tasks the window is cut into.
        {
            rocket_conv2d_desc gd;
            std::memset(&gd, 0, sizeof gd);
            gd.ic = in.dims->data[3];
            gd.oc = dw ? in.dims->data[3] : f.dims->data[0];
            gd.ih = in.dims->data[1]; gd.iw = in.dims->data[2];
            gd.kh = f.dims->data[1];  gd.kw = f.dims->data[2];
            gd.stride_y = sy; gd.stride_x = sx;
            gd.dil_y = 1; gd.dil_x = 1;
            gd.depthwise = dw;
            int oh, ow, ly, lx, ty, tx;
            resolve_pad(gd.ih, gd.kh, sy, pad, -1, 0, &oh, &ly, &ty);
            resolve_pad(gd.iw, gd.kw, sx, pad, -1, 0, &ow, &lx, &tx);
            gd.pad_top = ly; gd.pad_left = lx;
            gd.direct_datapath = (!dw && gd.ic <= 4) ? 1 : 0;
            if (rocket_conv2d_int8_plan_rk3576(&gd) != ROCKET_OK) return false;
        }
        // AND THE PART'S PER-AXIS ACCURACY BOUND, ASKED RATHER THAN ASSUMED — the same
        // pure library check the pack will apply, so a claim made here cannot fail there.
        return peraxis_ok(f, b, dw, in.dims->data[3], isc, izp - shift_of(in.type), osc,
                          ozp - shift_of(o.type));
    }
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinMaxPool2d: {
        const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
        if (!q8(in.type) || !in.dims || in.dims->size != 4 || in.dims->data[0] != 1)
            return false;
        const auto *p = reinterpret_cast<const TfLitePoolParams *>(node->builtin_data);
        if (!p || p->activation != kTfLiteActNone) return false;
        if (p->stride_height < 1 || p->stride_width < 1) return false;
        float isc; int izp;
        if (!quant1(in, &isc, &izp)) return false;
        // The PPU reduces; it does not requantize. A pool that changes scale is TFLite's
        // arithmetic and not this program's.
        if (in.type != o.type || isc != osc || izp != ozp) return false;
        // AND THE PART'S OWN BOUND, ASKED RATHER THAN ASSUMED. rocket_pool_int8_rk3576_plan()
        // is pure, so the shape can be priced here, where a refusal costs one node TFLite
        // runs itself — against Prepare, where it would fail the model. An explicit PAD
        // ahead of a pool is not folded in yet at this point, so a padded shape is checked
        // at the geometry the node itself carries.
        rocket_pool_desc pd;
        std::memset(&pd, 0, sizeof pd);
        pd.c = in.dims->data[3]; pd.ih = in.dims->data[1]; pd.iw = in.dims->data[2];
        pd.kh = p->filter_height; pd.kw = p->filter_width;
        pd.stride_y = p->stride_height; pd.stride_x = p->stride_width;
        int oh, ow, ly, lx, ty, tx;
        resolve_pad(pd.ih, pd.kh, pd.stride_y, p->padding, -1, 0, &oh, &ly, &ty);
        resolve_pad(pd.iw, pd.kw, pd.stride_x, p->padding, -1, 0, &ow, &lx, &tx);
        pd.pad_top = ly; pd.pad_bottom = ty; pd.pad_left = lx; pd.pad_right = tx;
        pd.method = reg->builtin_code == kTfLiteBuiltinAveragePool2d ? POOL_METHOD_AVG
                                                                    : POOL_METHOD_MAX;
        pd.avg_exclude_pad = pd.method == POOL_METHOD_AVG ? 1 : 0;
        return rocket_pool_int8_rk3576_plan(&pd) == ROCKET_OK;
    }
    case kTfLiteBuiltinAdd: {
        if (node->inputs->size != 2) return false;
        const auto *p = reinterpret_cast<const TfLiteAddParams *>(node->builtin_data);
        if (!p || !act_is_free(p->activation, osc, ozp, o.type)) return false;
        const TfLiteTensor &a = ctx->tensors[node->inputs->data[0]];
        const TfLiteTensor &b = ctx->tensors[node->inputs->data[1]];
        if (!q8(a.type) || a.type != b.type || a.type != o.type) return false;
        if (!a.dims || !b.dims || !o.dims) return false;
        if (a.dims->size != 4 || b.dims->size != 4 || o.dims->size != 4) return false;
        for (int k = 0; k < 4; k++)
            if (a.dims->data[k] != b.dims->data[k] || a.dims->data[k] != o.dims->data[k])
                return false;                                   // no broadcast
        float s; int z;
        if (!quant1(a, &s, &z) || !quant1(b, &s, &z)) return false;
        return !a.data.data && !b.data.data;                    // both are activations
    }
    case kTfLiteBuiltinConcatenation: {
        const auto *p =
            reinterpret_cast<const TfLiteConcatenationParams *>(node->builtin_data);
        if (!p || p->activation != kTfLiteActNone) return false;
        if (!o.dims || o.dims->size != 4) return false;
        if (p->axis != 3 && p->axis != -1) return false;
        if (node->inputs->size < 2 || node->inputs->size > ROCKET_GRAPH_MAX_SRC)
            return false;
        int off = 0;
        for (int k = 0; k < node->inputs->size; k++) {
            const TfLiteTensor &t = ctx->tensors[node->inputs->data[k]];
            if (!q8(t.type) || t.type != o.type || !t.dims || t.dims->size != 4)
                return false;
            if (t.data.data) return false;                      // a constant operand
            float s; int z;
            if (!quant1(t, &s, &z)) return false;
            // A CONCATENATION IS PLACEMENT ONLY WHERE EVERY OPERAND ALREADY CARRIES THE
            // OUTPUT'S QUANTIZATION. TFLite's own kernel requantizes one that does not,
            // and neither this lowering nor the planner's placement can.
            if (s != osc || z != ozp) return false;
            // A placed slice starts every SIXTEEN channels — what one cube atom
            // interleaves. An operand landing anywhere else could still be copied on the
            // host, but it could never be wired, which is the whole value here.
            if (off % 16) return false;
            off += t.dims->data[3];
        }
        return off == o.dims->data[3];
    }
    case kTfLiteBuiltinQuantize: {
        // A REQUANTIZATION EDGE. It lowers onto a DEPTHWISE 1x1 IDENTITY: a unit weight at
        // unit scale leaves the convolution's own epilogue computing exactly TFLite's
        // kernel, so it needs no new op, and the layer can write into a concatenation's
        // slice like any other producer.
        const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
        if (!q8(in.type) || in.type != o.type) return false;
        if (!in.dims || !o.dims || in.dims->size != 4 || o.dims->size != 4) return false;
        for (int k = 0; k < 4; k++)
            if (in.dims->data[k] != o.dims->data[k]) return false;
        if (in.dims->data[0] != 1) return false;
        float s; int z;
        return quant1(in, &s, &z);
    }
    case kTfLiteBuiltinPad: {
        // Not a layer: the lead/trail pair of whichever conv or pool consumes it. Whether
        // it FOLDS is decided over the whole graph, below.
        if (node->inputs->size != 2) return false;
        const TfLiteTensor &in = ctx->tensors[node->inputs->data[0]];
        const TfLiteTensor &pv = ctx->tensors[node->inputs->data[1]];
        if (!q8(in.type) || in.type != o.type) return false;
        if (!in.dims || in.dims->size != 4) return false;
        if (!pv.data.data || pv.type != kTfLiteInt32 || !pv.dims || pv.dims->size != 2 ||
            pv.dims->data[0] != 4 || pv.dims->data[1] != 2) return false;
        const int32_t *p = pv.data.i32;
        if (p[0] || p[1] || p[6] || p[7]) return false;         // batch or channel padding
        if (p[2] < 0 || p[3] < 0 || p[4] < 0 || p[5] < 0) return false;
        float isc; int izp; float osc2; int ozp2;
        if (!quant1(in, &isc, &izp) || !quant1(o, &osc2, &ozp2)) return false;
        return isc == osc2 && izp == ozp2;
    }
    default:
        return false;
    }
}

// Does `node` consume tensor `t` as an operand a PAD could fold into? Only operand 0 of a
// convolution or a pool: everything else reads a constant or a second activation.
inline bool folds_pad(const TfLiteRegistration *reg, const TfLiteNode *node, int t) {
    switch (reg->builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinMaxPool2d:
        return node->inputs->size > 0 && node->inputs->data[0] == t;
    default:
        return false;
    }
}

// The claimed set over the whole model, by node index. Deterministic, so the partitioner
// and the kernel agree by construction rather than by both being careful.
inline bool build_claim(TfLiteContext *ctx, std::vector<char> *claimed,
                        std::vector<int> *order) {
    TfLiteIntArray *plan = nullptr;
    if (ctx->GetExecutionPlan(ctx, &plan) != kTfLiteOk) return false;
    order->assign(plan->data, plan->data + plan->size);

    int max_node = 0;
    for (int n : *order) if (n + 1 > max_node) max_node = n + 1;
    claimed->assign(max_node, 0);

    std::vector<TfLiteNode *> nodes(max_node, nullptr);
    std::vector<TfLiteRegistration *> regs(max_node, nullptr);
    for (int n : *order) {
        if (ctx->GetNodeAndRegistration(ctx, n, &nodes[n], &regs[n]) != kTfLiteOk)
            return false;
        (*claimed)[n] = kind_supported(regs[n], nodes[n], ctx) ? 1 : 0;
    }

    // A claimed PAD must fold into every reader of its output, and have at least one.
    // Dropping one can only shrink the set (a PAD ahead of a PAD), so the fixpoint
    // terminates; in practice it settles in one pass.
    for (bool changed = true; changed;) {
        changed = false;
        for (int n : *order) {
            if (!(*claimed)[n] || regs[n]->builtin_code != kTfLiteBuiltinPad) continue;
            const int t = nodes[n]->outputs->data[0];
            int readers = 0;
            bool all_fold = true;
            for (int m : *order) {
                if (m == n) continue;
                bool reads = false;
                for (int k = 0; k < nodes[m]->inputs->size; k++)
                    if (nodes[m]->inputs->data[k] == t) { reads = true; break; }
                if (!reads) continue;
                readers++;
                if (!(*claimed)[m] || !folds_pad(regs[m], nodes[m], t)) all_fold = false;
            }
            if (!readers || !all_fold) { (*claimed)[n] = 0; changed = true; }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// One layer of a partition, as this frontend needs it.
// ---------------------------------------------------------------------------

// A PAD op folded into the node that consumes it.
struct FoldedPad { int src_t, top, bottom, left, right; };

struct Layer {
    rocket_graph_kind kind = ROCKET_GRAPH_CONV;
    int node = -1;
    int in_t = -1, out_t = -1;                 // TFLite tensor indices
    int src[ROCKET_GRAPH_MAX_SRC];             // producer LAYER index, -1 = outside
    int src_t[ROCKET_GRAPH_MAX_SRC];           // and the tensor each operand is
    int src_c[ROCKET_GRAPH_MAX_SRC];           // its channel count (a concat's slices)

    unsigned ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0;
    unsigned kh = 1, kw = 1, sy = 1, sx = 1;
    unsigned pl_y = 0, pl_x = 0, pt_y = 0, pt_x = 0;
    int in_zp = 0, w_zp = 0, out_zp = 0;
    float in_scale = 1.f, w_scale = 1.f, out_scale = 1.f;
    int in2_zp = 0;
    float in2_scale = 0.f;
    bool avg = false;                          // pooling method

    std::vector<int8_t> w;                     // library layout, int8 domain
    std::vector<int32_t> bias;
    std::vector<float> w_scale_oc;             // empty => per-tensor

    rocket_conv2d_int8_weights_rk3576 *conv = nullptr;
    rocket_pool_int8_rk3576_handle *pool = nullptr;
    std::vector<int8_t> buf;                   // the materialised CHW output, when it has one

    Layer() {
        for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++) {
            src[k] = -1; src_t[k] = -1; src_c[k] = 0;
        }
    }
};

// The descriptor a convolution layer runs under. The leading pad is the CNA's register and
// the trailing one is DERIVED from the output extent, which is why `oh/ow` are stated
// rather than left to the symmetric formula: TFLite's SAME at an even plane and stride 2 is
// a leading pad of zero against an extent one larger than that formula gives.
inline void conv_desc_of(const Layer &L, rocket_conv2d_desc *d, int direct) {
    std::memset(d, 0, sizeof *d);
    d->ic = (int)L.ic; d->ih = (int)L.ih; d->iw = (int)L.iw;
    d->oc = (int)L.oc; d->kh = (int)L.kh; d->kw = (int)L.kw;
    d->stride_y = (int)L.sy; d->stride_x = (int)L.sx;
    d->dil_y = d->dil_x = 1;
    d->pad_top = (int)L.pl_y; d->pad_left = (int)L.pl_x;
    d->oh = (int)L.oh; d->ow = (int)L.ow;
    d->depthwise = L.kind == ROCKET_GRAPH_DWCONV;
    d->direct_datapath = direct;
}

inline void pool_desc_of(const Layer &L, rocket_pool_desc *p) {
    std::memset(p, 0, sizeof *p);
    p->c = (int)L.ic; p->ih = (int)L.ih; p->iw = (int)L.iw;
    p->kh = (int)L.kh; p->kw = (int)L.kw;
    p->stride_y = (int)L.sy; p->stride_x = (int)L.sx;
    p->pad_top = (int)L.pl_y; p->pad_left = (int)L.pl_x;
    p->pad_bottom = (int)L.pt_y; p->pad_right = (int)L.pt_x;
    p->method = L.avg ? POOL_METHOD_AVG : POOL_METHOD_MAX;
    // TFLite's AVERAGE_POOL_2D divides a border window by the taps that fell inside the
    // plane. That is the model's arithmetic and not a choice: the PPU has a mode bit for
    // each, and at pad 0 the two are the same function.
    p->avg_exclude_pad = L.avg ? 1 : 0;
}

// An add's descriptor: a 1x1 convolution over its two operands concatenated along
// channels. `direct_datapath` because a narrow residual would otherwise be routed to the
// packed-image first conv, which is a different program.
inline void add_desc_of(const Layer &L, rocket_conv2d_desc *d) {
    std::memset(d, 0, sizeof *d);
    d->ic = (int)(rocket_graph_add_boff(L.oc) + L.oc);
    d->ih = (int)L.oh; d->iw = (int)L.ow;
    d->oc = (int)L.oc;
    d->kh = d->kw = 1;
    d->stride_y = d->stride_x = 1;
    d->dil_y = d->dil_x = 1;
    d->direct_datapath = 1;
}

// ---------------------------------------------------------------------------
// The kernel: one per delegated partition.
// ---------------------------------------------------------------------------
class Kernel {
public:
    explicit Kernel(bool profile) : profile_(profile) {}

    ~Kernel() {
        // The plan BORROWS the handles for its chains and owns the shared buffers, so it
        // goes first.
        rocket_graph_plan_free(plan_);
        plan_ = nullptr;
        for (Layer &L : layers_) {
            if (L.conv) rocket_conv2d_int8_weights_free_rk3576(fd_, L.conv);
            if (L.pool) rocket_pool_int8_free_rk3576(fd_, L.pool);
        }
        if (fd_ >= 0) rocket_close(fd_);
    }

    // ---- Init: the TFLite lowering ----------------------------------------
    TfLiteStatus Init(TfLiteContext *ctx, const TfLiteDelegateParams *params) {
        std::vector<int> mine(params->nodes_to_replace->data,
                              params->nodes_to_replace->data +
                                  params->nodes_to_replace->size);

        // PAD folding. A claimed PAD is the lead/trail of its consumer and never a layer.
        std::vector<std::pair<int, FoldedPad>> pads;

        for (int n : mine) {
            TfLiteNode *node = nullptr;
            TfLiteRegistration *reg = nullptr;
            if (ctx->GetNodeAndRegistration(ctx, n, &node, &reg) != kTfLiteOk)
                return kTfLiteError;
            if (reg->builtin_code == kTfLiteBuiltinPad) {
                const int32_t *p = ctx->tensors[node->inputs->data[1]].data.i32;
                const int s = node->inputs->data[0];
                FoldedPad f{s, p[2], p[3], p[4], p[5]};
                for (const auto &pr : pads)                   // a pad ahead of a pad
                    if (pr.first == s) {
                        f.src_t = pr.second.src_t;
                        f.top += pr.second.top;   f.bottom += pr.second.bottom;
                        f.left += pr.second.left; f.right += pr.second.right;
                    }
                pads.emplace_back(node->outputs->data[0], f);
                continue;
            }
            Layer L;
            L.node = n;
            if (!lower(ctx, reg, node, pads, &L)) {
                std::fprintf(stderr,
                             "[rocket/rk3576] node %d was claimed but does not lower\n", n);
                return kTfLiteError;
            }
            producer_[L.out_t] = (int)layers_.size();
            layers_.push_back(std::move(L));
        }

        // The operand graph, BY PRODUCER. A residual network needs that: an add's second
        // operand is produced three to five layers back and a ping-pong of buffers cannot
        // express it — and a ResNet block that changes width reads its two operands the
        // other way round, so "the layer before" is wrong in both directions.
        for (Layer &L : layers_)
            for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++) {
                if (L.src_t[k] < 0) continue;
                const auto it = producer_.find(L.src_t[k]);
                L.src[k] = it == producer_.end() ? -1 : it->second;
            }

        for (int i = 0; i < params->output_tensors->size; i++)
            part_out_.insert(params->output_tensors->data[i]);
        return kTfLiteOk;
    }

    // ---- Prepare: pack the weights, then plan -----------------------------
    TfLiteStatus Prepare(TfLiteContext *ctx) {
        (void)ctx;
        if (fd_ < 0) {
            fd_ = rocket_open();
            if (fd_ < 0) {
                std::fprintf(stderr, "[rocket/rk3576] no NPU device (%d)\n", fd_);
                return kTfLiteError;
            }
        }
        if (planned_) return kTfLiteOk;

        for (size_t i = 0; i < layers_.size(); i++)
            if (!pack(layers_[i])) {
                std::fprintf(stderr, "[rocket/rk3576] layer %zu (%s) would not pack\n", i,
                             kind_name(layers_[i].kind));
                return kTfLiteError;
            }

        graph_.assign(layers_.size(), rocket_graph_layer{});
        for (size_t i = 0; i < layers_.size(); i++) {
            const Layer &L = layers_[i];
            rocket_graph_layer &G = graph_[i];
            G.kind = L.kind;
            G.ic = L.ic; G.oc = L.oc; G.oh = L.oh; G.ow = L.ow;
            G.w_zp = L.w_zp; G.out_zp = L.out_zp;
            for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++)
                G.src[k] = L.src[k] < 0 ? ROCKET_GRAPH_NO_SRC : (unsigned)L.src[k];
            // Nothing here prepares a layer's input on the host: a partition input is an
            // ordinary row-major scatter, which a chain does for itself.
            G.host_input = 0;
            // A TENSOR READ BOTH INSIDE AND OUTSIDE THE PARTITION MUST BE MATERIALISED,
            // and the planner cannot know that — it sees only the inside reader, and would
            // leave the tensor in cube layout for it. Hiding the handle from the
            // DESCRIPTION is how a layer says it cannot be linked; the handle itself is
            // still what runs it. It costs one join and never a wrong answer, where the
            // other way round costs a partition output nothing ever writes.
            const bool escapes = part_out_.count(L.out_t) && has_inside_consumer(i);
            G.conv = escapes ? nullptr : L.conv;
            G.pool = escapes ? nullptr : L.pool;
            // A CONCATENATION HAS NO HANDLE TO HIDE, so it says the same thing through the
            // description's own flag. Placing one wires its operands into slices of a
            // buffer its consumers read as a cube and leaves it owning no row-major
            // tensor, which for a partition output is a surface the runtime reads and
            // nothing writes. Set on every partition output rather than only where there
            // is an inside consumer: the planner places a concatenation only when one
            // exists, so the wider condition costs nothing and states the fact plainly.
            G.row_major_out = part_out_.count(L.out_t) ? 1u : 0u;
        }

        plan_ = rocket_graph_plan_new(fd_, graph_.data(), (unsigned)graph_.size(),
                                      profile_);
        if (!plan_) return kTfLiteError;
        rocket_graph_plan_kicks(plan_);
        planned_ = true;

        // A materialised layer needs a buffer of its own. One per layer rather than a
        // ping-pong: a skip source is read several layers later, and a buffer per producer
        // makes that a property of the allocation instead of a lifetime rule to get right.
        for (size_t i = 0; i < layers_.size(); i++) {
            Layer &L = layers_[i];
            if (plan_->cube_out[i] || rocket_graph_is_placement(plan_, (unsigned)i))
                continue;
            L.buf.assign((size_t)L.oc * L.oh * L.ow, 0);
        }
        // A partition output with no row-major tensor would be a silent wrong answer, so
        // it is a loud refusal here instead. Both routes to it are now closed above — a
        // convolution or a pool by its hidden handle, a concatenation by `row_major_out` —
        // so this is the assertion that they are, and not a path a model reaches. It stays
        // because the alternative to reaching it is a surface the runtime reads and
        // nothing writes.
        for (size_t i = 0; i < layers_.size(); i++)
            if (part_out_.count(layers_[i].out_t) && layers_[i].buf.empty()) {
                std::fprintf(stderr, "[rocket/rk3576] layer %zu is a partition output and "
                             "the plan leaves it in cube layout\n", i);
                return kTfLiteError;
            }

        if (profile_)
            std::fprintf(stderr, "[rocket/rk3576] %zu layers: %d joins, %d far links, "
                         "%d run(s) over %d layers\n", layers_.size(), plan_->joins,
                         plan_->far_links, plan_->kick_runs, plan_->kick_layers);
        return kTfLiteOk;
    }

    // ---- Eval: run the plan ------------------------------------------------
    TfLiteStatus Eval(TfLiteContext *ctx) {
        // The partition's INPUTS, transposed once into the CHW the entries take.
        for (const Layer &L : layers_)
            for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++) {
                if (L.src_t[k] < 0 || L.src[k] >= 0) continue;
                const TfLiteTensor &t = ctx->tensors[L.src_t[k]];
                const int H = tdim(t, 1), W = tdim(t, 2), C = tdim(t, 3);
                std::vector<int8_t> &dst = ext_[L.src_t[k]];
                dst.resize((size_t)C * H * W);
                if (!t.data.data) return kTfLiteError;
                nhwc_to_chw(reinterpret_cast<const uint8_t *>(t.data.data), C, H, W,
                            shift_of(t.type), dst.data());
            }

        for (unsigned i = 0; i < layers_.size(); i++) {
            // A CROSS-LAYER KICK covers layers i..kick_end-1 in ONE submit. Its input is
            // the run's first layer's and its output the run's LAST layer's — a run may
            // start at a producer whose own tensor is not what the kick leaves behind.
            if (plan_->kick[i]) {
                const unsigned last = plan_->kick_end[i] - 1u;
                const int8_t *in = plan_->cube_in[i] ? nullptr : operand(i, 0);
                int8_t *out = plan_->cube_out[last] ? nullptr : layers_[last].buf.data();
                const int rc =
                    rocket_conv2d_int8_chain_run_rk3576(fd_, plan_->kick[i], in, out);
                if (rc != ROCKET_OK) {
                    std::fprintf(stderr,
                                 "[rocket/rk3576] the kick at layer %u returned %d\n", i, rc);
                    return kTfLiteError;
                }
                i = last;
                continue;
            }
            if (rocket_graph_is_placement(plan_, i)) continue;
            const int rc = run_one(i);
            if (rc != ROCKET_OK) {
                std::fprintf(stderr, "[rocket/rk3576] layer %u (%s) returned %d\n", i,
                             kind_name(layers_[i].kind), rc);
                return kTfLiteError;
            }
        }

        // The partition's OUTPUTS, back to NHWC.
        for (const Layer &L : layers_) {
            if (!part_out_.count(L.out_t)) continue;
            TfLiteTensor &t = ctx->tensors[L.out_t];
            if (!t.data.data) return kTfLiteError;
            chw_to_nhwc(L.buf.data(), L.oc, L.oh, L.ow, shift_of(t.type),
                        reinterpret_cast<uint8_t *>(t.data.data));
        }
        return kTfLiteOk;
    }

private:
    static const char *kind_name(rocket_graph_kind k) {
        switch (k) {
        case ROCKET_GRAPH_CONV:    return "conv";
        case ROCKET_GRAPH_DWCONV:  return "dwconv";
        case ROCKET_GRAPH_AVGPOOL: return "avgpool";
        case ROCKET_GRAPH_MAXPOOL: return "maxpool";
        case ROCKET_GRAPH_ADD:     return "add";
        case ROCKET_GRAPH_CONCAT:  return "concat";
        default:                   return "host";
        }
    }

    bool has_inside_consumer(size_t i) const {
        for (const Layer &L : layers_)
            for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++)
                if (L.src[k] == (int)i) return true;
        return false;
    }

    // NHWC storage-type bytes -> CHW int8: the rebase and the transpose in one pass.
    static void nhwc_to_chw(const uint8_t *src, int C, int H, int W, int shift,
                            int8_t *dst) {
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++) {
                const uint8_t *r = src + ((size_t)y * W) * C + c;
                int8_t *o = dst + ((size_t)c * H + y) * W;
                for (int x = 0; x < W; x++) o[x] = (int8_t)((int)r[(size_t)x * C] - shift);
            }
    }

    static void chw_to_nhwc(const int8_t *src, unsigned C, unsigned H, unsigned W,
                            int shift, uint8_t *dst) {
        for (unsigned c = 0; c < C; c++)
            for (unsigned y = 0; y < H; y++) {
                const int8_t *r = src + ((size_t)c * H + y) * W;
                uint8_t *o = dst + ((size_t)y * W) * C + c;
                for (unsigned x = 0; x < W; x++)
                    o[(size_t)x * C] = (uint8_t)((int)r[x] + shift);
            }
    }

    // Where operand `k` of layer `i` lives as a row-major CHW tensor.
    const int8_t *operand(unsigned i, int k) {
        const Layer &L = layers_[i];
        if (L.src[k] >= 0) {
            const std::vector<int8_t> &b = layers_[L.src[k]].buf;
            return b.empty() ? nullptr : b.data();
        }
        const auto it = ext_.find(L.src_t[k]);
        return it == ext_.end() ? nullptr : it->second.data();
    }

    int run_one(unsigned i) {
        Layer &L = layers_[i];
        const int ci = plan_->cube_in[i], co = plan_->cube_out[i];
        int8_t *out = co ? nullptr : L.buf.data();

        switch (L.kind) {
        case ROCKET_GRAPH_CONCAT: {
            if (co) return ROCKET_OK;            // wired: the producers wrote the slices
            const size_t px = (size_t)L.oh * L.ow;
            size_t off = 0;
            for (int k = 0; k < ROCKET_GRAPH_MAX_SRC; k++) {
                if (L.src_t[k] < 0) continue;
                const int8_t *s = operand(i, k);
                if (!s) return ROCKET_E_SHAPE;
                const size_t n = (size_t)L.src_c[k] * px;
                std::memcpy(out + off, s, n);
                off += n;
            }
            return ROCKET_OK;
        }
        case ROCKET_GRAPH_AVGPOOL:
        case ROCKET_GRAPH_MAXPOOL:
            return rocket_pool_int8_prepacked_rk3576(fd_, L.pool,
                                                     ci ? nullptr : operand(i, 0), out);
        case ROCKET_GRAPH_ADD: {
            // A WIRED ADD HAS NO HOST OPERANDS AT ALL: its two producers wrote their own
            // slices of the buffer its handle reads as a cube.
            if (ci) return rocket_conv2d_int8_prepacked_rk3576(fd_, L.conv, nullptr, out);
            const unsigned boff = rocket_graph_add_boff(L.oc);
            const size_t px = (size_t)L.oh * L.ow;
            const size_t n = (size_t)L.oc * px;
            const int8_t *a = operand(i, 0), *b = operand(i, 1);
            if (!a || !b) return ROCKET_E_SHAPE;
            cat_.assign((size_t)boff * px + n, 0);
            std::memcpy(cat_.data(), a, n);
            std::memcpy(cat_.data() + (size_t)boff * px, b, n);
            return rocket_conv2d_int8_prepacked_rk3576(fd_, L.conv, cat_.data(), out);
        }
        default:
            return rocket_conv2d_int8_prepacked_rk3576(fd_, L.conv,
                                                       ci ? nullptr : operand(i, 0), out);
        }
    }

    // ---- the lowering, per node -------------------------------------------
    bool lower(TfLiteContext *ctx, const TfLiteRegistration *reg, const TfLiteNode *node,
               const std::vector<std::pair<int, FoldedPad>> &pads, Layer *L) {
        int in_t = node->inputs->data[0];
        int ex_top = -1, ex_bottom = 0, ex_left = -1, ex_right = 0;
        for (const auto &pr : pads)
            if (pr.first == in_t) {
                in_t = pr.second.src_t;
                ex_top = pr.second.top;   ex_bottom = pr.second.bottom;
                ex_left = pr.second.left; ex_right = pr.second.right;
            }
        const TfLiteTensor &in = ctx->tensors[in_t];
        const TfLiteTensor &o = ctx->tensors[node->outputs->data[0]];

        L->in_t = in_t;
        L->out_t = node->outputs->data[0];
        L->src_t[0] = in_t;
        L->src_c[0] = tdim(in, 3);
        if (!quant1(in, &L->in_scale, &L->in_zp)) return false;
        if (!quant1(o, &L->out_scale, &L->out_zp)) return false;
        L->in_zp -= shift_of(in.type);
        L->out_zp -= shift_of(o.type);
        L->ih = (unsigned)tdim(in, 1); L->iw = (unsigned)tdim(in, 2);
        L->ic = (unsigned)tdim(in, 3);
        L->oh = (unsigned)tdim(o, 1);  L->ow = (unsigned)tdim(o, 2);
        L->oc = (unsigned)tdim(o, 3);

        switch (reg->builtin_code) {
        case kTfLiteBuiltinConv2d:
        case kTfLiteBuiltinDepthwiseConv2d: {
            const bool dw = reg->builtin_code == kTfLiteBuiltinDepthwiseConv2d;
            L->kind = dw ? ROCKET_GRAPH_DWCONV : ROCKET_GRAPH_CONV;
            TfLitePadding pad;
            if (dw) {
                const auto *p =
                    reinterpret_cast<const TfLiteDepthwiseConvParams *>(node->builtin_data);
                L->sy = (unsigned)p->stride_height; L->sx = (unsigned)p->stride_width;
                pad = p->padding;
            } else {
                const auto *p =
                    reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);
                L->sy = (unsigned)p->stride_height; L->sx = (unsigned)p->stride_width;
                pad = p->padding;
            }
            const TfLiteTensor &f = ctx->tensors[node->inputs->data[1]];
            const TfLiteTensor &b = ctx->tensors[node->inputs->data[2]];
            L->kh = (unsigned)f.dims->data[1];
            L->kw = (unsigned)f.dims->data[2];
            int oh, ow, ly, lx, ty, tx;
            resolve_pad((int)L->ih, (int)L->kh, (int)L->sy, pad, ex_top, ex_bottom,
                        &oh, &ly, &ty);
            resolve_pad((int)L->iw, (int)L->kw, (int)L->sx, pad, ex_left, ex_right,
                        &ow, &lx, &tx);
            if (oh != (int)L->oh || ow != (int)L->ow) return false;
            L->pl_y = (unsigned)ly; L->pt_y = (unsigned)ty;
            L->pl_x = (unsigned)lx; L->pt_x = (unsigned)tx;
            if (!filter_quant(f, (int)L->oc, L)) return false;
            if (!take_weights(f, dw, L)) return false;
            L->bias.assign(b.data.i32, b.data.i32 + L->oc);
            return true;
        }
        case kTfLiteBuiltinQuantize: {
            // The requantization edge as a 1x1 IDENTITY. A unit weight at unit scale leaves
            // the convolution's own epilogue computing exactly TFLite's kernel.
            //
            // DEPTHWISE where it can be — linear in the channel count where the direct form
            // is quadratic — but a depthwise of four or fewer channels has no form on this
            // part, so a narrow one takes the direct identity instead. That is the same
            // arithmetic and, at four channels, sixteen weights. An SSD box encoding is
            // exactly this shape.
            if (L->ic != L->oc) return false;
            L->kh = L->kw = L->sy = L->sx = 1;
            L->w_scale = 1.f; L->w_zp = 0;
            L->bias.assign(L->oc, 0);
            if (L->ic <= 4) {
                L->kind = ROCKET_GRAPH_CONV;
                L->w.assign((size_t)L->ic * L->ic, 0);
                for (unsigned c = 0; c < L->ic; c++) L->w[(size_t)c * L->ic + c] = 1;
            } else {
                L->kind = ROCKET_GRAPH_DWCONV;
                L->w.assign(L->ic, (int8_t)1);
            }
            return true;
        }
        case kTfLiteBuiltinAveragePool2d:
        case kTfLiteBuiltinMaxPool2d: {
            const auto *p = reinterpret_cast<const TfLitePoolParams *>(node->builtin_data);
            L->avg = reg->builtin_code == kTfLiteBuiltinAveragePool2d;
            L->kind = L->avg ? ROCKET_GRAPH_AVGPOOL : ROCKET_GRAPH_MAXPOOL;
            L->kh = (unsigned)p->filter_height; L->kw = (unsigned)p->filter_width;
            L->sy = (unsigned)p->stride_height; L->sx = (unsigned)p->stride_width;
            int oh, ow, ly, lx, ty, tx;
            resolve_pad((int)L->ih, (int)L->kh, (int)L->sy, p->padding, ex_top, ex_bottom,
                        &oh, &ly, &ty);
            resolve_pad((int)L->iw, (int)L->kw, (int)L->sx, p->padding, ex_left, ex_right,
                        &ow, &lx, &tx);
            if (oh != (int)L->oh || ow != (int)L->ow) return false;
            L->pl_y = (unsigned)ly; L->pt_y = (unsigned)ty;
            L->pl_x = (unsigned)lx; L->pt_x = (unsigned)tx;
            return true;
        }
        case kTfLiteBuiltinAdd: {
            L->kind = ROCKET_GRAPH_ADD;
            const TfLiteTensor &b = ctx->tensors[node->inputs->data[1]];
            L->src_t[1] = node->inputs->data[1];
            L->src_c[1] = tdim(b, 3);
            float bs; int bz;
            if (!quant1(b, &bs, &bz)) return false;
            L->in2_scale = bs;
            L->in2_zp = bz - shift_of(b.type);
            L->ic = L->oc; L->ih = L->oh; L->iw = L->ow;
            return true;
        }
        case kTfLiteBuiltinConcatenation: {
            L->kind = ROCKET_GRAPH_CONCAT;
            for (int k = 0; k < node->inputs->size; k++) {
                L->src_t[k] = node->inputs->data[k];
                L->src_c[k] = tdim(ctx->tensors[L->src_t[k]], 3);
            }
            L->ic = L->oc; L->ih = L->oh; L->iw = L->ow;
            return true;
        }
        default:
            return false;
        }
    }

    // Per-tensor or per-axis filter quantization, rebased into the int8 domain.
    static bool filter_quant(const TfLiteTensor &f, int OC, Layer *L) {
        const int shift = shift_of(f.type);
        if (f.quantization.type == kTfLiteAffineQuantization && f.quantization.params) {
            const auto *aq =
                reinterpret_cast<const TfLiteAffineQuantization *>(f.quantization.params);
            if (!aq->scale || aq->scale->size < 1) return false;
            const int ss = aq->scale->size;
            if (ss != 1 && ss != OC) return false;
            const int zs = aq->zero_point ? aq->zero_point->size : 0;
            if (ss == OC) {
                // PER-AXIS. The weight zero point has to be symmetric: the coefficient
                // group's B term carries one, and every per-axis quantizer emits zero.
                for (int k = 0; k < zs; k++)
                    if (aq->zero_point->data[k] != 0) return false;
                L->w_zp = 0;
                L->w_scale = aq->scale->data[0];
                L->w_scale_oc.assign(aq->scale->data, aq->scale->data + OC);
                return true;
            }
            L->w_scale = aq->scale->data[0];
            L->w_zp = (zs ? aq->zero_point->data[0] : 0) - shift;
            return L->w_scale > 0.f;
        }
        if (f.params.scale <= 0.f) return false;
        L->w_scale = f.params.scale;
        L->w_zp = f.params.zero_point - shift;
        return true;
    }

    static bool take_weights(const TfLiteTensor &f, bool dw, Layer *L) {
        return transpose_weights(f, dw, &L->w);
    }

    // ---- the resident handle ----------------------------------------------
    bool pack(Layer &L) {
        rocket_conv2d_desc d;
        switch (L.kind) {
        case ROCKET_GRAPH_CONCAT:
            return true;                        // placement or a host copy; no program
        case ROCKET_GRAPH_AVGPOOL:
        case ROCKET_GRAPH_MAXPOOL: {
            rocket_pool_desc p;
            pool_desc_of(L, &p);
            if (rocket_pool_int8_rk3576_plan(&p) != ROCKET_OK) return false;
            L.pool = rocket_pool_int8_pack_rk3576(fd_, &p, L.in_zp);
            return L.pool != nullptr;
        }
        case ROCKET_GRAPH_ADD: {
            // The add's weights are the residual lowering's two diagonal blocks, moved to
            // the 16-channel group boundary the cube layout can address. Pure, and a
            // function of the QUANTIZATION alone, so they are built once here.
            const unsigned c = L.oc, boff = rocket_graph_add_boff(c);
            std::vector<int8_t> narrow((size_t)c * 2u * c);
            L.w.assign((size_t)c * (boff + c), 0);
            L.bias.assign(c, 0);
            float ws = 1.f;
            if (rocket_residual_add_weights_rk3576(c, L.in_scale, L.in2_scale, L.in_zp,
                                                   L.in2_zp, narrow.data(), L.bias.data(),
                                                   &ws, nullptr) != ROCKET_OK)
                return false;
            for (unsigned oc = 0; oc < c; oc++) {
                std::memcpy(L.w.data() + (size_t)oc * (boff + c),
                            narrow.data() + (size_t)oc * 2u * c, c);
                std::memcpy(L.w.data() + (size_t)oc * (boff + c) + boff,
                            narrow.data() + (size_t)oc * 2u * c + c, c);
            }
            L.w_scale = ws;
            L.w_zp = 0;
            add_desc_of(L, &d);
            L.conv = rocket_conv2d_int8_pack_rk3576(fd_, &d, L.w.data(), L.bias.data(),
                                                    L.in_scale, ws, nullptr, L.out_scale,
                                                    L.in_zp, 0, L.out_zp);
            return L.conv != nullptr;
        }
        default:
            break;
        }

        // A NARROW INPUT-CHANNEL COUNT: the part has two encodings for an image of four or
        // fewer channels, and which one it takes is ASKED rather than predicted. The
        // packed-image first conv programs ~8x fewer MACs, but it carries geometry bounds
        // the library owns — a non-zero left pad, an output width of exactly iw/stride,
        // that width also a multiple of 16 — so the decision is taken by ATTEMPTING the
        // pack, and a refusal leaves the layer on the direct datapath, which is exact at
        // any geometry and any zero point.
        //
        // Only at a ZERO input zero point: past that the entry materialises its pad
        // columns, its surface is then wider than the caller's plane, and no convolution
        // can read it as a cube — which costs a join, and on the graphs measured on this
        // part more than the encoding buys.
        const bool narrow = L.kind == ROCKET_GRAPH_CONV && L.ic <= 4;
        // A 1x1 has no leading pad to give it, so it can never meet the packed encoding's
        // bounds; asking would cost a pack and print a refusal that says nothing.
        if (narrow && L.in_zp == 0 && L.kh > 1 && L.kw > 1) {
            conv_desc_of(L, &d, 0);
            L.conv = rocket_conv2d_int8_pack_rk3576(
                fd_, &d, L.w.data(), L.bias.data(), L.in_scale, L.w_scale,
                L.w_scale_oc.empty() ? nullptr : L.w_scale_oc.data(), L.out_scale,
                L.in_zp, L.w_zp, L.out_zp);
            if (L.conv) {
                if (profile_)
                    std::fprintf(stderr, "[rocket/rk3576] the packed-image first conv takes "
                                 "ic=%u %ux%u k%u s%u pad %u,%u\n", L.ic, L.ih, L.iw, L.kh,
                                 L.sy, L.pl_y, L.pl_x);
                return true;
            }
        }
        conv_desc_of(L, &d, narrow ? 1 : 0);
        L.conv = rocket_conv2d_int8_pack_rk3576(
            fd_, &d, L.w.data(), L.bias.data(), L.in_scale, L.w_scale,
            L.w_scale_oc.empty() ? nullptr : L.w_scale_oc.data(), L.out_scale,
            L.in_zp, L.w_zp, L.out_zp);
        return L.conv != nullptr;
    }

    bool profile_ = false;
    int fd_ = -1;
    bool planned_ = false;
    std::vector<Layer> layers_;
    std::vector<rocket_graph_layer> graph_;
    rocket_graph_plan *plan_ = nullptr;
    std::map<int, int> producer_;                    // tensor index -> layer index
    std::set<int> part_out_;
    std::map<int, std::vector<int8_t>> ext_;         // partition inputs, CHW
    std::vector<int8_t> cat_;                        // an unwired add's concatenation
};

}  // namespace rocket_rk3576

#endif  // ROCKET_RK3576_NET_H
