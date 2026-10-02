#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
requant_chain.py — how the on-chip requant's within-one difference compounds across a chain
of a real detector's per-tensor quantized layers.

A per-tensor on-chip requant (the RK3588 DPU's OUT_CVT: a 15-bit multiplier and a shift) is
within one of TFLite's (a 31-bit fixed-point multiplier) at a rounding boundary, per op. A
resident inter-op lever keeps each op's on-chip output and feeds it to the next op, so what
the next op sees differs from what TFLite's next op sees. This measures that, per layer.

The model's CONV_2D, DEPTHWISE_CONV_2D and ADD ops are walked in execution order. For each,
TFLite's reference kernels (BUILTIN_REF, every tensor preserved) supply the oracle input and
output. Every conv runs under several requant ARMS:

  tfl   TFLite's own arithmetic re-implemented here (Q31 multiplier, gemmlowp rounding).
        Applied to TFLite's input it must reproduce TFLite's output exactly on every layer;
        that is the check that this script's accumulator, padding and requant are right. On
        the two uint8 detectors it does on every layer but three, where 1-4 elements of a
        layer differ by one (flagged TFL-REIMPL-MISMATCH in the report); the chained `tfl`
        column then shows what one such element does downstream.
  even  the OUT_CVT model the RK3588 ships: MUL/SHIFT from the float scale by the vendor
        derivation, `acc*MUL >> SHIFT` rounded half to even (CVT_ROUND clear), plus the
        output zero point, saturated. tests/requant_model.h, in numpy.
  away  the same with ties away from zero (CVT_ROUND set, ROCKET_OUT_CVT_ROUND=1).
  m16   a candidate derivation the SCALE register can hold [expected, host-only]: a 16-bit
        multiplier rounded to nearest, where the shipped one truncates to 15 bits and adds 1.
  exact the real product acc * (in_s*w_s/out_s) in double, rounded ONCE, ties away. Neither
        kernel computes it; it separates TFLite's double rounding (gemmlowp rounds at the Q31
        high-mul and again at the shift, which at a small shift moves a .38-.49 fraction up)
        from the OUT_CVT multiplier's own error.

A context chain, `tflopt`, is TFLite's optimized builtin kernels (no XNNPACK) run on the same
image with every tensor preserved and scored against the reference kernels per layer: what
TFLite's own kernel variants do to each other along the same chain.

`even` against `away` separates the tie rule from the multiplier derivation as the source of
the within-one; `m16` asks whether a better derivation removes it.

Per layer and arm the script reports two readings, each as elements one away, two or more
away, and the max:
  isolated  the arm applied to TFLite's own input for that op (the per-op within-one);
  chained   the arm applied to its own previous outputs (what a resident chain would see).
ADD is TFLite's integer ADD re-implemented exactly and applied to each arm's chained inputs,
so an ADD adds no requant difference of its own. The chain ends at the ops it does not model
(RESHAPE/QUANTIZE/CONCATENATION, the head), so the last conv outputs are the drift at the end.

Weights are uint8 with ASYMMETRIC zero points on the per-tensor detectors; the on-chip path for
that is DPU_BS_OW_OP (CPEND) = -zw in the int8 domain, measured bit-exact on the depthwise
program (tests/cpend_wzp_probe). The accumulator is the same function either way, so the arms
differ only in the requant.

--device <runner> additionally runs every DEPTHWISE layer that fits one CBUF pass on the NPU
(through rocket-userspace's tests/cpend_wzp_probe in its `layer` mode), fed the `even` arm's
chained input, and reports whether the device output equals the `even` arm's bit for bit.

    python3 requant_chain.py MODEL.tflite --image IMG [--device PATH] [--json OUT]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from ai_edge_litert.interpreter import Interpreter, OpResolverType

ARMS = ("tfl", "exact", "even", "away", "m16")


# ---- TFLite's fixed-point arithmetic ---------------------------------------------------------
def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: (int32 Q31 multiplier, shift), m = q * 2^shift."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    qf = int(math.floor(q * (1 << 31) + 0.5))   # TfLiteRound: half away from zero, q > 0
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return qf, shift


def _trunc_div(v, d):
    return np.where(v >= 0, v // d, -((-v) // d))


def srdhm(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul(a, b), a int64 array in int32 range."""
    ab = a.astype(np.int64) * np.int64(b)
    nudge = np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    return _trunc_div(ab + nudge, np.int64(1 << 31))


def rdbpot(x, e):
    """gemmlowp RoundingDivideByPOT."""
    if e == 0:
        return x
    mask = np.int64((1 << e) - 1)
    rem = x & mask
    thr = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> e) + (rem > thr).astype(np.int64)


def mbqm(x, mult, shift):
    """TFLite MultiplyByQuantizedMultiplier (double rounding)."""
    left = shift if shift > 0 else 0
    right = -shift if shift < 0 else 0
    return rdbpot(srdhm(x * np.int64(1 << left), mult), right)


# ---- the OUT_CVT model -------------------------------------------------------------------------
def outcvt_params(in_s, w_s, out_s):
    """tests/requant_model.h requant_params(), from the float32 scale the emitter computes."""
    cs = np.float32(np.float32(in_s) * np.float32(w_s)) / np.float32(out_s)
    bits = int(np.array([cs], dtype=np.float32).view(np.uint32)[0])
    shift = 127 + 31 - 32 - (bits >> 23) + 16 - 1
    m = ((bits >> 9) & 0x7FFF) + 1
    if m < (1 << 14):
        m |= (1 << 14)
    return m, shift


def m16_params(in_s, w_s, out_s):
    """A 16-bit multiplier rounded to nearest [expected: the SCALE field is [15:0]]."""
    cs = float(np.float32(np.float32(in_s) * np.float32(w_s)) / np.float32(out_s))
    e = 0
    while cs * (2.0 ** e) < (1 << 15):
        e += 1
    while cs * (2.0 ** e) >= (1 << 16):
        e -= 1
    m = int(math.floor(cs * (2.0 ** e) + 0.5))
    if m == (1 << 16):
        m //= 2
        e -= 1
    return m, e


def round_shift(p, shift, away):
    if shift == 0:
        return p
    half = np.int64(1 << (shift - 1))
    v = (p + half) >> shift
    rem = p & np.int64((1 << shift) - 1)
    tie = rem == half
    if away:
        v = v - (tie & (p < 0)).astype(np.int64)
    else:
        v = v - (tie & ((v & 1) == 1)).astype(np.int64)
    return v


# ---- the accumulator ---------------------------------------------------------------------------
def same_or_valid(ih, iw, oh, ow, kh, kw):
    """Every (stride, pads) the shapes allow, SAME first, then VALID, strides 1..3. More than
    one can fit (5x5 -> 3x3 at k3 is VALID s1 or SAME s2); the caller keeps the one under which
    TFLite's own arithmetic reproduces TFLite's output."""
    out = []
    for s in (1, 2, 3):
        for mode in ("SAME", "VALID"):
            if mode == "SAME":
                eh, ew = -(-ih // s), -(-iw // s)
            else:
                eh, ew = -(-(ih - kh + 1) // s), -(-(iw - kw + 1) // s)
            if (eh, ew) == (oh, ow):
                if mode == "SAME":
                    ph = max((oh - 1) * s + kh - ih, 0)
                    pw = max((ow - 1) * s + kw - iw, 0)
                    out.append((s, (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)))
                else:
                    out.append((s, (0, 0, 0, 0)))
    if not out:
        raise ValueError(f"cannot infer stride/pad for {ih}x{iw} -> {oh}x{ow} k{kh}x{kw}")
    return out


def pad_input(x, zx, pads):
    """x [H,W,C] int64, padded with the input zero point (a padded tap contributes nothing)."""
    pt, pb, pl, pr = pads
    return np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=zx)


def conv_acc(x_u8, w, bias, zx, zw, s, pads, oh, ow, depthwise):
    """TFLite's int32 accumulator, exact (float64 BLAS on integers below 2^53)."""
    x = pad_input(x_u8[0].astype(np.int64), zx, pads) - zx      # [H,W,IC]
    kh, kw = w.shape[1], w.shape[2]
    wz = w.astype(np.int64) - zw                                 # conv [OC,KH,KW,IC]; dw [1,KH,KW,C]
    if depthwise:
        C = wz.shape[3]
        acc = np.zeros((oh, ow, C), dtype=np.int64)
        for i in range(kh):
            for j in range(kw):
                acc += x[i:i + s * (oh - 1) + 1:s, j:j + s * (ow - 1) + 1:s, :] * wz[0, i, j, :]
    else:
        OC, IC = wz.shape[0], wz.shape[3]
        cols = np.empty((oh, ow, kh, kw, IC), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :] = x[i:i + s * (oh - 1) + 1:s, j:j + s * (ow - 1) + 1:s, :]
        acc = (cols.reshape(oh * ow, kh * kw * IC) @
               wz.reshape(OC, kh * kw * IC).T.astype(np.float64)).reshape(oh, ow, OC)
        acc = np.rint(acc).astype(np.int64)
    return acc + bias.astype(np.int64)


def requant(acc, arm, q):
    """acc -> uint8 under one arm. q: dict of the layer's quant parameters."""
    zo, lo, hi = q["zo"], q["act_lo"], q["act_hi"]
    if arm == "tfl":
        v = mbqm(acc, q["tfl_m"], q["tfl_s"]) + zo
    elif arm in ("even", "away"):
        v = round_shift(acc * np.int64(q["cvt_m"]), q["cvt_s"], arm == "away") + zo
    elif arm == "m16":
        v = round_shift(acc * np.int64(q["m16_m"]), q["m16_s"], False) + zo
    elif arm == "exact":
        y = acc.astype(np.float64) * q["cs64"]
        v = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64) + zo
    else:
        raise ValueError(arm)
    return np.clip(v, lo, hi).astype(np.uint8)[None, ...]


# ---- ADD ---------------------------------------------------------------------------------------
def add_params(s1, s2, so):
    left = 20
    twice = 2.0 * float(max(np.float32(s1), np.float32(s2)))
    m1 = quantize_multiplier(float(np.float32(s1)) / twice)
    m2 = quantize_multiplier(float(np.float32(s2)) / twice)
    mo = quantize_multiplier(twice / ((1 << left) * float(np.float32(so))))
    return left, m1, m2, mo


def tfl_add(a, b, qa, qb, qo, lo, hi, prm):
    left, (m1, s1), (m2, s2), (mo, so) = prm
    a1 = (a.astype(np.int64) - qa) * np.int64(1 << left)
    b1 = (b.astype(np.int64) - qb) * np.int64(1 << left)
    sa = rdbpot(srdhm(a1, m1), -s1)
    sb = rdbpot(srdhm(b1, m2), -s2)
    out = rdbpot(srdhm(sa + sb, mo), -so) + qo
    return np.clip(out, lo, hi).astype(np.uint8)


# ---- scoring -----------------------------------------------------------------------------------
def score(got, want):
    d = np.abs(got.astype(np.int64) - want.astype(np.int64))
    return {"n": int(d.size), "off1": int((d == 1).sum()), "off2p": int((d >= 2).sum()),
            "max": int(d.max()) if d.size else 0, "mean": float(d.mean()) if d.size else 0.0}


def load_image(path, h, w):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)[None, ...]


# ---- the device runner -------------------------------------------------------------------------
def device_dw(runner, x_u8, w, bias, zx, zw, zo, in_s, w_s, out_s, s, pads, oh, ow, workdir):
    """Run one depthwise layer on the NPU. The pad is materialized here (value zx), so the
    device sees a pad-0 program; channels are chunked by the runner. Returns uint8 [1,OH,OW,C]
    or None when the layer does not fit one pass."""
    xp = pad_input(x_u8[0].astype(np.int64), zx, pads)            # [Hp,Wp,C]
    Hp, Wp, C = xp.shape
    kh, kw = w.shape[1], w.shape[2]
    # the int8 domain: u - 128, zero points - 128
    xi = (xp - 128).astype(np.int8).transpose(2, 0, 1).copy()     # [C,Hp,Wp]
    wi = (w[0].astype(np.int64) - 128).astype(np.int8).transpose(2, 0, 1).copy()  # [C,KH,KW]
    f_in, f_w, f_b, f_o = (os.path.join(workdir, n) for n in ("in.bin", "w.bin", "b.bin", "o.bin"))
    xi.tofile(f_in); wi.tofile(f_w); bias.astype(np.int32).tofile(f_b)
    args = [runner, "layer", str(C), str(Hp), str(Wp), str(kh), str(kw), str(s),
            repr(float(in_s)), repr(float(w_s)), repr(float(out_s)),
            str(zx - 128), str(zw - 128), str(zo - 128), f_in, f_w, f_b, f_o]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 3:
        return None, r.stdout.strip()
    if r.returncode != 0:
        raise RuntimeError(f"runner rc {r.returncode}: {r.stdout} {r.stderr}")
    o = np.fromfile(f_o, dtype=np.int8).reshape(C, oh, ow).astype(np.int64) + 128
    return o.transpose(1, 2, 0).astype(np.uint8)[None, ...], r.stdout.strip()


# ---- main --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None, help="path to cpend_wzp_probe (layer mode)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    it = Interpreter(model_path=args.model, experimental_preserve_all_tensors=True,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x0 = load_image(args.image, int(inp["shape"][1]), int(inp["shape"][2]))
    it.set_tensor(inp["index"], x0)
    it.invoke()
    td = {t["index"]: t for t in it.get_tensor_details()}
    opt = Interpreter(model_path=args.model, experimental_preserve_all_tensors=True,
                      experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    opt.allocate_tensors()
    opt.set_tensor(opt.get_input_details()[0]["index"], x0)
    opt.invoke()

    def qp(i):
        q = td[i]["quantization_parameters"]
        return float(q["scales"][0]), int(q["zero_points"][0]), len(q["scales"])

    def T(i):
        return it.get_tensor(i)

    chained = {a: {inp["index"]: x0} for a in ARMS}
    rows = []
    workdir = tempfile.mkdtemp(prefix="rqchain-")
    dev_rows = []
    for o in it._get_ops_details():
        name = o["op_name"]
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "ADD"):
            continue
        ins = [i for i in o["inputs"] if i >= 0]
        out = o["outputs"][0]
        if td[out]["dtype"] != np.uint8 and td[out]["dtype"] != np.int8:
            continue
        want = T(out)
        row = {"op": int(o["index"]), "type": name, "shape": [int(v) for v in td[out]["shape"]]}
        if name == "ADD":
            if opt is not None:
                row["tflopt"] = score(opt.get_tensor(out), want)
            (sa, za, _), (sb, zb, _), (so, zo, _) = qp(ins[0]), qp(ins[1]), qp(out)
            prm = add_params(sa, sb, so)
            iso = tfl_add(T(ins[0]), T(ins[1]), za, zb, zo, 0, 255, prm)
            row["add_iso"] = score(iso, want)
            for a in ARMS:
                ca = chained[a].get(ins[0], T(ins[0]))
                cb = chained[a].get(ins[1], T(ins[1]))
                if ins[0] not in chained[a] or ins[1] not in chained[a]:
                    row["broken"] = True
                y = tfl_add(ca, cb, za, zb, zo, 0, 255, prm)
                chained[a][out] = y
                row[a] = {"iso": row["add_iso"], "chain": score(y, want)}
            rows.append(row)
            continue

        depthwise = name == "DEPTHWISE_CONV_2D"
        xin, wt, bs = ins[0], ins[1], ins[2]
        (in_s, zx, _), (w_s, zw, nws), (out_s, zo, _) = qp(xin), qp(wt), qp(out)
        if nws != 1:
            row["skip"] = "per-axis"
            rows.append(row)
            continue
        w = T(wt)
        bias = T(bs).astype(np.int64)
        ih, iw = int(td[xin]["shape"][1]), int(td[xin]["shape"][2])
        oh, ow = int(td[out]["shape"][1]), int(td[out]["shape"][2])
        cands = same_or_valid(ih, iw, oh, ow, int(w.shape[1]), int(w.shape[2]))
        tm, ts = quantize_multiplier(float(np.float32(in_s)) * float(np.float32(w_s)) /
                                     float(np.float32(out_s)))
        cm, cs = outcvt_params(in_s, w_s, out_s)
        mm, ms = m16_params(in_s, w_s, out_s)
        q = {"zo": zo, "tfl_m": tm, "tfl_s": ts, "cvt_m": cm, "cvt_s": cs, "m16_m": mm, "m16_s": ms,
             "cs64": float(np.float32(in_s)) * float(np.float32(w_s)) / float(np.float32(out_s))}
        # the geometry and the fused activation: the pair under which TFLite's own arithmetic
        # reproduces TFLite's output
        act = None
        for s, pads in cands:
            acc_iso = conv_acc(T(xin), w, bias, zx, zw, s, pads, oh, ow, depthwise)
            for lo, hi, nm in ((0, 255, "none"), (zo, min(255, zo + int(round(6.0 / out_s))), "relu6"),
                               (zo, 255, "relu")):
                q["act_lo"], q["act_hi"] = max(0, lo), hi
                if np.array_equal(requant(acc_iso, "tfl", q), want):
                    act = nm
                    break
            if act:
                break
        if act is None:
            s, pads = cands[0]
            acc_iso = conv_acc(T(xin), w, bias, zx, zw, s, pads, oh, ow, depthwise)
            q["act_lo"], q["act_hi"] = 0, 255
            act = "UNMATCHED"
        row.update({"K": [int(w.shape[1]), int(w.shape[2])], "s": s, "act": act, "zw": zw,
                    "cvt": [cm, cs], "m16": [mm, ms], "tflq": [tm, ts], "cs64": q["cs64"]})
        if opt is not None:
            row["tflopt"] = score(opt.get_tensor(out), want)
        for a in ARMS:
            iso = requant(acc_iso, a, q)
            if xin in chained[a]:
                acc_c = conv_acc(chained[a][xin], w, bias, zx, zw, s, pads, oh, ow, depthwise)
                y = requant(acc_c, a, q)
            else:
                row["broken"] = True
                y = iso
            chained[a][out] = y
            row[a] = {"iso": score(iso, want), "chain": score(y, want)}
        if args.device and depthwise:
            xd = chained["even"][xin]
            got, msg = device_dw(args.device, xd, w, bias.astype(np.int32), zx, zw, zo,
                                 in_s, w_s, out_s, s, pads, oh, ow, workdir)
            if got is None:
                dev_rows.append({"op": row["op"], "ran": False, "why": msg})
            else:
                dev_rows.append({"op": row["op"], "ran": True,
                                 "vs_even": score(got, chained["even"][out]),
                                 "vs_tflite": score(got, want), "runner": msg})
        rows.append(row)

    # ---- report -----------------------------------------------------------------------------
    print(f"model {os.path.basename(args.model)}  image {os.path.basename(args.image)}  "
          f"input {tuple(int(v) for v in inp['shape'])}")
    print("  each cell: elements off by one / off by two or more / max |diff|, against TFLite's")
    print("  reference kernels. iso = the arm on TFLite's input; chain = on its own outputs.")
    print("  op  type shape          K s act   q31shift | exact iso  | even iso   | even chain        "
          "| exact chain       | m16 chain         | tflopt chain")
    for r in rows:
        if "skip" in r:
            print(f"  {r['op']:3d} {r['type'][:4]} skipped ({r['skip']})")
            continue
        t = r["type"][:4]
        shp = "x".join(str(v) for v in r["shape"][1:])
        f = lambda c: f"{c['off1']:6d}/{c['off2p']:5d}/{c['max']:3d}"
        g = lambda c: f"{c['off1']:6d}/{c['max']:1d}"
        if r["type"] == "ADD":
            lead = f"  {r['op']:3d} {t}  {shp:<14} - - -        -   |"
        else:
            lead = (f"  {r['op']:3d} {t} {shp:<14} {r['K'][0]} {r['s']} {r['act']:<5} "
                    f"{r['tflq'][1]:4d}    |")
        cells = (f" {g(r['exact']['iso'])} | {g(r['even']['iso'])} | {f(r['even']['chain'])} | "
                 f"{f(r['exact']['chain'])} | {f(r['m16']['chain'])} | "
                 + (f(r['tflopt']) if 'tflopt' in r else '   -'))
        flag = ""
        if r["type"] != "ADD" and (r["tfl"]["iso"]["off1"] or r["tfl"]["iso"]["off2p"]):
            flag += " TFL-REIMPL-MISMATCH"
        if r["type"] == "ADD" and (r["add_iso"]["off1"] or r["add_iso"]["off2p"]):
            flag += " ADD-REIMPL-MISMATCH"
        if r.get("broken"):
            flag += " (chain broken)"
        print(lead + cells + flag)
    print("  totals over conv layers (elements):")
    convs = [r for r in rows if r["type"] != "ADD" and "skip" not in r]
    n = sum(r["even"]["iso"]["n"] for r in convs)
    for a in ARMS:
        io1 = sum(r[a]["iso"]["off1"] for r in convs)
        io2 = sum(r[a]["iso"]["off2p"] for r in convs)
        co1 = sum(r[a]["chain"]["off1"] for r in convs)
        co2 = sum(r[a]["chain"]["off2p"] for r in convs)
        cmx = max(r[a]["chain"]["max"] for r in convs)
        print(f"    {a:>6}: n {n}  isolated off1 {io1} ({100.0 * io1 / n:.3f}%) off2+ {io2}  |  "
              f"chained off1 {co1} ({100.0 * co1 / n:.3f}%) off2+ {co2} ({100.0 * co2 / n:.3f}%) "
              f"max {cmx}")
    if all("tflopt" in r for r in convs):
        o1 = sum(r["tflopt"]["off1"] for r in convs)
        o2 = sum(r["tflopt"]["off2p"] for r in convs)
        print(f"    tflopt: n {n}  chained off1 {o1} ({100.0 * o1 / n:.3f}%) off2+ {o2} "
              f"({100.0 * o2 / n:.3f}%) max {max(r['tflopt']['max'] for r in convs)}")
    if dev_rows:
        print("  device depthwise layers (fed the `even` arm's chained input):")
        for d in dev_rows:
            if not d["ran"]:
                print(f"    op {d['op']:3d}: not run ({d['why']})")
            else:
                e, t = d["vs_even"], d["vs_tflite"]
                print(f"    op {d['op']:3d}: vs even-arm model: off1 {e['off1']} off2+ {e['off2p']} "
                      f"max {e['max']} of {e['n']} | vs TFLite: off1 {t['off1']} off2+ {t['off2p']} "
                      f"max {t['max']}  [{d['runner']}]")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "device": dev_rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
