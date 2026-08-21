#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
rk3576_net_ab.py — run a quantized TFLite classifier with and without the delegate on an
RK3576, and score the delegated answer against TFLite's own.

WHAT IT ASSERTS, AND WHY EACH IS THE RIGHT SHAPE:

  the LABEL. Top-1 must agree and the whole top-5 is reported. That is the assertion a
  classifier can carry end to end; the per-layer bit-exactness against a CPU model of the
  part's arithmetic is rk3576_net_gate's job and needs the blob's goldens.

  the DISTANCE, reported and not asserted. This DPU's requant is not TFLite's — a 15-bit
  multiplier rounding ties to even against TFLite's 31 bits rounding half away from zero —
  so a small population one count apart is the EXPECTED reading, and it compounds down a
  chain because layer n+1 is fed the part's own output rather than TFLite's. Asserting it
  away would be asserting that this chip is a different chip.

  the WALL, warm. The first invoke pays the plan and the BO pool; a graph's steady state is
  from the second on, so the reported figure is the median of the warm ones.

The reference arm runs the SAME interpreter with no delegate, so the two differ in the
delegate alone rather than in the framework.

  python3 rk3576_net_ab.py model.tflite --delegate build/libtflite_rocket.so \
      --image grace_hopper.bmp --labels labels.txt --iters 20
"""
import argparse
import statistics
import sys
import time

import numpy as np


def load_tflite():
    # BUILTIN_WITHOUT_DEFAULT_DELEGATES stops XNNPACK claiming the convolutions before the
    # external delegate gets a turn.
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
    return Interpreter, load_delegate, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


def feed(detail, image):
    _, IH, IW, IC = [int(v) for v in detail["shape"]]
    if image:
        from PIL import Image
        img = Image.open(image).convert("RGB").resize((IW, IH), Image.BILINEAR)
        a = np.asarray(img, dtype=np.uint8)
    else:
        a = np.random.default_rng(0).integers(0, 256, size=(IH, IW, IC), dtype=np.uint8)
    # THE NETWORK INPUT IS THE IMAGE BYTES reinterpreted in the model's storage type — a
    # convention about the model, not a derivation from its input quantization.
    if detail["dtype"] == np.int8:
        return (a.astype(np.int16) - 128).astype(np.int8)[None, ...]
    return a[None, ...]


def run(Interpreter, model, delegates, ndr, x, iters):
    kw = {}
    if delegates:
        kw["experimental_op_resolver_type"] = ndr
    interp = Interpreter(model_path=model, experimental_delegates=delegates, **kw)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], x)
    walls = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        interp.invoke()
        walls.append((time.perf_counter() - t0) * 1e3)
    outs = [interp.get_tensor(d["index"]).reshape(-1).copy()
            for d in interp.get_output_details()]
    return outs, walls


def top5(v):
    return list(np.argsort(-v.astype(np.int32))[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--delegate", required=True)
    ap.add_argument("--option", action="append", default=[])
    ap.add_argument("--image")
    ap.add_argument("--labels")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    Interpreter, load_delegate, ndr = load_tflite()
    opts = dict(kv.split("=", 1) for kv in args.option)

    base = Interpreter(model_path=args.model)
    base.allocate_tensors()
    x = feed(base.get_input_details()[0], args.image)

    ref, ref_walls = run(Interpreter, args.model, None, ndr, x, args.iters)
    dele = [load_delegate(args.delegate, options=opts)]
    got, walls = run(Interpreter, args.model, dele, ndr, x, args.iters)

    print(f"\n{args.model}")
    warm = lambda w: statistics.median(w[1:]) if len(w) > 1 else w[0]
    print(f"   wall: delegate {warm(walls):.2f} ms, CPU {warm(ref_walls):.2f} ms "
          f"({warm(ref_walls) / warm(walls):.2f}x), warm median of {args.iters - 1}")

    for k, (g, r) in enumerate(zip(got, ref)):
        d = np.abs(g.astype(np.float64) - r.astype(np.float64))
        print(f"   output[{k}]: {d.size} elements, differing {int((d != 0).sum())} "
              f"({100.0 * (d != 0).sum() / d.size:.4f}%), max {d.max():g}")

    # THE LABEL is the end-to-end assertion, and only a classifier has one. A detector's
    # outputs are an NMS-ordered list, so a single count of drift can reorder or drop a
    # box: scoring one is a mAP question and not this script's.
    if len(got) != 1 or got[0].size < 100:
        print("   (multi-output or short: no top-5 to assert — the per-output distance "
              "above is what this reports)")
        return 0

    names = None
    if args.labels:
        names = open(args.labels).read().splitlines()
        if len(names) == len(ref[0]) + 1:
            names = names[1:]

    def label(i):
        return f"{names[i]} ({i})" if names else str(i)

    m, r5 = top5(got[0]), top5(ref[0])
    print("   NPU    top-5 " + ", ".join(label(i) for i in m))
    print("   TFLite top-5 " + ", ".join(label(i) for i in r5))
    if m[0] != r5[0]:
        print("   TOP-1 DISAGREES")
        return 1
    print("   top-1 agrees" + (", and so does the whole top-5" if m == r5
                               else " (the tail of the top-5 reorders)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
