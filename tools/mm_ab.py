#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""mm_ab.py — A/B for the int8/uint8 1x1->matmul routing (perf Step 1).

Two checks:
  1. ACCURACY vs CPU TFLite (the ground truth): runs the delegate with the 1x1 convs on
     the resident int8 matmul (mm_int8=1) vs the native conv pool (mm_int8=0), and reports
     each one's max/mean |delegate - CPU| per output. Both the matmul and the conv path are
     bit-exact to the int64 reference (tests/mm_vs_conv_acc.c), so mm_int8=1 and mm_int8=0
     both match CPU, and on matmul-eligible shapes (M%4||M==1) the two agree byte-for-byte.
  2. nthreads=1 vs N (mm_int8=1) BYTE-IDENTICAL: the matmul is worker-count independent.

Usage: mm_ab.py <model.tflite> <delegate.so> [--nthreads N]
"""
import argparse, sys
import numpy as np

def load():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
        return Interpreter, load_delegate, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except ImportError:
        import tensorflow as tf
        return (tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

def run(Interp, ndr, model, feeds, load_delegate=None, opts=None):
    dl = [load_delegate(opts["__so__"], options={k: v for k, v in opts.items() if k != "__so__"})] \
         if opts else None
    kw = {"experimental_op_resolver_type": ndr} if opts else {}
    it = Interp(model_path=model, experimental_delegates=dl, **kw)
    it.allocate_tensors()
    for d in it.get_input_details():
        it.set_tensor(d["index"], feeds[d["index"]])
    it.invoke()
    return [it.get_tensor(d["index"]).copy() for d in it.get_output_details()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("delegate")
    ap.add_argument("--nthreads", default="4")
    args = ap.parse_args()
    Interp, load_delegate, ndr = load()
    base = Interp(model_path=args.model); base.allocate_tensors()
    rng = np.random.default_rng(0)
    feeds = {}
    for d in base.get_input_details():
        dt = d["dtype"]
        feeds[d["index"]] = (rng.standard_normal(d["shape"]).astype(np.float32) if dt == np.float32
                             else rng.integers(np.iinfo(dt).min, np.iinfo(dt).max + 1,
                                               size=d["shape"], dtype=dt))
    so = args.delegate
    base_opt = {"__so__": so, "native_int8": "1", "nthreads": args.nthreads}
    cpu = run(Interp, ndr, args.model, feeds)                                   # CPU TFLite ref
    on  = run(Interp, ndr, args.model, feeds, load_delegate, {**base_opt, "mm_int8": "1"})
    off = run(Interp, ndr, args.model, feeds, load_delegate, {**base_opt, "mm_int8": "0"})
    nt1 = run(Interp, ndr, args.model, feeds, load_delegate,
              {**base_opt, "mm_int8": "1", "nthreads": "1"})

    print("  accuracy vs CPU TFLite (lower = more correct; matmul should win):")
    win = 0
    for i, (o, f, c) in enumerate(zip(on, off, cpu)):
        do = np.abs(o.astype(np.float64) - c.astype(np.float64))
        df = np.abs(f.astype(np.float64) - c.astype(np.float64))
        flag = "matmul closer" if do.mean() <= df.mean() else "CONV closer (!)"
        win += do.mean() <= df.mean()
        print(f"    out[{i}]: mm=1 mean={do.mean():.5g} max={do.max():.4g} | "
              f"mm=0 mean={df.mean():.5g} max={df.max():.4g}  -> {flag}")
    ntok = all(np.array_equal(a, b) for a, b in zip(nt1, on))
    print(f"  nthreads=1 vs {args.nthreads} (mm on): "
          f"{'BYTE-IDENTICAL -> PASS' if ntok else 'DIFFERS -> FAIL'}")
    ok = (win == len(on)) and ntok
    print("==== %s ====" % ("PASS (matmul >= conv on every output, nt-independent)" if ok else "REVIEW"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
