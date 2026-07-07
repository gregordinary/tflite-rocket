#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
run_delegate.py — load a .tflite with an external delegate and run one inference;
optionally A/B it against the plain-CPU run.

Drives ANY TFLite external delegate .so: Mesa Teflon (libteflon.so) or our own
(libtflite_rocket.so). Two jobs:
  * the Mesa depthwise CAPTURE — run the minimal model through libteflon with the
    dump env set (this script just drives the inference; ROCKET_DEBUG=dump_bos in
    the environment makes Mesa write the mesa-*.bin BOs to the CWD), and
  * the standing delegate VALIDATION (README roadmap) — --compare runs the model
    with and without the delegate and reports the output delta vs plain CPU.

Works with either `tflite_runtime` or full `tensorflow` (whichever imports).

Examples:
  # capture (dump files land in the CWD; see the Mesa build notes)
  cd /tmp/dwcap && ROCKET_DEBUG=dump_bos \\
    python3 run_delegate.py dw_c64_8x8_3x3_s1.tflite --delegate /path/to/libteflon.so

  # validate our delegate's output vs CPU on a real detector
  python3 run_delegate.py ssd.tflite --delegate ./build/libtflite_rocket.so \\
      --option profile=1 --compare
"""
import argparse
import time
import numpy as np


def parse_options(pairs, parser):
    """Parse repeated --option k=v entries into a dict, erroring cleanly on a
    malformed entry (missing '=') instead of raising a bare ValueError."""
    opts = {}
    for kv in pairs:
        if "=" not in kv:
            parser.error(f"--option must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        opts[k] = v
    return opts

def load_tflite():
    # Returns (Interpreter, load_delegate, no_default_resolver). The resolver type
    # BUILTIN_WITHOUT_DEFAULT_DELEGATES stops TFLite from auto-applying XNNPACK during
    # construction — otherwise XNNPACK claims the conv before our external delegate
    # (Teflon / rocket) gets a turn, and the op never reaches the NPU.
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
        return Interpreter, load_delegate, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except ImportError:
        pass
    try:
        # LiteRT (the maintained successor to tflite_runtime; the only wheel on recent
        # Python). Same interpreter + external-delegate + no-default-resolver API.
        from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
        return Interpreter, load_delegate, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except ImportError:
        pass
    import tensorflow as tf
    try:
        ndr = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except AttributeError:
        ndr = None
    return tf.lite.Interpreter, tf.lite.experimental.load_delegate, ndr

def make_input(detail, rng):
    shape = detail["shape"]
    dt = detail["dtype"]
    if dt == np.float32:
        return rng.standard_normal(shape).astype(np.float32)
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max + 1, size=shape, dtype=dt)

def make_interp(Interpreter, model, delegates, no_default_resolver):
    kw = {}
    # disable default delegates only when WE supply one, so the external delegate
    # isn't pre-empted by XNNPACK; the plain-CPU reference run keeps the default.
    if delegates and no_default_resolver is not None:
        kw["experimental_op_resolver_type"] = no_default_resolver
    return Interpreter(model_path=model, experimental_delegates=delegates, **kw)

def run(Interpreter, model, delegates, feeds, ndr, iters=1):
    interp = make_interp(Interpreter, model, delegates, ndr)
    interp.allocate_tensors()
    ins, outs = interp.get_input_details(), interp.get_output_details()
    for d in ins:
        interp.set_tensor(d["index"], feeds[d["index"]])
    # Loop invoke() within ONE process so the resident state warms up: the delegate's
    # weights are packed in Prepare (before the first invoke), but the conv BO pool only
    # grows on the first invoke, so the per-op profile is steady-state from invoke #2 on.
    # Re-running the script does NOT warm anything (a fresh interpreter/delegate per run).
    for it in range(max(1, iters)):
        if iters > 1:
            tag = "WARM" if it == iters - 1 else "cold" if it == 0 else "..."
            print(f"--- invoke {it + 1}/{iters} ({tag}) ---", flush=True)
        t0 = time.perf_counter()
        interp.invoke()
        if iters > 1:
            print(f"--- invoke {it + 1}/{iters}: {(time.perf_counter() - t0) * 1e3:.3f} ms wall ---",
                  flush=True)
    return [interp.get_tensor(d["index"]).copy() for d in outs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--delegate", help="path to the external delegate .so")
    ap.add_argument("--option", action="append", default=[],
                    help="delegate option key=value (repeatable), e.g. profile=1")
    ap.add_argument("--compare", action="store_true",
                    help="also run plain-CPU and report max|delegate - CPU|")
    ap.add_argument("--iters", type=int, default=1,
                    help="invoke the delegated model N times in one process; the LAST "
                         "iteration's per-op profile is the warm steady-state (use >=2 to "
                         "see the resident-BO win, which is masked on the first invoke)")
    ap.add_argument("--save", metavar="PATH",
                    help="np.savez the delegate outputs to PATH (.npz) — diff two saves to "
                         "assert a change is byte-identical to the pre-change delegate output")
    args = ap.parse_args()

    Interpreter, load_delegate, ndr = load_tflite()

    # shared random input so the two runs are comparable
    base = Interpreter(model_path=args.model)
    base.allocate_tensors()
    rng = np.random.default_rng(0)
    feeds = {d["index"]: make_input(d, rng) for d in base.get_input_details()}

    delegates = None
    if args.delegate:
        opts = parse_options(args.option, ap)
        delegates = [load_delegate(args.delegate, options=opts)]
        print(f"loaded delegate {args.delegate} options={opts}")

    got = run(Interpreter, args.model, delegates, feeds, ndr, iters=args.iters)
    print(f"ran {args.model}: {len(got)} output(s); "
          f"shapes {[o.shape for o in got]} dtypes {[o.dtype for o in got]}")

    if args.save:
        np.savez(args.save, **{f"out{i}": o for i, o in enumerate(got)})
        print(f"  saved {len(got)} delegate output(s) to {args.save}")

    if args.compare:
        ref = run(Interpreter, args.model, None, feeds, ndr)   # CPU reference: one invoke
        for i, (g, r) in enumerate(zip(got, ref)):
            d = np.abs(g.astype(np.float64) - r.astype(np.float64))
            print(f"  output[{i}]: max|delegate-CPU|={d.max():.6g} mean={d.mean():.6g}")

if __name__ == "__main__":
    main()
