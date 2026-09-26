#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
dw_native_e2e.py — the native int8 depthwise route end to end through the delegate, scored
against TFLite's own kernels.

The route's only delegate-level check had been a zero difference against a Mesa Teflon
capture, and that capture is Mesa's uint8 arithmetic on int8 bytes, not the model's function.
This runs rocket-userspace's depthwise fixture model instead (dw_pt.tflite: 64 channels, 8x8,
K3x3, stride 1, per-tensor int8) on its seeded input, twice in one process:

  * LiteRT's reference kernels, which must reproduce the stored litert-out.bin exactly, so the
    fixture and the kernel agree before either is used as the answer;
  * the delegate under native_int8=1 and profile=1, which must claim the node, run it on the
    NPU (its counters), name route=native-dw-int8 on its profile line, and return TFLite's
    output within one count on at most 2% of elements. The part rounds its requant ties to
    even where TFLite rounds them away from zero, so a few outputs one count apart are
    expected, and only those.

    python3 dw_native_e2e.py --delegate build/libtflite_rocket.so \\
        [--fixtures ../rocket-userspace/tests/data/teflon-dw-capture]

Exits 0 on a pass and 1 on any failed check.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

import delegate_counts

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIX = os.path.join(HERE, "..", "..", "rocket-userspace", "tests", "data",
                           "teflon-dw-capture")
C, H, W = 64, 8, 8
MAX_FRAC_OFF_BY_ONE = 0.02


def load_litert():
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
    except ImportError:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
    return Interpreter, load_delegate, OpResolverType


def invoke(interp, x_nhwc):
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x_nhwc)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delegate", required=True)
    ap.add_argument("--fixtures", default=DEFAULT_FIX)
    args = ap.parse_args()

    model = os.path.join(args.fixtures, "dw_pt.tflite")
    x_chw = np.fromfile(os.path.join(args.fixtures, "litert-in.bin"), np.int8)
    y_chw = np.fromfile(os.path.join(args.fixtures, "litert-out.bin"), np.int8)
    if x_chw.size != C * H * W or y_chw.size != C * H * W:
        print(f"FAIL: fixture sizes {x_chw.size}, {y_chw.size}, expected {C * H * W}")
        return 1
    x = x_chw.reshape(C, H, W).transpose(1, 2, 0)[None, ...].copy()     # CHW -> NHWC
    y_ref_stored = y_chw.reshape(C, H, W).transpose(1, 2, 0)[None, ...]

    Interpreter, load_delegate, OpResolverType = load_litert()
    ok = True

    ref = invoke(Interpreter(model_path=model,
                             experimental_op_resolver_type=OpResolverType.BUILTIN_REF), x)
    same = np.array_equal(ref, y_ref_stored)
    print(f"reference kernels reproduce litert-out.bin: {same}")
    ok = ok and same

    # The delegate's profile lines go to the process's stderr from C, so fd 2 is pointed at
    # a file for the arm and restored after it.
    dele = [load_delegate(args.delegate, options={"native_int8": "1", "profile": "1"})]
    c0 = delegate_counts.read(args.delegate)
    log = tempfile.TemporaryFile(mode="w+b")
    saved = os.dup(2)
    sys.stderr.flush()
    os.dup2(log.fileno(), 2)
    try:
        got = invoke(Interpreter(
            model_path=model, experimental_delegates=dele,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES), x)
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
    log.seek(0)
    text = log.read().decode(errors="replace")
    routes = [ln.strip() for ln in text.splitlines() if " route=" in ln]
    for ln in routes:
        print(f"   {ln}")
    named = any("route=native-dw-int8" in ln for ln in routes)
    print(f"the profile names the native depthwise route: {named}")
    ok = ok and named
    ok = delegate_counts.check(c0, delegate_counts.read(args.delegate)) and ok

    d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
    n_off1, n_worse = int((d == 1).sum()), int((d > 1).sum())
    frac = n_off1 / d.size
    good = n_worse == 0 and frac <= MAX_FRAC_OFF_BY_ONE
    print(f"delegate against the reference kernels: {n_off1} of {d.size} off by one "
          f"({100.0 * frac:.2f}%), {n_worse} by more, max {int(d.max())} -> "
          f"{'ok' if good else 'FAIL'}")
    ok = ok and good

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
