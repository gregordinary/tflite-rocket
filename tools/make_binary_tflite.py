#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_binary_tflite.py — emit a (conv_a, conv_b) -> MAXIMUM|MINIMUM|MUL|SUB|DIV -> conv .tflite.

MAXIMUM (TFLite builtin 55) / MINIMUM (57) are standalone two-input elementwise
builtins (no fused activation) the delegate claims as NodeKind::Add with op set to
ROCKET_BINOP_MAX/MIN. Like the ADD residual generator this sandwiches the binary op
between convs so the whole graph is ONE contiguous partition — `run_delegate.py
--option profile=1` then prints conv / add(host, op=max|min) / conv in a single
partition (the un-spill + contiguity win). The trailing conv also keeps the binary
op's output an INTERNAL tensor (so --compare exercises the delegate-managed buffer).

float / int8 / uint8. Needs TensorFlow (the converter).
Usage:
    python3 make_binary_tflite.py --op maximum --dtype float
    python3 make_binary_tflite.py --op minimum --dtype int8
    python3 make_binary_tflite.py --op maximum --dtype uint8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["maximum", "minimum", "mul", "sub", "div"], default="maximum")
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps the 1x1 on the NPU)")
    ap.add_argument("--hw", type=int, default=16, help="input H=W")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    a = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(inp)
    b = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(inp)
    if args.op in ("maximum", "minimum"):
        y = (tf.keras.layers.Maximum() if args.op == "maximum" else tf.keras.layers.Minimum())([a, b])
    elif args.op == "mul":
        y = tf.keras.layers.Multiply()([a, b])
    elif args.op == "sub":
        y = tf.keras.layers.Subtract()([a, b])
    else:  # div — no keras layer; plain elementwise division (TFLite DIV builtin)
        y = tf.keras.layers.Lambda(lambda t: t[0] / t[1])([a, b])
    y = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(y)
    model = tf.keras.Model(inp, y)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.dtype in ("int8", "uint8"):
        rng = np.random.default_rng(0)
        def rep():
            for _ in range(64):
                yield [rng.standard_normal((1, args.hw, args.hw, args.c)).astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        t = tf.int8 if args.dtype == "int8" else tf.uint8
        conv.inference_input_type = t
        conv.inference_output_type = t

    tflite = conv.convert()
    out = args.out or f"bin_{args.op}_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
