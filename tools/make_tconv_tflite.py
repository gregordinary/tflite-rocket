#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_tconv_tflite.py — emit a  conv -> TRANSPOSE_CONV -> conv  .tflite. The
TRANSPOSE_CONV (TFLite builtin 67) is the segmentation / decoder learned-upsample the
delegate claims as NodeKind::TConv (exact float host scatter kernel; the NPU route is a
follow-on). Sandwiched between convs so the whole graph is one partition.

float only (the delegate's v1 TransposeConv path; int8 stays on CPU).
Usage:
    python3 make_tconv_tflite.py --padding same
    python3 make_tconv_tflite.py --padding valid --k 4 --stride 2
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--k", type=int, default=3, help="transpose-conv kernel")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--padding", choices=["same", "valid"], default="same")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    x = tf.keras.layers.Conv2DTranspose(args.c, args.k, strides=args.stride,
                                        padding=args.padding, use_bias=True)(x)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(x)
    model = tf.keras.Model(inp, x)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()
    out = args.out or f"tconv_{args.padding}_k{args.k}s{args.stride}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
