#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_resize_tflite.py — emit a  conv -> RESIZE_{NEAREST_NEIGHBOR|BILINEAR} -> conv .tflite.

RESIZE_NEAREST_NEIGHBOR (TFLite builtin 97) / RESIZE_BILINEAR (23) are the FPN /
decoder-neck upsample the delegate claims as NodeKind::Resize (exact TFLite-coordinate
host kernel by default; opt-in resize_npu routes an integer-factor half-pixel-off
NEAREST resize onto the NPU). Sandwiched between convs so the whole graph is one
partition and the resize output is an internal tensor.

half_pixel_centers is the TFLite/Keras default (UpSampling2D uses it); --align_corners
exercises the other coordinate mode. float / int8 / uint8.
Usage:
    python3 make_resize_tflite.py --op nearest  --dtype float
    python3 make_resize_tflite.py --op bilinear --dtype int8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["nearest", "bilinear"], default="nearest")
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--factor", type=int, default=2, help="upsample factor")
    ap.add_argument("--align_corners", action="store_true")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    OH = OW = args.hw * args.factor
    method = "nearest" if args.op == "nearest" else "bilinear"

    def resize(t):
        return tf.image.resize(t, [OH, OW], method=method)

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    x = tf.keras.layers.Lambda(resize)(x)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(x)
    model = tf.keras.Model(inp, x)

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
    out = args.out or f"resize_{args.op}_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
