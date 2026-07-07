#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_reduce_tflite.py — emit a  conv -> MEAN|REDUCE_MAX|REDUCE_MIN(axes=[1,2]) -> conv1x1
.tflite. The spatial reduce (TFLite MEAN=40 / REDUCE_MAX=82 / REDUCE_MIN=89 over the
H,W axes) is the GlobalAveragePool / global-max head the delegate claims as
NodeKind::Reduce (exact host kernel by default; opt-in pool_npu routes float reduces
onto the NPU PPU). keepdims=True keeps the [1,1,1,C] shape so a trailing 1x1 conv keeps
the whole graph one partition (and the reduce output an internal tensor).

float / int8 / uint8. Needs TensorFlow (the converter).
Usage:
    python3 make_reduce_tflite.py --op mean --dtype float
    python3 make_reduce_tflite.py --op max  --dtype int8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["mean", "max", "min"], default="mean")
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=14, help="input H=W")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    reduce_fn = {"mean": tf.reduce_mean, "max": tf.reduce_max, "min": tf.reduce_min}[args.op]
    x = tf.keras.layers.Lambda(lambda t: reduce_fn(t, axis=[1, 2], keepdims=True))(x)
    x = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(x)
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
    out = args.out or f"reduce_{args.op}_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
