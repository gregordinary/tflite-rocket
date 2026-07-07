#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_l2norm_tflite.py — emit a  conv -> L2_NORMALIZATION -> conv  .tflite. The
L2_NORMALIZATION (TFLite builtin 11) normalizes each spatial position over the channel
axis; the delegate claims it as NodeKind::L2Norm (exact host kernel by default; opt-in
norm_npu routes float L2Norm onto the NPU via rocket_l2norm_fp16). Sandwiched between
convs so the whole graph is one partition.

float / int8 / uint8. Needs TensorFlow (the converter).
Usage:
    python3 make_l2norm_tflite.py --dtype float
    python3 make_l2norm_tflite.py --dtype int8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    # normalize over the channel (last) axis -> TFLite L2_NORMALIZATION
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(x)
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
    out = args.out or f"l2norm_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
