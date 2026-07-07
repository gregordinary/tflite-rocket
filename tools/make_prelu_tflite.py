#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_prelu_tflite.py — emit a  conv -> PRELU -> conv  .tflite.

PRELU (TFLite builtin 54) is a standalone op with a PER-CHANNEL learned negative
slope (a constant alpha tensor), out = x>=0 ? x : alpha[c]*x — the YOLO / segmentation
nonlinearity. The delegate claims it as NodeKind::Prelu (exact host kernel by default;
opt-in act_npu routes float PReLU onto the NPU via rocket_prelu_fp16). Keras PReLU with
shared_axes=[1,2] gives the [1,1,C] (per-channel, broadcast over spatial) alpha the
delegate accepts. Sandwiched between convs so the whole graph is one partition.

float / int8 / uint8. Needs TensorFlow (the converter).
Usage:
    python3 make_prelu_tflite.py --dtype float
    python3 make_prelu_tflite.py --dtype int8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=16, help="input H=W")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    # shared_axes=[1,2] -> one slope per channel (alpha shape [1,1,C]); init non-zero so
    # the negative branch is actually exercised.
    x = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.1),
                              shared_axes=[1, 2])(x)
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
    out = args.out or f"prelu_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
