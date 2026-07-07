#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_layout_tflite.py — emit a conv -> {TRANSPOSE|PAD|SLICE|SPLIT} -> conv .tflite to
exercise the delegate's byte-exact layout ops (NodeKind::Transpose/Pad/Slice/Split).
Each op is wrapped BETWEEN convs so it is an internal partition tensor — the point of
claiming it is keeping `conv -> layout -> conv` ONE delegated partition (un-spilling the
layout node from the CPU) instead of splitting the graph around a CPU op.

  transpose : spatial H<->W swap, perm [0,2,1,3] (stays 4D NHWC for the trailing conv)
  pad       : spatial 1-px border (PAD); --value makes it PADV2 with a constant
  slice     : crop a spatial + channel sub-block (tf.slice begin/size)
  split     : split channels into N, concat them BACK in reverse, then conv — so the
              split's N outputs are all internal tensors (exercises multi-output binding)

float / int8 / uint8. Needs TensorFlow (the converter).
Usage:
    python3 make_layout_tflite.py --op transpose --dtype float
    python3 make_layout_tflite.py --op pad --value 0.0 --dtype int8
    python3 make_layout_tflite.py --op split --nsplit 2 --dtype uint8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["transpose", "pad", "slice", "split",
                                     "strided_slice", "strided_neg", "strided_shrink",
                                     "strided_ellipsis", "strided_ellipsis_mid",
                                     "strided_newaxis", "strided_ellipsis_newaxis"], required=True)
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--nsplit", type=int, default=2, help="split: number of equal parts (divides C)")
    ap.add_argument("--value", type=float, default=None, help="pad: constant fill -> emits PADV2")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)

    if args.op == "transpose":
        x = tf.keras.layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]))(x)
    elif args.op == "pad":
        pads = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        if args.value is None:
            x = tf.keras.layers.Lambda(lambda t: tf.pad(t, pads))(x)
        else:
            v = float(args.value)
            x = tf.keras.layers.Lambda(lambda t: tf.pad(t, pads, constant_values=v))(x)
    elif args.op == "slice":
        # crop the spatial border by 1 and take the first C-8 (or all) channels
        ch = max(8, args.c - 8)
        x = tf.keras.layers.Lambda(
            lambda t: tf.slice(t, [0, 1, 1, 0], [1, args.hw - 2, args.hw - 2, ch]))(x)
    elif args.op == "split":  # N equal parts along channels, concat back reversed (not identity)
        n = args.nsplit
        assert args.c % n == 0, "C must divide nsplit"
        x = tf.keras.layers.Lambda(
            lambda t: tf.concat(tf.split(t, n, axis=3)[::-1], axis=3))(x)
    elif args.op == "strided_slice":  # spatial crop -> STRIDED_SLICE (begin/end masks on N,C)
        x = tf.keras.layers.Lambda(lambda t: t[:, 1:-1, 1:-1, :])(x)
    elif args.op == "strided_neg":    # NEGATIVE stride: H fully reversed (::-1), W cropped+reversed /2
        x = tf.keras.layers.Lambda(lambda t: t[:, ::-1, -2:0:-2, :])(x)
    elif args.op == "strided_ellipsis":   # ELLIPSIS covers N,H,W; crop channels -> stays rank-4
        x = tf.keras.layers.Lambda(lambda t: t[..., 1:-1])(x)
    elif args.op == "strided_ellipsis_mid":  # crop H, ELLIPSIS covers W,C -> stays rank-4
        x = tf.keras.layers.Lambda(lambda t: t[:, 1:-1, ...])(x)
    elif args.op == "strided_newaxis":    # NEW_AXIS after N + spatial crop -> output rank-5
        x = tf.keras.layers.Lambda(lambda t: t[:, tf.newaxis, 1:-1, 1:-1, :])(x)
    elif args.op == "strided_ellipsis_newaxis":  # ELLIPSIS over all 4 axes + trailing NEW_AXIS -> rank-5
        x = tf.keras.layers.Lambda(lambda t: t[..., tf.newaxis])(x)
    else:  # strided_shrink: drop the H axis at index 2 -> shrink_axis_mask (output rank 3)
        x = tf.keras.layers.Lambda(lambda t: t[:, 2, :, :])(x)

    # rank-changing slices (shrink->3D, newaxis->5D) can't feed a 4D conv -> terminal output
    rank_preserving = args.op not in ("strided_shrink", "strided_newaxis", "strided_ellipsis_newaxis")
    if rank_preserving:
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
    tag = args.op + ("v2" if (args.op == "pad" and args.value is not None) else "")
    out = args.out or f"layout_{tag}_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
