#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_depthwise_tflite.py — emit a MINIMAL single-op DEPTHWISE_CONV_2D .tflite.

Purpose: the cleanest possible input for the Mesa Teflon depthwise regcmd capture.
A one-layer model means Teflon emits exactly ONE depthwise weight BO + one
regcmd, so the dump diffs apples-to-apples against the driver's gen_conv2d_dw_fp16 for
the SAME shape. Defaults match a shape conv2d_fp16_rocket already exercises
(C=64, 8x8, 3x3, stride 1, SAME) so the comparison is direct.

Full INT8 quantization (Teflon is int8-only). The captured regcmd is int8, but the
STRUCTURAL fields we need (DW_EN, CONV_MODE, OC padding-to-64, size_e, the weight
feature/channel pairing, the DPU output path) are what name the driver's wrong field.

Needs TensorFlow (the converter). Run on a machine with TensorFlow installed; copy the
.tflite to the target device. Usage:
    python3 make_depthwise_tflite.py                 # dw_c64_8x8_3x3_s1.tflite
    python3 make_depthwise_tflite.py --c 128 --hw 7 --k 5 --stride 1 --pad valid
"""
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=64, help="channels (C=OC=IC)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--k", type=int, default=3, help="kernel H=W")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pad", choices=["same", "valid"], default="same")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tensorflow as tf  # imported late so --help works without TF

    out = args.out or f"dw_c{args.c}_{args.hw}x{args.hw}_{args.k}x{args.k}_s{args.stride}.tflite"

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=args.k, strides=args.stride, padding=args.pad,
            depth_multiplier=1, use_bias=True),
    ])

    # representative dataset → full-integer (int8) quantization
    rng = np.random.default_rng(0)
    def rep():
        for _ in range(64):
            yield [rng.standard_normal((1, args.hw, args.hw, args.c)).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    # Force PER-TENSOR weight quantization (single scale, not a per-channel array).
    # Mesa Teflon's rocket gate rejects per-axis tensors (tensor_quantization_supported
    # requires scales/zero_points == NULL), exactly like Ethos-U's vela. Without this
    # the default int8 path emits per-channel weights and Teflon never claims the op.
    conv._experimental_disable_per_channel = True
    tflite = conv.convert()

    with open(out, "wb") as f:
        f.write(tflite)
    print(f"wrote {out} ({len(tflite)} bytes): DEPTHWISE_CONV_2D "
          f"C={args.c} {args.hw}x{args.hw} K={args.k}x{args.k} s={args.stride} {args.pad} int8")

if __name__ == "__main__":
    main()
