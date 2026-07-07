#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_add_tflite.py — emit a MINIMAL int8 model whose only "interesting" op is an
elementwise ADD that Mesa Teflon will accept, for the rocket EW-operand capture.

Teflon does NOT support a standalone two-input ADD: rkt_ml.c FUSES the addition
into the convolution that produces one of its inputs (the add_tensor becomes the
EW residual of a conv). So the minimal capturable graph is a residual:

    x -> conv_a(1x1) ->\
    x -> conv_b(1x1) -> Add -> out

Teflon emits conv_a, then conv_b WITH add_tensor = conv_a.output → the conv_b
regcmd carries the EW-add operand path (DPU_EW_CFG EW_ALU_ALGO=add + EW_OP_SRC,
the DPU_RDMA ERDMA_CFG / EW_BASE_ADDR / SURF_NOTCH / COMB_USE) that we need to
diff against gen_ew_mul_fp16. Full INT8, per-tensor (Teflon rejects per-axis).

    python3 make_add_tflite.py                 # add_c8_4x4.tflite
    python3 make_add_tflite.py --c 8 --hw 4
"""
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=8, help="channels")
    ap.add_argument("--hw", type=int, default=4, help="input H=W")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tensorflow as tf

    out = args.out or f"add_c{args.c}_{args.hw}x{args.hw}.tflite"

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    a = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(inp)
    b = tf.keras.layers.Conv2D(args.c, 1, padding="same", use_bias=True)(inp)
    y = tf.keras.layers.Add()([a, b])
    model = tf.keras.Model(inp, y)

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
    conv._experimental_disable_per_channel = True   # Teflon needs per-tensor
    tflite = conv.convert()

    with open(out, "wb") as f:
        f.write(tflite)
    print(f"wrote {out} ({len(tflite)} bytes): residual ADD of two 1x1 convs, "
          f"C={args.c} {args.hw}x{args.hw} int8")

if __name__ == "__main__":
    main()
