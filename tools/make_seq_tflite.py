#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_seq_tflite.py — emit a small  conv -> {LOG | LOG_SOFTMAX | CUMSUM} -> conv .tflite
for the delegate's last-axis sequence ops. The delegate claims:
  * LOG          (builtin 73)  as NodeKind::Act  (exact host logf; act_npu = the
                                domain-limited DPU LUT). Inputs forced positive (softplus).
  * LOG_SOFTMAX  (builtin 50)  as NodeKind::LogSoftmax (exact host kernel).
  * CUMSUM       (builtin 128) as NodeKind::Cumsum (exact host prefix sum, last axis).
Sandwiched between convs so the whole graph is one partition. float only here.

Usage:
    python3 make_seq_tflite.py --op log
    python3 make_seq_tflite.py --op logsoftmax
    python3 make_seq_tflite.py --op cumsum
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["log", "logsoftmax", "cumsum"], default="logsoftmax")
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 keeps convs on the NPU)")
    ap.add_argument("--hw", type=int, default=8, help="input H=W")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tensorflow as tf

    inp = tf.keras.Input(shape=(args.hw, args.hw, args.c), batch_size=1)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(inp)
    if args.op == "log":
        # force a positive domain so log is well-defined (matches the convert_test x>0 gate)
        x = tf.keras.layers.Lambda(lambda t: tf.math.log(tf.nn.softplus(t) + 1.0))(x)
    elif args.op == "logsoftmax":
        x = tf.keras.layers.Lambda(lambda t: tf.nn.log_softmax(t, axis=-1))(x)
    else:  # cumsum along the channel (last) axis
        x = tf.keras.layers.Lambda(lambda t: tf.math.cumsum(t, axis=-1))(x)
    x = tf.keras.layers.Conv2D(args.c, 3, padding="same", use_bias=True)(x)
    model = tf.keras.Model(inp, x)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()
    out = args.out or f"seq_{args.op}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
