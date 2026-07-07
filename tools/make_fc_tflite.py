#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_fc_tflite.py — emit a FULLY_CONNECTED (Dense) .tflite, the classifier / FC-head op
the delegate now claims (NodeKind::FC, the resident fp16 matmul). float only (v1).

A Dense layer is a matmul C[M,N] = A[M,K]·B[N,K]^T + bias: M = batch, K = input_dim,
N = units. K is padded to %32 and N to %16 by the delegate, so any classifier width (1001,
90, 10, ...) offloads. Needs TensorFlow; run on a machine with TensorFlow installed. Usage:
    python3 make_fc_tflite.py --k 1024 --n 1001            # ImageNet classifier head
    python3 make_fc_tflite.py --k 256 --n 90 --act relu6   # detector class head
    python3 make_fc_tflite.py --m 4 --k 64 --n 32          # batched FC
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1, help="batch (1 or a multiple of 4)")
    ap.add_argument("--k", type=int, default=256, help="input_dim")
    ap.add_argument("--n", type=int, default=90, help="units (output features)")
    ap.add_argument("--act", choices=["none", "relu", "relu6"], default="none")
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    act = {"none": None, "relu": "relu", "relu6": tf.nn.relu6}[args.act]
    inp = tf.keras.Input(shape=(args.k,), batch_size=args.m)
    x = tf.keras.layers.Dense(args.n, activation=act, use_bias=not args.no_bias)(inp)
    model = tf.keras.Model(inp, x)

    tflite = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    out = args.out or f"fc_m{args.m}_k{args.k}_n{args.n}_{args.act}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
