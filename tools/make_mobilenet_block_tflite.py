#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_mobilenet_block_tflite.py — emit a single MobileNetV2 inverted-residual block
.tflite, the smallest model that exercises the WHOLE delegate at once.

A block is: 1x1 pointwise EXPAND (relu6) -> KxK DEPTHWISE (relu6) -> 1x1 pointwise
PROJECT (linear) -> optional residual ADD. So one block touches every op the delegate
claims along a real MobileNet/SSD backbone:

  * the two 1x1 pointwise convs   -> NPU (float: the matmul fast path; int8: the
                                     dequant->fp16-conv->requant path),
  * the KxK depthwise             -> NPU (native DW_EN, channel-tiled if wide),
  * the residual ADD              -> host aux kernel (same-shape, keeps the partition
                                     contiguous).

That makes it the natural NEXT step after the single-op depthwise model
(make_depthwise_tflite.py): instead of one claimed op it gives the partitioner a
contiguous conv->dw->conv->add subgraph, so `run_delegate.py --option profile=1` shows
several `[rocket] …` lines (pointwise convs, a `dwconv`, and an add) in one partition.

Channel rules so every op is claimed (else it silently falls to CPU): pick c_in a
multiple of 32, expand so c_exp = c_in*expand is a multiple of 32 (the depthwise group
G), and c_out a multiple of 16. A residual ADD needs stride 1 and c_in == c_out (same
shape); it's auto-enabled then unless --no-residual.

Defaults (int8, per-channel weights — the realistic detector quantization our delegate
handles): c_in=32, expand=6 (c_exp=192), c_out=32, 32x32, 3x3, stride 1, residual on.
Pass --dtype float for a float block (then the 1x1 convs take the multicore matmul fast
path and the comparison is closer to exact). --per-tensor matches Teflon's quantization
(single weight scale) if you also want to A/B against libteflon.so.

Needs TensorFlow (the converter). Run on whatever box has TF (e.g. a machine with TensorFlow installed);
copy the .tflite next to run_delegate.py. Usage:
    python3 make_mobilenet_block_tflite.py                       # int8 residual block
    python3 make_mobilenet_block_tflite.py --dtype float
    python3 make_mobilenet_block_tflite.py --c-in 64 --c-out 96 --stride 2 --hw 56
"""
import argparse


def build_model(tf, c_in, c_exp, c_out, hw, k, stride, residual):
    """MobileNetV2 inverted-residual block via the functional API (functional so the
    residual Add can reference the block input). Separate ReLU(6.) layers are fused into
    the preceding conv by the TFLite converter (-> a CONV_2D with FusedActivation RELU6,
    which the delegate maps to ROCKET_ACT_RELU6); the project conv is left linear."""
    inp = tf.keras.Input(shape=(hw, hw, c_in), batch_size=1)
    # 1x1 pointwise expand + relu6
    x = tf.keras.layers.Conv2D(c_exp, 1, padding="same", use_bias=True)(inp)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    # KxK depthwise + relu6
    x = tf.keras.layers.DepthwiseConv2D(k, strides=stride, padding="same",
                                        use_bias=True)(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    # 1x1 pointwise project (linear)
    x = tf.keras.layers.Conv2D(c_out, 1, padding="same", use_bias=True)(x)
    if residual:
        x = tf.keras.layers.Add()([inp, x])      # same-shape residual (the aux ADD)
    return tf.keras.Model(inp, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-in", type=int, default=32, help="input channels (mult of 32)")
    ap.add_argument("--c-out", type=int, default=32, help="output channels (mult of 16)")
    ap.add_argument("--expand", type=int, default=6, help="expansion factor (c_exp=c_in*expand, mult of 32)")
    ap.add_argument("--hw", type=int, default=32, help="input H=W")
    ap.add_argument("--k", type=int, default=3, help="depthwise kernel H=W")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-residual", action="store_true",
                    help="disable the residual ADD even when shapes allow it")
    ap.add_argument("--dtype", choices=["int8", "float"], default="int8")
    ap.add_argument("--per-tensor", action="store_true",
                    help="int8 only: force per-tensor weight quant (Teflon-compatible)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c_in, c_out = args.c_in, args.c_out
    c_exp = c_in * args.expand
    # residual needs stride 1 and matching in/out shape
    residual = (not args.no_residual) and args.stride == 1 and c_in == c_out

    # warn (don't fail) on alignment the delegate gates on, so a bad choice is obvious
    warns = []
    if c_in % 32:  warns.append(f"c_in={c_in} not a multiple of 32 (1x1 matmul K / pad)")
    if c_exp % 32: warns.append(f"c_exp={c_exp} not a multiple of 32 (depthwise group G)")
    if c_out % 16: warns.append(f"c_out={c_out} not a multiple of 16 (conv OC group)")
    for w in warns:
        print(f"  warning: {w} -> that op may fall to CPU")

    import numpy as np       # late imports so --help works without TF/numpy installed
    import tensorflow as tf

    tag = args.dtype + ("_pt" if (args.dtype == "int8" and args.per_tensor) else "")
    out = args.out or (f"mbv2_block_c{c_in}e{args.expand}_{args.hw}x{args.hw}"
                       f"_k{args.k}_s{args.stride}{'_res' if residual else ''}_{tag}.tflite")

    model = build_model(tf, c_in, c_exp, c_out, args.hw, args.k, args.stride, residual)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.dtype == "int8":
        rng = np.random.default_rng(0)
        def rep():
            for _ in range(64):
                yield [rng.standard_normal((1, args.hw, args.hw, c_in)).astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if args.per_tensor:
            # single weight scale per tensor (what Teflon's rocket gate requires); our
            # own delegate handles BOTH per-channel (default) and per-tensor.
            conv._experimental_disable_per_channel = True
    # float: no optimizations -> a plain float32 graph (the 1x1 convs hit the matmul path)

    tflite = conv.convert()
    with open(out, "wb") as f:
        f.write(tflite)
    print(f"wrote {out} ({len(tflite)} bytes): MobileNetV2 block "
          f"c_in={c_in} c_exp={c_exp} c_out={c_out} {args.hw}x{args.hw} k={args.k} "
          f"s={args.stride} {'residual ' if residual else ''}{args.dtype}")
    print("run it through the delegate:\n"
          f"  python3 tools/run_delegate.py {out} \\\n"
          "      --delegate ./build/libtflite_rocket.so --option profile=1 --compare")


if __name__ == "__main__":
    main()
