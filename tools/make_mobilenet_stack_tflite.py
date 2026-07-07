#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_mobilenet_stack_tflite.py — emit a small STACK of MobileNetV2 inverted-residual
blocks, the next step up from the single block (make_mobilenet_block_tflite.py).

Where one block proves the partitioner taking a conv->dw->conv->add subgraph, a stack
of several blocks (with a strided downsample block in the middle and a couple of
residual blocks around it) gives the delegate a LONG contiguous partition that mirrors
a real MobileNet/SSD backbone segment: many pointwise 1x1 (matmul / dequant-conv),
several KxK depthwise (native DW, channel- and now spatially-tiled), and the residual
ADDs that keep the partition from splitting. profile=1 then shows a dozen-plus
`[rocket] …` lines in ONE partition, the realistic per-layer cost profile.

A block is 1x1 expand (relu6) -> KxK depthwise (relu6) -> 1x1 project (linear), with a
residual ADD when stride==1 and c_in==c_out. The stack is described by --spec, a comma
list of `c_out:stride` blocks (expand factor fixed by --expand); the input channels of
block i are the output of block i-1. Default spec walks 32->32 (res) ->64 s2 ->64 (res)
->96 — i.e. two residual blocks, a downsample, another residual, and a channel-growth
block: 5 blocks, ~15 NPU ops.

Channel alignment (so every op stays on the NPU, not silently CPU): c_out a multiple of
16, and c_out*expand a multiple of 32 (the depthwise group G). The tool warns on a bad
choice rather than failing. Needs TensorFlow (the converter); run on the box with TF
(e.g. a machine with TensorFlow installed) and copy the .tflite next to run_delegate.py.

Usage:
    python3 make_mobilenet_stack_tflite.py                       # int8 default stack
    python3 make_mobilenet_stack_tflite.py --dtype float
    python3 make_mobilenet_stack_tflite.py --c-in 32 --spec 48:1,48:1,96:2,96:1 --hw 48
"""
import argparse


def build_stack(tf, c_in, blocks, expand, hw, k):
    """blocks: list of (c_out, stride). Returns a Keras functional model chaining the
    inverted-residual blocks. ReLU6 layers fuse into the preceding conv at convert time
    (-> FusedActivation RELU6); the project conv stays linear; an Add appears whenever a
    block keeps shape (stride 1, c_in==c_out)."""
    inp = tf.keras.Input(shape=(hw, hw, c_in), batch_size=1)
    x = inp
    cin = c_in
    for (c_out, stride) in blocks:
        c_exp = cin * expand
        shortcut = x
        y = tf.keras.layers.Conv2D(c_exp, 1, padding="same", use_bias=True)(x)   # expand
        y = tf.keras.layers.ReLU(max_value=6.0)(y)
        y = tf.keras.layers.DepthwiseConv2D(k, strides=stride, padding="same",
                                            use_bias=True)(y)                     # depthwise
        y = tf.keras.layers.ReLU(max_value=6.0)(y)
        y = tf.keras.layers.Conv2D(c_out, 1, padding="same", use_bias=True)(y)    # project
        if stride == 1 and cin == c_out:
            y = tf.keras.layers.Add()([shortcut, y])                             # residual
        x = y
        cin = c_out
    return tf.keras.Model(inp, x)


def parse_spec(spec):
    """`c_out:stride,c_out:stride,...` -> [(c_out, stride), ...]."""
    blocks = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            c_out, stride = tok.split(":")
        else:
            c_out, stride = tok, "1"
        blocks.append((int(c_out), int(stride)))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-in", type=int, default=32, help="input channels (mult of 32)")
    ap.add_argument("--spec", default="32:1,32:1,64:2,64:1,96:1",
                    help="comma list of c_out:stride blocks")
    ap.add_argument("--expand", type=int, default=6, help="expansion factor (c_exp=c_in*expand)")
    ap.add_argument("--hw", type=int, default=32, help="input H=W")
    ap.add_argument("--k", type=int, default=3, help="depthwise kernel H=W")
    ap.add_argument("--dtype", choices=["int8", "float"], default="int8")
    ap.add_argument("--per-tensor", action="store_true",
                    help="int8 only: force per-tensor weight quant (Teflon-compatible)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blocks = parse_spec(args.spec)
    if not blocks:
        ap.error("empty --spec")

    # warn (don't fail) on the alignments the delegate gates on
    warns = []
    if args.c_in % 32:
        warns.append(f"c_in={args.c_in} not a multiple of 32")
    cin = args.c_in
    for i, (c_out, stride) in enumerate(blocks):
        if (cin * args.expand) % 32:
            warns.append(f"block {i}: c_exp={cin*args.expand} not a multiple of 32 (DW group)")
        if c_out % 16:
            warns.append(f"block {i}: c_out={c_out} not a multiple of 16")
        cin = c_out
    for w in warns:
        print(f"  warning: {w} -> that op may fall to CPU")

    import numpy as np            # late imports so --help works without TF/numpy
    import tensorflow as tf

    tag = args.dtype + ("_pt" if (args.dtype == "int8" and args.per_tensor) else "")
    spec_tag = "_".join(f"{c}s{s}" for (c, s) in blocks)
    out = args.out or (f"mbv2_stack_c{args.c_in}e{args.expand}_{args.hw}_"
                       f"{spec_tag}_{tag}.tflite")

    model = build_stack(tf, args.c_in, blocks, args.expand, args.hw, args.k)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.dtype == "int8":
        rng = np.random.default_rng(0)
        def rep():
            for _ in range(64):
                yield [rng.standard_normal((1, args.hw, args.hw, args.c_in)).astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if args.per_tensor:
            conv._experimental_disable_per_channel = True

    tflite = conv.convert()
    with open(out, "wb") as f:
        f.write(tflite)
    print(f"wrote {out} ({len(tflite)} bytes): MobileNetV2 stack "
          f"c_in={args.c_in} expand={args.expand} {args.hw}x{args.hw} blocks={blocks} {args.dtype}")
    print("run it through the delegate:\n"
          f"  python3 tools/run_delegate.py {out} \\\n"
          "      --delegate ./build/libtflite_rocket.so --option profile=1 --compare")


if __name__ == "__main__":
    main()
