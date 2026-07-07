#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_act_tflite.py — emit a  conv -> ACTIVATION -> conv  .tflite, where ACTIVATION is a
STANDALONE HARD_SWISH or LOGISTIC(sigmoid) op — the MobileNetV3 / detector nonlinearity
the delegate now claims (NodeKind::Act). Unlike Relu/Relu6 (conv-FUSED), HardSwish and
Sigmoid are their own TFLite builtins, emitted as separate nodes between convs.

Sandwiching the activation between two convs makes the whole graph one CONTIGUOUS
partition, so `run_delegate.py --option profile=1` prints
    [rocket] op0: conv ...
    [rocket] op1: act (host) ...
    [rocket] op2: conv ...
in a single partition — the un-spill + contiguity win (without claiming the act, the
partition would split conv | act(CPU) | conv).

float / int8 / uint8. Needs TensorFlow (the converter); run on a machine with TensorFlow installed and copy
the .tflite next to run_delegate.py. The script prints the converted op list so you can
confirm a standalone HARD_SWISH / LOGISTIC builtin (not a decomposed mul/add) is present.
Usage:
    python3 make_act_tflite.py --act hardswish --dtype float
    python3 make_act_tflite.py --act sigmoid   --dtype int8
    python3 make_act_tflite.py --act hardswish --dtype uint8
"""
import argparse


def hard_swish(tf, x):
    # the canonical pattern the TFLite converter fuses into a single HARD_SWISH builtin
    return x * tf.nn.relu6(x + 3.0) * (1.0 / 6.0)


# The relu family (RELU / RELU6 / RELU_N1_TO_1) are FUSED activations: the TFLite
# converter folds them into a preceding Conv's fused_activation slot, so they never
# appear as a standalone builtin there. To emit a STANDALONE node we apply them right
# off the model input (no conv to fuse into), then a trailing conv keeps the graph one
# delegated partition. The non-fused ops (sigmoid/tanh/elu/exp/sqrt/.../softmax) are
# their own builtins between two convs regardless.
_FUSED_RELU = {"relu", "relu6", "relu_n1_1"}


def apply_act(tf, act, t):
    if act == "hardswish":
        return hard_swish(tf, t)
    if act == "tanh":
        return tf.keras.layers.Activation("tanh")(t)
    if act == "elu":
        return tf.keras.layers.Activation("elu")(t)
    if act == "sigmoid":
        return tf.keras.layers.Activation("sigmoid")(t)
    if act == "relu":
        return tf.nn.relu(t)
    if act == "relu6":
        return tf.nn.relu6(t)
    if act == "relu_n1_1":
        return tf.clip_by_value(t, -1.0, 1.0)
    if act == "leaky_relu":
        return tf.nn.leaky_relu(t, alpha=0.1)
    if act == "exp":
        return tf.math.exp(t)
    if act == "sqrt":
        return tf.math.sqrt(tf.math.abs(t) + 0.5)   # keep the domain > 0
    if act == "rsqrt":
        return tf.math.rsqrt(tf.math.abs(t) + 0.5)  # keep the domain > 0
    if act == "abs":
        return tf.math.abs(t)
    if act == "neg":
        return tf.math.negative(t)
    if act == "square":
        return tf.math.square(t)
    if act == "floor":
        return tf.math.floor(t)
    if act == "softmax":
        return tf.keras.layers.Activation("softmax")(t)   # over the last axis (channels)
    raise ValueError(act)


def build_model(tf, act, c, hw, k, noconv=False):
    inp = tf.keras.Input(shape=(hw, hw, c), batch_size=1)
    if noconv:
        # input -> ACT -> output, no conv anywhere. The delegate's only op is the
        # activation, so --compare measures the activation's exactness directly (the
        # fp16 conv error can't mask it). The host activation kernels are exact float,
        # so delegate==CPU bit-exact for them (softmax: transcendental, ~1e-7).
        x = tf.keras.layers.Lambda(lambda t: apply_act(tf, act, t))(inp)
        return tf.keras.Model(inp, x)
    if act in _FUSED_RELU:
        # off the input so the relu can't fuse into a conv -> standalone builtin
        x = tf.keras.layers.Lambda(lambda t: apply_act(tf, act, t))(inp)
        x = tf.keras.layers.Conv2D(c, k, padding="same", use_bias=True)(x)
    else:
        x = tf.keras.layers.Conv2D(c, k, padding="same", use_bias=True)(inp)
        x = tf.keras.layers.Lambda(lambda t: apply_act(tf, act, t))(x)
        x = tf.keras.layers.Conv2D(c, k, padding="same", use_bias=True)(x)
    return tf.keras.Model(inp, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", choices=["hardswish", "sigmoid", "tanh", "elu", "relu", "relu6",
                                       "relu_n1_1", "leaky_relu", "exp", "sqrt", "rsqrt", "abs",
                                       "neg", "square", "floor", "softmax"], default="hardswish")
    ap.add_argument("--c", type=int, default=32, help="channels (mult of 32 so the 1x1/3x3 stays on the NPU)")
    ap.add_argument("--hw", type=int, default=16, help="input H=W")
    ap.add_argument("--k", type=int, default=3, help="conv kernel H=W")
    ap.add_argument("--dtype", choices=["float", "int8", "uint8"], default="float")
    ap.add_argument("--noconv", action="store_true",
                    help="input->ACT->output with no conv, so --compare isolates the activation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    model = build_model(tf, args.act, args.c, args.hw, args.k, args.noconv)
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
    tag = "_noconv" if args.noconv else ""
    out = args.out or f"act_{args.act}{tag}_{args.dtype}_c{args.c}_{args.hw}x{args.hw}.tflite"
    with open(out, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    ops = [d["op_name"] for d in interp._get_ops_details()]
    print(f"wrote {out} ({len(tflite)} bytes); ops = {ops}")


if __name__ == "__main__":
    main()
