#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
mk_claim_probe.py — a quantized TFLite model built to REACH a bound the corpus does not.

WHY IT IS WRITTEN RATHER THAN FOUND. The delegate's claim gate asks the library about
every bound it can before claiming a node, because a node claimed and then refused at
pack time fails `AllocateTensors` and takes the whole model with it, where an unclaimed
node merely runs on the CPU. That distinction is untestable against a corpus in which
nothing reaches any of those bounds: the five classifiers, the two detectors and the
per-axis ResNet-18 all sit comfortably inside the envelope, so every one of them passes
whether the gate asks or not. The arm has to be a model that lands outside it.

WHAT IT LANDS ON. The middle convolution is ic=256, oc=32, 8x8, k5x5 SAME — the one
shape in `rk3576_conv_shapes.h` marked `lib_refuse`. Its resident weight slice is
32*256*5*5 = 200 KiB, past what the CBUF pool holds at a SINGLE output-channel group, so
there is no group count left for the library's output-channel split to fall back to and
the recourse is an input-channel split the on-chip requant forecloses. It is refused, and
that refusal is correct — the question this model asks is WHERE it surfaces.

It is surrounded by two ordinary 1x1 convolutions that the part does take, so a run
either way has claimed work in it: the expected outcome is TWO delegated partitions with
the refused node on the CPU between them, and the model's output matching CPU TFLite's.

    tests/data/rk3576-net/.venv-build/bin/python tools/mk_claim_probe.py [out.tflite]
"""
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "claim_probe_slice.tflite"


def main():
    import tensorflow as tf
    from tensorflow import keras

    rng = np.random.default_rng(0)
    x = keras.Input(shape=(8, 8, 32), batch_size=1)
    # Up to the deep channel count the bound is stated over, then the refused layer,
    # then back down. Every kernel is small and random: this model asserts a PLACEMENT,
    # not an accuracy, and its own TFLite output is the reference either way.
    y = keras.layers.Conv2D(256, 1, padding="same", activation="relu")(x)
    y = keras.layers.Conv2D(32, 5, padding="same", activation="relu")(y)
    y = keras.layers.Conv2D(32, 1, padding="same")(y)
    model = keras.Model(x, y)

    def rep():
        for _ in range(32):
            yield [rng.standard_normal((1, 8, 8, 32)).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    buf = conv.convert()
    with open(OUT, "wb") as f:
        f.write(buf)
    print(f"wrote {OUT} ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
