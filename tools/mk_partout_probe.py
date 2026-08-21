#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
mk_partout_probe.py — a quantized TFLite model whose partition output the planner would
leave in CUBE layout, which is the one refusal left that no shape check can ask.

WHY IT IS WRITTEN RATHER THAN FOUND. Every bound decidable from a node's own SHAPE is now
asked before the node is claimed (`mk_claim_probe.py` is the arm for those). What is left
fails at Prepare over a property of the whole PARTITION, and the corpus reaches none of it:
five classifiers, two detectors and a per-axis ResNet-18 all have their concatenations
strictly interior.

WHAT IT LANDS ON. A CONCATENATION is placement — its operands are written straight into
slices of one buffer that its consumer reads as a cube, so the layer runs no program and
owns no row-major tensor. That is correct while every reader is inside the partition. Make
the same tensor a MODEL OUTPUT as well and the frontend has to hand the runtime a NHWC
surface nothing ever writes, so it refuses the whole model at `AllocateTensors` instead:

    [rocket/rk3576] layer N is a partition output and the plan leaves it in cube layout

A convolution or a pool in the same position is already handled — the frontend hides its
handle from the planner's description, which costs one join and never a wrong answer — but
a concatenation has no handle to hide, so it is the shape that reaches the refusal.

THE MODEL. Two 1x1 convolutions the part takes, concatenated on the channel axis at a
16-channel boundary (a placed slice starts every sixteen channels, which is what one cube
atom interleaves), then a third convolution that reads the concatenation from INSIDE the
partition. Both the concatenation and the last convolution are model outputs, so the
concatenation is a partition output with an inside consumer — the two conditions together.
Every kernel is small and random: this model asserts a PLACEMENT, not an accuracy, and its
own TFLite output is the reference either way.

    tests/data/rk3576-net/.venv-build/bin/python tools/mk_partout_probe.py [out.tflite]
"""
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "partout_probe_concat.tflite"


def main():
    import tensorflow as tf
    from tensorflow import keras

    rng = np.random.default_rng(0)
    x = keras.Input(shape=(16, 16, 32), batch_size=1)
    a = keras.layers.Conv2D(32, 1, padding="same", activation="relu")(x)
    b = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    c = keras.layers.Concatenate(axis=-1)([a, b])
    d = keras.layers.Conv2D(32, 1, padding="same")(c)
    model = keras.Model(x, [c, d])

    def rep():
        for _ in range(32):
            yield [rng.standard_normal((1, 16, 16, 32)).astype(np.float32)]

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
