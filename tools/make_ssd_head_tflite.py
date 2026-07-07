#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
make_ssd_head_tflite.py — emit an SSD-style detection HEAD: parallel 1x1 CONV_2D
predictors over a feature map, each reshaped and joined by CONCATENATION. This is the
op mix a real detector carries AFTER the backbone, and the natural complement to the
backbone generators (make_mobilenet_block / _stack): instead of conv->dw->conv->add it
gives the delegate a fan-out of CONV_2D heads feeding a CONCAT — the SSD/MobileNet-SSD
box+class predictor pattern.

For each of --levels feature levels (here modelled as one shared input feature map fed
through a 1x1 to a per-level channel count, to keep the model single-input and small),
two predictor convs are emitted:
  * a BOX predictor: 1x1 CONV_2D, OC = anchors*4
  * a CLASS predictor: 1x1 CONV_2D, OC = anchors*(classes+1)
Each predictor output is RESHAPEd to [1, H*W*anchors, *] and the per-level results are
CONCATENATED along axis 1 into the final box / class tensors. With anchors=6,
classes=20 the class head is OC = 6*21 = 126 and the box head OC = 24 — both NON-multiples
of 16, which exercises the driver's OC-pad path (heads whose OC is not a multiple of 16).

Everything is 1x1 stride-1 so the convs are the pointwise/matmul case; the point of the
model is op COVERAGE (several CONV_2D + RESHAPE + CONCATENATION in one partition), not
spatial conv. Needs TensorFlow; run on the TF box and copy the .tflite to run_delegate.

Usage:
    python3 make_ssd_head_tflite.py                         # int8, anchors=6 classes=20
    python3 make_ssd_head_tflite.py --dtype float --levels 2
    python3 make_ssd_head_tflite.py --anchors 4 --classes 80 --c-in 256 --hw 10
"""
import argparse


def build_head(tf, c_in, c_level, hw, levels, anchors, classes):
    """Single-input SSD-style head. The input feature map is projected to c_level by a
    1x1 (the 'backbone tap'), then each level runs a box (OC=anchors*4) and class
    (OC=anchors*(classes+1)) 1x1 predictor; predictions are reshaped to
    [1, H*W*anchors, k] and concatenated across levels along axis 1."""
    inp = tf.keras.Input(shape=(hw, hw, c_in), batch_size=1)
    feat = tf.keras.layers.Conv2D(c_level, 1, padding="same", use_bias=True)(inp)

    box_k = 4
    cls_k = classes + 1
    boxes, clses = [], []
    for lv in range(levels):
        # a small per-level 1x1 so each level's predictor sees a distinct tensor
        f = tf.keras.layers.Conv2D(c_level, 1, padding="same", use_bias=True,
                                   name=f"lvl{lv}_tap")(feat)
        box = tf.keras.layers.Conv2D(anchors * box_k, 1, padding="same", use_bias=True,
                                     name=f"lvl{lv}_box")(f)
        cls = tf.keras.layers.Conv2D(anchors * cls_k, 1, padding="same", use_bias=True,
                                     name=f"lvl{lv}_cls")(f)
        boxes.append(tf.keras.layers.Reshape((hw * hw * anchors, box_k))(box))
        clses.append(tf.keras.layers.Reshape((hw * hw * anchors, cls_k))(cls))

    if levels > 1:
        box_out = tf.keras.layers.Concatenate(axis=1, name="box_concat")(boxes)
        cls_out = tf.keras.layers.Concatenate(axis=1, name="cls_concat")(clses)
    else:
        box_out, cls_out = boxes[0], clses[0]
    return tf.keras.Model(inp, [box_out, cls_out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-in", type=int, default=128, help="input feature channels (mult of 32)")
    ap.add_argument("--c-level", type=int, default=64, help="per-level channel count (mult of 32)")
    ap.add_argument("--hw", type=int, default=8, help="feature map H=W")
    ap.add_argument("--levels", type=int, default=2, help="number of feature levels (concat inputs)")
    ap.add_argument("--anchors", type=int, default=6, help="anchors per location")
    ap.add_argument("--classes", type=int, default=20, help="object classes (+1 background)")
    ap.add_argument("--dtype", choices=["int8", "float"], default="int8")
    ap.add_argument("--per-tensor", action="store_true",
                    help="int8 only: force per-tensor weight quant (Teflon-compatible)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    box_oc = args.anchors * 4
    cls_oc = args.anchors * (args.classes + 1)
    print(f"  box head OC={box_oc} ({'mult-16' if box_oc % 16 == 0 else 'OC%16 -> OC-pad path'}), "
          f"class head OC={cls_oc} ({'mult-16' if cls_oc % 16 == 0 else 'OC%16 -> OC-pad path'})")
    if args.c_in % 32 or args.c_level % 32:
        print("  warning: c_in / c_level not a multiple of 32 -> a 1x1 may fall to CPU")

    import numpy as np            # late imports so --help works without TF/numpy
    import tensorflow as tf

    tag = args.dtype + ("_pt" if (args.dtype == "int8" and args.per_tensor) else "")
    out = args.out or (f"ssd_head_c{args.c_in}_{args.hw}x{args.hw}_l{args.levels}"
                       f"_a{args.anchors}_c{args.classes}_{tag}.tflite")

    model = build_head(tf, args.c_in, args.c_level, args.hw, args.levels,
                       args.anchors, args.classes)

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
    print(f"wrote {out} ({len(tflite)} bytes): SSD-style head "
          f"c_in={args.c_in} {args.hw}x{args.hw} levels={args.levels} "
          f"anchors={args.anchors} classes={args.classes} {args.dtype}")
    print("run it through the delegate:\n"
          f"  python3 tools/run_delegate.py {out} \\\n"
          "      --delegate ./build/libtflite_rocket.so --option profile=1 --compare")


if __name__ == "__main__":
    main()
