#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# End-to-end smoke test for the tflite-rocket delegate's on-NPU pooling (pool_npu).
# Builds tiny float TFLite pool models, runs each (a) plain on the CPU interpreter and
# (b) through libtflite_rocket.so with pool_npu=1, and compares. With profile=1 the
# delegate prints a per-op location tag to stderr; "pool (npu-ppu)" proves the op ran on
# the PPU (not the host fallback). Run on RK3588 hardware (needs a device + tensorflow):
#   python3 run_pool_delegate.py /path/to/libtflite_rocket.so
import sys, numpy as np, tensorflow as tf

SO = sys.argv[1] if len(sys.argv) > 1 else "../build/libtflite_rocket.so"

def pool_tflite(kind, ksize, strides):
    inp = tf.keras.Input(shape=(8, 8, 16), batch_size=1)
    lyr = (tf.keras.layers.AveragePooling2D if kind == "avg" else tf.keras.layers.MaxPooling2D)(
        pool_size=ksize, strides=strides, padding="valid")(inp)
    m = tf.keras.Model(inp, lyr)
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    return c.convert()

def run(model, delegate=None):
    it = tf.lite.Interpreter(model_content=model,
                             experimental_delegates=[delegate] if delegate else None)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(inp["shape"]).astype(np.float32)
    it.set_tensor(inp["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"]).copy()

fails = 0
for kind, k, s in [("avg", 2, 2), ("max", 3, 2), ("avg", 4, 4)]:
    model = pool_tflite(kind, k, s)
    cpu = run(model)
    deleg = tf.lite.experimental.load_delegate(
        SO, options={"aux_ops": "1", "pool_npu": "1", "profile": "1"})
    npu = run(model, deleg)
    md = float(np.max(np.abs(cpu - npu)))
    ok = md <= 0.02
    fails += 0 if ok else 1
    print(f"[{kind} {k}x{k} s{s}] cpu-vs-delegate max_abs={md:.5f} -> {'PASS' if ok else 'FAIL'}",
          flush=True)

print("==== %s ====" % ("PASS" if fails == 0 else "FAIL"))
sys.exit(1 if fails else 0)
