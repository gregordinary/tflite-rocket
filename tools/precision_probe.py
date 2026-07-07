#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# precision-vs-bug probe: build ONE model (fixed seed), emit a paired float and
# int8 tflite with IDENTICAL weights, then ask whether nchw_resident=1 (fp16-NCHW
# inter-op, skips int8 requant) lands CLOSER to the fp32 ground truth than pure int8
# (resident=0). If closer -> the delegate-vs-int8 delta is precision (acceptable, judge
# by mAP); if further -> a bug. Runs on a target RK3588 device (TensorFlow + the rocket delegate present).
import argparse, sys, os
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from make_ssd_head_tflite import build_head


def emit_pair(build_fn, hw, c_in, prefix):
    tf.keras.utils.set_random_seed(0)
    model = build_fn()
    # float
    cf = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tfl = cf.convert()
    with open(prefix + "_f.tflite", "wb") as f: f.write(float_tfl)
    # int8 (quantize the SAME in-memory model)
    rng = np.random.default_rng(0)
    def rep():
        for _ in range(64):
            yield [rng.standard_normal((1, hw, hw, c_in)).astype(np.float32)]
    ci = tf.lite.TFLiteConverter.from_keras_model(model)
    ci.optimizations = [tf.lite.Optimize.DEFAULT]
    ci.representative_dataset = rep
    ci.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    ci.inference_input_type = tf.int8
    ci.inference_output_type = tf.int8
    int8_tfl = ci.convert()
    with open(prefix + "_i8.tflite", "wb") as f: f.write(int8_tfl)
    return prefix + "_f.tflite", prefix + "_i8.tflite"


def load():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
        return Interpreter, load_delegate, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except ImportError:
        ndr = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        return tf.lite.Interpreter, tf.lite.experimental.load_delegate, ndr


def interp_for(Interpreter, model, delegate, ndr):
    kw = {}
    delegates = None
    if delegate is not None:
        delegates = [delegate]
        kw["experimental_op_resolver_type"] = ndr
    it = Interpreter(model_path=model, experimental_delegates=delegates, **kw)
    it.allocate_tensors()
    return it


def run(it, feeds):
    for d in it.get_input_details():
        it.set_tensor(d["index"], feeds[d["index"]])
    it.invoke()
    return it.get_output_details(), [it.get_tensor(d["index"]).copy() for d in it.get_output_details()]


def deq(out_detail, arr):
    s = out_detail["quantization_parameters"]["scales"]
    z = out_detail["quantization_parameters"]["zero_points"]
    if s.size == 0:
        return arr.astype(np.float64)
    return (arr.astype(np.float64) - z[0]) * s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delegate", required=True)
    ap.add_argument("--hw", type=int, default=8)
    ap.add_argument("--c-in", type=int, default=128)
    args = ap.parse_args()

    Interpreter, load_delegate, ndr = load()
    build_fn = lambda: build_head(tf, args.c_in, 64, args.hw, 2, 6, 20)
    f_model, i8_model = emit_pair(build_fn, args.hw, args.c_in, "/tmp/rocket_precision_ssd")

    # int8 input feed (what run_delegate --compare uses), and its real-valued version.
    base = Interpreter(model_path=i8_model); base.allocate_tensors()
    rng = np.random.default_rng(0)
    in_det = base.get_input_details()[0]
    s = in_det["quantization_parameters"]["scales"][0]
    z = in_det["quantization_parameters"]["zero_points"][0]
    info = np.iinfo(in_det["dtype"])
    x_i8 = rng.integers(info.min, info.max + 1, size=in_det["shape"], dtype=in_det["dtype"])
    x_real = (x_i8.astype(np.float64) - z) * s

    # fp32 ground truth: feed the float model the SAME real values.
    fit = interp_for(Interpreter, f_model, None, ndr)
    fdet = fit.get_input_details()[0]
    f_feeds = {fdet["index"]: x_real.astype(np.float32)}
    _, gt = run(fit, f_feeds)

    # int8 variants
    def run_i8(delegate):
        it = interp_for(Interpreter, i8_model, delegate, ndr)
        det = it.get_input_details()[0]
        _, outs = run(it, {det["index"]: x_i8})
        od = it.get_output_details()
        return [deq(od[i], outs[i]) for i in range(len(outs))]

    cpu = run_i8(None)
    r0 = run_i8(load_delegate(args.delegate, options={"nchw_resident": "0"}))
    r1 = load_delegate(args.delegate, options={"nchw_resident": "1"})
    r1 = run_i8(r1)

    print("output |  vs-fp32 max abs error  (lower = closer to truth)")
    print("       |   int8-CPU    resident=0   resident=1")
    for i in range(len(gt)):
        g = gt[i].astype(np.float64)
        e_cpu = np.abs(cpu[i] - g).max()
        e_r0 = np.abs(r0[i] - g).max()
        e_r1 = np.abs(r1[i] - g).max()
        verdict = "PRECISION (resident closer)" if e_r1 <= e_r0 else "WORSE (resident further -> loss/bug)"
        print(f"  [{i}]  |  {e_cpu:9.5f}   {e_r0:9.5f}   {e_r1:9.5f}   {verdict}")
        m_cpu = np.abs(cpu[i] - g).mean(); m_r0 = np.abs(r0[i] - g).mean(); m_r1 = np.abs(r1[i] - g).mean()
        print(f"       |  mean {m_cpu:.5f}    {m_r0:.5f}    {m_r1:.5f}")


if __name__ == "__main__":
    main()
