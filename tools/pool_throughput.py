#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# multi-instance context-pool THROUGHPUT harness for the tflite-rocket delegate.
#
# Detection latency on a single stream is host-bound (pack + readback + dispatch dominate
# the small 1x1 ops), so video-rate for a multi-camera deployment (Frigate) comes from
# THROUGHPUT — running several detection contexts concurrently so one context's CPU phase
# overlaps another's NPU phase — NOT from a faster per-conv kernel. The driver gate
# tests/ctx_pool_throughput.c measured up to ~3.9x at P=4, but ONLY when the contexts are
# spread across the four A76 big cores; left alone the library auto-affinity pins every
# 1-thread context's worker to the same big core (idx 0) and the pool caps at ~2.1x.
#
# THE RECIPE (no delegate/driver code change — affinity is env-controlled):
#   run one PROCESS per stream (exactly how Frigate runs cameras), and in each process set
#     ROCKET_CPU_AFFINITY=<one distinct big core>   (e.g. "4","5","6","7")
#   before the delegate loads. The delegate's worker then pins to THAT core, so P processes
#   land on P distinct cores. (One shared-process / multi-thread pool can't differ the env
#   per instance — it would need a per-context core-base API; multi-process is the Frigate case.)
#
# This harness spawns P such processes against one representative submit-bound model and
# reports aggregate inferences/sec and the speedup vs P=1.
#
# Run on RK3588 hardware (needs a device + tensorflow):
#   sudo modprobe rocket rocket_npu_clk_hz=600000000
#   python3 pool_throughput.py /path/to/libtflite_rocket.so [--big 4,5,6,7] [--iters 200]

import argparse, os, sys, time, multiprocessing as mp

def make_model(path):
    """A stack of 1x1 convs = the submit-bound detection unit (each 1x1 routes to the
    resident matmul; many small ops per inference make the dispatch floor dominate)."""
    import tensorflow as tf
    x = tf.keras.Input(shape=(32, 32, 64), batch_size=1)
    y = x
    for i in range(8):
        y = tf.keras.layers.Conv2D(64, 1, padding='same', name=f'pw{i}')(y)
    m = tf.keras.Model(x, y)
    with open(path, 'wb') as f: f.write(tf.lite.TFLiteConverter.from_keras_model(m).convert())

def worker(args):
    """One detection context, pinned to a single big core, running `iters` inferences."""
    so, model, core, iters = args
    os.environ['ROCKET_CPU_AFFINITY'] = str(core)         # delegate worker pins to THIS core
    try:
        os.sched_setaffinity(0, {core})                   # and the process itself
    except OSError:
        pass
    import numpy as np, tensorflow as tf
    deleg = tf.lite.experimental.load_delegate(so, options={'nthreads': '1'})
    it = tf.lite.Interpreter(model_path=model, experimental_delegates=[deleg])
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.default_rng(core).standard_normal(inp['shape']).astype(np.float32)
    it.set_tensor(inp['index'], x)
    it.invoke()                                           # warm (clock ride-up + first-run pack)
    t0 = time.perf_counter()
    for _ in range(iters):
        it.invoke()
    dt = time.perf_counter() - t0
    return iters / dt                                     # this context's inferences/sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('delegate')
    ap.add_argument('--big', default='4,5,6,7', help='big-core ids to spread across')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--model', default='/tmp/pool_unit.tflite')
    a = ap.parse_args()
    big = [int(c) for c in a.big.split(',')]
    if not os.path.exists(a.model):
        make_model(a.model)
        print(f'wrote {a.model}')
    base = None
    print(f'P  aggregate_inf/s  speedup   per-ctx(min..max)')
    for P in range(1, len(big) + 1):
        cores = big[:P]
        with mp.get_context('spawn').Pool(P) as pool:
            rates = pool.map(worker, [(a.delegate, a.model, c, a.iters) for c in cores])
        agg = sum(rates)
        if base is None:
            base = agg
        print(f'{P}  {agg:13.1f}  {agg/base:6.2f}x  {min(rates):6.1f}..{max(rates):.1f}')

if __name__ == '__main__':
    main()
