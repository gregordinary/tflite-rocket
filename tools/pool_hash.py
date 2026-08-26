#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""pool_hash.py — P concurrent delegate PROCESSES, each hashing its own outputs.

The multi-camera deployment shape (one process per stream) is also the concurrency
CORRECTNESS test, which pool_throughput.py does not make: several contexts share one
NPU, and on some drivers one IOMMU domain and one set of cores, so an operand-reuse or
scheduling fault between them produces a full, correctly sized, entirely plausible wrong
surface. This asserts instead that every process's output hash equals every other's, and
that each process is deterministic across its own repeats — the two failures a rate
number cannot see.

Reports, per P: aggregate inferences/sec, the speedup over P=1, and how many distinct
output hashes the pool produced (1 is the pass).

Usage: pool_hash.py delegate.so model.tflite image [--big 4,5,6,7] [--iters 30]
Under LiteRT, LD_PRELOAD the libtflite_cshim.so the build produces.
"""
import argparse, os, sys, time, hashlib, multiprocessing as mp

def worker(args):
    so, model, img, core, iters = args
    os.environ['ROCKET_CPU_AFFINITY'] = str(core)
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        pass
    import numpy as np
    from PIL import Image
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
    deleg = load_delegate(so, options={'nthreads': '1', 'native_int8': '1'})
    it = Interpreter(model_path=model, experimental_delegates=[deleg],
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    _, H, W, _ = ind['shape']
    x = np.asarray(Image.open(img).convert('RGB').resize((W, H))).astype(ind['dtype']).reshape(1, H, W, 3)
    it.set_tensor(ind['index'], x)
    it.invoke()                                  # warm
    t0 = time.perf_counter()
    digests = set()
    for _ in range(iters):
        it.invoke()
        h = hashlib.md5()
        for d in it.get_output_details():
            h.update(np.ascontiguousarray(it.get_tensor(d['index'])).tobytes())
        digests.add(h.hexdigest())
    dt = time.perf_counter() - t0
    return core, sorted(digests), iters / dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('delegate'); ap.add_argument('model'); ap.add_argument('image')
    ap.add_argument('--big', default='4,5,6,7')
    ap.add_argument('--iters', type=int, default=30)
    a = ap.parse_args()
    big = [int(c) for c in a.big.split(',')]
    ref, base = None, None
    for P in range(1, len(big) + 1):
        cores = big[:P]
        with mp.get_context('spawn').Pool(P) as pool:
            res = pool.map(worker, [(a.delegate, a.model, a.image, c, a.iters) for c in cores])
        agg = sum(r[2] for r in res)
        base = base or agg
        hs = set()
        for _, ds, _ in res:
            hs.update(ds)
        ref = ref or (sorted(hs)[0] if len(hs) == 1 else None)
        ok = (len(hs) == 1 and (ref is None or hs == {ref}))
        print(f'P={P}  agg={agg:6.2f} inf/s  {agg/base:5.2f}x  '
              f'hashes={len(hs)} {"OK" if ok else "MISMATCH " + str(sorted(hs))}')
        if not ok:
            for c, ds, r in res:
                print(f'   core {c}: {ds}  {r:.2f} inf/s')
    print(f'reference hash: {ref}')

if __name__ == '__main__':
    main()
