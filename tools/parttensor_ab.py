#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where does a delegated graph first depart from CPU TFLite?

A model-level score says a graph is wrong; it does not say WHERE. Every tensor a
delegate node writes is a partition OUTPUT and is a real tensor in the interpreter,
so it can be read back and compared against the CPU interpreter's own value for the
same input. The first partition output that departs is the one to localise inside.

The distance is reported in QUANTIZED COUNTS, because that is the unit the part's
arithmetic is exact in and the unit a per-op gate is quoted in.

Usage:
  parttensor_ab.py model.tflite --delegate libtflite_rocket.so --image img.jpg
                   [--option k=v]... [--dump out.npz]
"""
import argparse, os, sys
import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
    NDR = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
except ImportError:
    from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
    NDR = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


def prep(detail, path):
    _, H, W, _ = detail["shape"]
    im = Image.open(path).convert("RGB").resize((W, H))
    a = np.asarray(im)
    if detail["dtype"] == np.uint8:
        return a.astype(np.uint8).reshape(1, H, W, 3)
    if detail["dtype"] == np.int8:
        return (a.astype(np.int32) - 128).astype(np.int8).reshape(1, H, W, 3)
    return (a.astype(np.float32) / 127.5 - 1.0).reshape(1, H, W, 3)


def run(model, image, delegate=None, options=None, preserve=False):
    kw = {}
    if delegate:
        kw["experimental_delegates"] = [load_delegate(delegate, options=options or {})]
        kw["experimental_op_resolver_type"] = NDR
    else:
        kw["experimental_op_resolver_type"] = NDR
    if preserve:
        kw["experimental_preserve_all_tensors"] = True
    it = Interpreter(model_path=model, **kw)
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    it.set_tensor(ind["index"], prep(ind, image))
    it.invoke()
    return it


def tensors_of(it, idxs):
    out = {}
    for i in idxs:
        try:
            out[i] = np.array(it.get_tensor(i))
        except (ValueError, RuntimeError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--delegate", required=True)
    ap.add_argument("--option", action="append", default=[])
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    opts = dict(kv.split("=", 1) for kv in a.option)

    # BOTH runs preserve. Without it the arena reuses a partition output's bytes for a
    # later tensor, so reading it back after invoke() scores a tensor that is no longer
    # there — and the CONTROL arm, which is at parity, reports hundreds of counts.
    dl = run(a.model, a.image, a.delegate, opts, preserve=True)
    # `_get_ops_details()` returns the model's OWN ops plus the substituted nodes, so a
    # partition is the node the runtime named DELEGATE. Matched case-insensitively: the
    # name is the runtime's, not ours.
    ops = [o for o in dl._get_ops_details() if "delegate" in o["op_name"].lower()]
    part_out, part_in = [], []
    for op in ops:
        part_out.extend(int(t) for t in op["outputs"])
        part_in.extend(int(t) for t in op["inputs"])
    part_out = sorted(set(part_out))
    part_in = sorted(set(part_in))
    if not part_out:
        print("no delegate node in the plan — nothing to compare", file=sys.stderr)
        return 2

    cpu = run(a.model, a.image, preserve=True)
    ref = tensors_of(cpu, part_out + part_in)
    got = tensors_of(dl, part_out + part_in)

    det = {int(d["index"]): d for d in cpu.get_tensor_details()}

    print("%d delegate node(s), %d partition output tensor(s)" % (len(ops), len(part_out)))
    print("%-6s %-46s %-16s %9s %9s %8s" %
          ("tensor", "name", "shape", "maxdiff", "meanabs", "!=%"))
    worst = []
    for i in part_out:
        if i not in ref or i not in got:
            continue
        r, g = ref[i].astype(np.float64), got[i].astype(np.float64)
        if r.shape != g.shape:
            print("%-6d %-46s SHAPE %s vs %s" % (i, det[i]["name"][:46], r.shape, g.shape))
            continue
        d = np.abs(r - g)
        nz = float((d > 0).mean() * 100.0)
        print("%-6d %-46s %-16s %9.3f %9.4f %7.2f%%" %
              (i, det[i]["name"][-46:], str(list(r.shape)), d.max(), d.mean(), nz))
        worst.append((d.max(), i))
    if a.dump:
        np.savez(a.dump, **{("t%d_ref" % i): ref[i] for i in part_out if i in ref},
                 **{("t%d_got" % i): got[i] for i in part_out if i in got})
    worst.sort(reverse=True)
    if worst:
        print("worst partition output: tensor %d at %g counts" % (worst[0][1], worst[0][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
