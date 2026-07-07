#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dump per-conv quantization params (dtype, zero-points, per-tensor vs per-axis
weight quant) for a TFLite model. Decides the uint8-native design: are the conv
weight zero-points symmetric (==128 in uint8 / ==0 in int8) or asymmetric?"""
import sys, collections
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def main(path):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    det = it._get_ops_details() if hasattr(it, "_get_ops_details") else it._get_op_details()
    tdetail = it.get_tensor_details()
    by_idx = {t["index"]: t for t in tdetail}

    conv_ops = [o for o in det if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    print(f"== {path}")
    print(f"   {len(conv_ops)} conv ops (of {len(det)} total ops)")

    # summary counters
    cls = collections.Counter()
    wzp_hist = collections.Counter()
    for o in conv_ops:
        ins = o["inputs"]; outs = o["outputs"]
        it_in  = by_idx[ins[0]]; it_w = by_idx[ins[1]]; it_out = by_idx[outs[0]]
        in_dt  = it_in["dtype"].__name__
        w_dt   = it_w["dtype"].__name__
        out_dt = it_out["dtype"].__name__
        in_zp  = int(it_in["quantization"][1]) if it_in["quantization"][0] else 0
        out_zp = int(it_out["quantization"][1]) if it_out["quantization"][0] else 0
        qp = it_w["quantization_parameters"]
        wz = qp["zero_points"]
        ws = qp["scales"]
        per_axis = len(ws) > 1
        wzset = sorted(set(int(z) for z in wz)) if len(wz) else [0]
        kind = f"{o['op_name'][:4]:4} {in_dt:7}/{w_dt:7}/{out_dt:7}"
        axis = "per-axis" if per_axis else "per-tens"
        # is the weight symmetric? int8 sym => wzp==0; uint8 sym => wzp==128
        wsym = all(z == 0 for z in wzset) if w_dt == "int8" else all(z == 128 for z in wzset)
        cls[(kind, axis, wsym)] += 1
        for z in wzset: wzp_hist[(w_dt, z)] += 1
        if len(conv_ops) <= 40 or not wsym:  # print non-symmetric always
            print(f"   {o['op_name'][:16]:16} in_zp={in_zp:4} out_zp={out_zp:4} "
                  f"w={w_dt} {axis} wzp={wzset[:6]} sym={wsym}")

    print("   --- summary (op kind, weight-axis, weight-symmetric) ---")
    for k, n in sorted(cls.items()):
        print(f"     {n:3}x  {k[0]}  {k[1]}  sym={k[2]}")
    print("   --- weight zero-point histogram (dtype, zp) -> count ---")
    for k, n in sorted(wzp_hist.items()):
        print(f"     {k[0]:7} zp={k[1]:4} : {n}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p); print()
