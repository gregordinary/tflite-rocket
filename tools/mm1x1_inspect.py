#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enumerate the 1x1 stride-1 CONV_2D ops in a TFLite model and report whether each
is alignment-eligible for the resident int8 matmul (K=IC%32==0, N=OC%32==0, M%4||M==1,
where M = IH*IW). Tells us how many of the single-tile 1x1 convs route to the matmul
for free vs. would need host K/N zero-padding. Direct (non-depthwise) convs only."""
import sys, collections
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def main(path):
    it = Interpreter(model_path=path); it.allocate_tensors()
    det = it._get_ops_details() if hasattr(it, "_get_ops_details") else it._get_op_details()
    by_idx = {t["index"]: t for t in it.get_tensor_details()}
    print(f"== {path}")
    n_1x1 = aligned = need_kpad = need_npad = need_mpad = 0
    rows = []
    for o in det:
        if o["op_name"] != "CONV_2D":
            continue
        ins, outs = o["inputs"], o["outputs"]
        w = by_idx[ins[1]]; itn = by_idx[ins[0]]; out = by_idx[outs[0]]
        wsh = w["shape"]                         # [OC,KH,KW,IC]
        if len(wsh) != 4 or wsh[1] != 1 or wsh[2] != 1:
            continue
        OC, KH, KW, IC = [int(x) for x in wsh]
        ish = itn["shape"]; osh = out["shape"]   # [1,IH,IW,IC] / [1,OH,OW,OC]
        IH, IW = int(ish[1]), int(ish[2])
        M = IH * IW
        ka = (IC % 32 == 0); na = (OC % 32 == 0); ma = (M % 4 == 0 or M == 1)
        n_1x1 += 1
        elig = ka and na and ma
        aligned += elig
        if not ka: need_kpad += 1
        if not na: need_npad += 1
        if not ma: need_mpad += 1
        rows.append((M, IC, OC, ka, na, ma, elig, itn["dtype"].__name__))
    rows.sort(key=lambda r: -r[0] * r[1] * r[2])   # by MACs desc
    print(f"   {n_1x1} 1x1 stride-1 convs;  aligned(no pad)={aligned}  "
          f"need_Kpad={need_kpad} need_Npad={need_npad} need_Mpad={need_mpad}")
    print(f"   {'M':>7} {'IC':>5} {'OC':>5} {'MACs':>12}  K%32 N%32 M%4  elig dtype")
    for M, IC, OC, ka, na, ma, elig, dt in rows:
        print(f"   {M:>7} {IC:>5} {OC:>5} {M*IC*OC:>12}  "
              f"{'Y' if ka else 'n':>4} {'Y' if na else 'n':>4} {'Y' if ma else 'n':>3} "
              f"{'YES' if elig else '-':>5} {dt}")
    macs_elig = sum(M*IC*OC for M,IC,OC,ka,na,ma,e,dt in rows if e)
    macs_all  = sum(M*IC*OC for M,IC,OC,ka,na,ma,e,dt in rows)
    print(f"   MACs: eligible={macs_elig/1e6:.1f}M / total-1x1={macs_all/1e6:.1f}M "
          f"({100*macs_elig/max(1,macs_all):.0f}%)")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p); print()
