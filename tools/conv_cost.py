#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-DIRECT-conv box-sum vs requant cost model for the native-uint8 path.
box-sum work ~ OH*OW*IC*KH*KW (adds, only when w_zp asymmetric).
requant work ~ OH*OW*OC. Channel-reduce+separable box-sum ~ IC*IHp*IWp + IHp*OW*KW + OH*OW*KH."""
import sys
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

def main(path):
    it = Interpreter(model_path=path); it.allocate_tensors()
    det = it._get_ops_details() if hasattr(it,"_get_ops_details") else it._get_op_details()
    by = {t["index"]: t for t in it.get_tensor_details()}
    tot_box=tot_box_sep=tot_req=0; n_box=0; n1x1=0; nkxk=0
    rows=[]
    for o in det:
        if o["op_name"]!="CONV_2D": continue   # DIRECT only (DEPTHWISE handled separately)
        ins,outs=o["inputs"],o["outputs"]
        ti,tw,to=by[ins[0]],by[ins[1]],by[outs[0]]
        if tw["dtype"].__name__!="uint8": continue
        ish=ti["shape"]; wsh=tw["shape"]; osh=to["shape"]
        IH,IW,IC=int(ish[1]),int(ish[2]),int(ish[3])
        OC,KH,KW=int(wsh[0]),int(wsh[1]),int(wsh[2])
        OH,OW=int(osh[1]),int(osh[2])
        wz=tw["quantization_parameters"]["zero_points"]
        asym = not all(int(z)==128 for z in wz)
        # padded input extent (SAME assumed: IHp~IH+KH-1 worst case; use stride from out)
        sy = max(1, round(IH/OH)); sx=max(1, round(IW/OW))
        IHp=(OH-1)*sy+KH; IWp=(OW-1)*sx+KW
        box = OH*OW*IC*KH*KW
        box_sep = IC*IHp*IWp + IHp*OW*KW + OH*OW*KH
        req = OH*OW*OC
        if KH==1 and KW==1: n1x1+=1
        else: nkxk+=1
        if asym:
            n_box+=1; tot_box+=box; tot_box_sep+=box_sep
        tot_req+=req
        rows.append((box if asym else 0, f"{KH}x{KW} IC={IC:4} OC={OC:4} OH={OH:3} OW={OW:3} "
                     f"asym={int(asym)} box={box:9} sep={box_sep:8} req={req:7}"))
    rows.sort(reverse=True)
    print(f"== {path}")
    print(f"   DIRECT uint8 convs: {n1x1} 1x1 + {nkxk} KxK ; {n_box} need box-sum")
    print(f"   TOTAL box(naive)={tot_box:,}  box(separable)={tot_box_sep:,}  "
          f"({tot_box/max(1,tot_box_sep):.1f}x)  requant={tot_req:,}")
    print(f"   box/(box+req) naive = {100*tot_box/max(1,tot_box+tot_req):.0f}%   "
          f"separable = {100*tot_box_sep/max(1,tot_box_sep+tot_req):.0f}%")
    print("   --- top convs by box-sum cost ---")
    for _,r in rows[:14]: print("   ",r)

if __name__=="__main__":
    main(sys.argv[1])
