#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""A/B: how well does each delegate path track CPU TFLite on a real image?
Runs CPU (reference), delegate native-u8 (native_int8=1), delegate fp16-approx
(native_int8=0), and for each delegate matches CPU detections (score>0.30) to the
best same-class IoU delegate detection, reporting mean IoU + mean |Δscore|.
Usage: detcmp_ab.py model.tflite image.jpg /path/to/libtflite_rocket.so"""
import sys, numpy as np
from PIL import Image
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate, OpResolverType
    NDR = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
except ImportError:
    try:
        # LiteRT, the maintained successor to tflite_runtime and the only wheel on
        # recent Python; same interpreter + external-delegate + resolver API.
        from ai_edge_litert.interpreter import Interpreter, load_delegate, OpResolverType
        NDR = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate
        NDR = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES

model, img_path, deleg = sys.argv[1], sys.argv[2], sys.argv[3]

def prep(detail, img_path):
    _, H, W, _ = detail["shape"]
    im = Image.open(img_path).convert("RGB").resize((W, H))
    return np.asarray(im).astype(detail["dtype"]).reshape(1, H, W, 3)

def run(native=None):
    dl, kw = None, {}
    if native is not None:
        dl = [load_delegate(deleg, options={"native_int8": native})]
        kw["experimental_op_resolver_type"] = NDR
    it = Interpreter(model_path=model, experimental_delegates=dl, **kw)
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    it.set_tensor(ind["index"], prep(ind, img_path))
    it.invoke()
    outs = it.get_output_details()
    arrs = [it.get_tensor(d["index"]) for d in outs]
    boxes = next(a for a in arrs if a.ndim == 3 and a.shape[-1] == 4)[0]
    twod = [a for a in arrs if a.ndim == 2]
    scores = min(twod, key=lambda a: a.max())[0]
    classes = max(twod, key=lambda a: a.max())[0]
    return boxes, classes, scores

def iou(a, b):
    yA, xA = max(a[0], b[0]), max(a[1], b[1])
    yB, xB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, yB - yA) * max(0, xB - xA)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

bc, cc, sc = run(None)             # CPU reference
ci = np.where(sc > 0.30)[0]
print(f"== {img_path}: {len(ci)} CPU detections (score>0.30)")

for label, native in [("native-u8", "1"), ("fp16-approx", "0")]:
    bd, cd, sd = run(native)
    ious, dscores, miss = [], [], 0
    for i in ci:
        cand = [(iou(bc[i], bd[j]), j) for j in range(len(sd))
                if int(cd[j]) == int(cc[i]) and sd[j] > 0.10]
        if cand:
            best = max(cand); j = best[1]
            ious.append(best[0]); dscores.append(abs(sc[i] - sd[j]))
        else:
            miss += 1
    n = len(ious)
    if n == 0:
        print(f"  {label:12} matched 0/{len(ci)} miss={miss}  (no matched detections)")
        continue
    print(f"  {label:12} matched {n}/{len(ci)} miss={miss}  "
          f"meanIoU={np.mean(ious):.4f}  minIoU={np.min(ious):.4f}  "
          f"mean|Δscore|={np.mean(dscores):.4f}  max|Δscore|={np.max(dscores):.4f}")
