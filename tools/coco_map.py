#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# COCO-val mAP@[.5:.95] for an SSD-postprocess TFLite detector, CPU TFLite vs the rocket
# delegate (native_int8). The model carries TFLite_Detection_PostProcess, so its 4 outputs
# ARE detections (boxes [ymin,xmin,ymax,xmax] normalized, classes, scores, count) — no NMS.
#
# Usage:
#   coco_map.py <model.tflite> [--delegate <so> [--option k=v ...]] \
#               [--images DIR] [--ann JSON] [--limit N] [--score-thr T]
# CPU-only (no --delegate) validates the index->coco_id map + preprocessing: a correct
# pipeline gives MobileDet ~0.22-0.25 mAP; a wrong class map gives ~0.
import argparse, json, os, sys
import numpy as np
from PIL import Image
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# TF OD API SSD/MobileDet COCO postprocess convention: the detection class output is a
# 0-based index into the 90-entry mscoco label map, so COCO category_id = class_index + 1
# (verified empirically: this model emits class indices 0..89 — e.g. idx 84->clock(85),
# 85->vase(86) — and the +1 map gives MobileDet's expected ~0.25 mAP where the 80-contiguous
# map gave 0.05). Indices landing on COCO's "gap" ids never fire (the model isn't trained on
# them), so loadRes simply sees only valid ids.
def class_to_cat_id(ci):
    return ci + 1

def parse_options(pairs, parser):
    """Parse repeated --option k=v entries into a dict, erroring cleanly on a
    malformed entry (missing '=') instead of raising a bare ValueError."""
    opts = {}
    for kv in pairs:
        if "=" not in kv:
            parser.error(f"--option must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        opts[k] = v
    return opts

def load_delegate_iface():
    return tf.lite.experimental.load_delegate

def make_interp(model, delegate_so=None, opts=None):
    dl = [tf.lite.experimental.load_delegate(delegate_so, options=opts)] if delegate_so else None
    it = tf.lite.Interpreter(model_path=model, experimental_delegates=dl)
    it.allocate_tensors()
    return it

def find_outputs(it):
    """Return (boxes_detail, [the two [1,N] details], count_detail) by tensor shape."""
    boxes = count = None
    cand_1n = []
    for o in it.get_output_details():
        s = list(o["shape"])
        if len(s) == 3 and s[-1] == 4:   boxes = o
        elif np.prod(s) == 1:            count = o
        elif len(s) == 2:                cand_1n.append(o)
    return boxes, cand_1n, count

def run_image(it, img_rgb, in_det, boxes_o, cand_1n, count_o):
    H0, W0 = img_rgb.shape[:2]
    _, ih, iw, _ = in_det["shape"]
    im = np.asarray(Image.fromarray(img_rgb).resize((iw, ih), Image.BILINEAR))
    if in_det["dtype"] == np.uint8:
        x = im.astype(np.uint8)[None]
    else:
        sc, zp = in_det["quantization"]
        x = ((im.astype(np.float32)/ (sc if sc else 1.0)) + zp).astype(in_det["dtype"])[None] \
            if in_det["dtype"] != np.float32 else (im.astype(np.float32)/127.5 - 1.0)[None]
    it.set_tensor(in_det["index"], x)
    it.invoke()
    boxes = it.get_tensor(boxes_o["index"])[0]            # [N,4] ymin,xmin,ymax,xmax
    a = it.get_tensor(cand_1n[0]["index"])[0]
    b = it.get_tensor(cand_1n[1]["index"])[0]
    # The two [1,N] outputs are class indices and scores. Discriminate by
    # integer-valuedness first — class indices are whole numbers (0..89), scores almost
    # never are; this separates them even when a frame's classes all exceed 1. Fall back
    # to the in-[0,1] range test, then to descending-sort, only when both tensors look
    # integer-valued (e.g. every detection is class 0/1 with degenerate scores).
    int_valued = lambda t: np.allclose(t, np.rint(t), atol=1e-6)
    a_int, b_int = int_valued(a), int_valued(b)
    if a_int and not b_int:   classes, scores = a, b
    elif b_int and not a_int: classes, scores = b, a
    else:
        a_sc, b_sc = a.max() <= 1.01, b.max() <= 1.01
        if a_sc and not b_sc:   scores, classes = a, b
        elif b_sc and not a_sc: scores, classes = b, a
        else:                   # ambiguous -> the descending-sorted tensor is scores
            scores, classes = (a, b) if np.all(np.diff(a) <= 1e-6) else (b, a)
    return boxes, np.rint(classes).astype(int), scores, (H0, W0)

def detections(it, ids, imgs, img_dir, score_thr):
    in_det = it.get_input_details()[0]
    boxes_o, cand_1n, count_o = find_outputs(it)
    res = []
    for iid in ids:
        fn = imgs[iid]["file_name"]
        p = os.path.join(img_dir, fn)
        if not os.path.exists(p): continue
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]
        boxes, classes, scores, _ = run_image(it, img, in_det, boxes_o, cand_1n, count_o)
        for k in range(len(scores)):
            s = float(scores[k])
            if s < score_thr: continue
            ci = int(classes[k])
            if ci < 0 or ci > 89: continue
            ymin, xmin, ymax, xmax = [float(v) for v in boxes[k]]
            x, y = xmin*W0, ymin*H0
            w, h = (xmax-xmin)*W0, (ymax-ymin)*H0
            if w <= 0 or h <= 0: continue
            res.append({"image_id": int(iid), "category_id": class_to_cat_id(ci),
                        "bbox": [x, y, w, h], "score": s})
    return res

def evaluate(coco_gt, res, ids):
    if not res:
        print("  no detections -> mAP 0"); return 0.0
    dt = coco_gt.loadRes(res)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.imgIds = ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--delegate", default=None)
    ap.add_argument("--option", action="append", default=[], help="k=v delegate option")
    ap.add_argument("--images", default="/tmp/coco/val2017")
    ap.add_argument("--ann", default="/tmp/coco/annotations/instances_val2017.json")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--score-thr", type=float, default=0.01)
    args = ap.parse_args()

    coco_gt = COCO(args.ann)
    all_imgs = {im["id"]: im for im in coco_gt.dataset["images"]}
    have = {int(os.path.splitext(f)[0]) for f in os.listdir(args.images) if f.endswith(".jpg")}
    ids = [i for i in sorted(all_imgs) if i in have][:args.limit]
    print(f"evaluating {len(ids)} images")

    print("== CPU TFLite ==")
    cpu = make_interp(args.model)
    cpu_map = evaluate(coco_gt, detections(cpu, ids, all_imgs, args.images, args.score_thr), ids)

    deleg_map = None
    if args.delegate:
        opts = parse_options(args.option, ap)
        print(f"== rocket delegate {opts} ==")
        npu = make_interp(args.model, args.delegate, opts)
        deleg_map = evaluate(coco_gt, detections(npu, ids, all_imgs, args.images, args.score_thr), ids)

    print("\n==== SUMMARY ====")
    print(f"  CPU      mAP@[.5:.95] = {cpu_map:.4f}")
    if deleg_map is not None:
        print(f"  delegate mAP@[.5:.95] = {deleg_map:.4f}  (delta {deleg_map-cpu_map:+.4f})")

if __name__ == "__main__":
    main()
