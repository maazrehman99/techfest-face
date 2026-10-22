"""Identification benchmark in the organizers' format: one gallery photo per person, the rest are probes,
top-1 / top-5 over all people of a split. Compares recognizers on the SAME aligned crops.
usage: python tools/id_bench.py <manifest.csv> <root> [split=test] [max_people]"""
import sys, os, csv, random, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import FacePipeline, imread, _unit
from insightface.utils.face_align import norm_crop
from insightface.model_zoo import get_model

man, root = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) > 3 else "test"
maxp = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
rows = [r for r in csv.DictReader(open(man, encoding="utf-8")) if r.get("split", split) == split]
ids = sorted({r["identity"] for r in rows})[:maxp]
rows = [r for r in rows if r["identity"] in ids]

pipe = FacePipeline()
r100 = get_model("models/glintr100.onnx", providers=["CPUExecutionProvider"]); r100.prepare(ctx_id=-1)

def crop(bgr):
    if max(bgr.shape[:2]) < 300: bgr = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if max(bgr.shape[:2]) < 700:
        p = int(0.35 * max(bgr.shape[:2])); bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REFLECT)
    bgr, faces = pipe._detect(bgr)
    if not faces: return None
    f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return norm_crop(bgr, f.kps, 112)

E = {"r50": [], "r100": []}; lab = []; miss = 0
for r in rows:
    im = imread(os.path.join(root, r["path"]))
    c = None if im is None else crop(im)
    if c is None: miss += 1; continue
    for k, m in (("r50", pipe.rec), ("r100", r100)):
        e = m.get_feat([c, c[:, ::-1].copy()]); E[k].append(_unit(e[0] + e[1]))
    lab.append(r["identity"])
lab = np.array(lab); print(f"{len(ids)} people, {len(lab)} imgs, no-face {miss}")
E = {k: np.array(v) for k, v in E.items()}
E["avg"] = np.hstack([E["r50"], E["r100"]]) / np.sqrt(2)
for k, X in E.items():
    t1 = t5 = n = 0
    for seed in range(10):
        rnd = random.Random(seed); g = {}
        for p in ids:
            idx = np.where(lab == p)[0]
            if len(idx) >= 2: g[p] = rnd.choice(list(idx))
        gp = list(g); G = X[[g[p] for p in gp]]
        for i in range(len(lab)):
            if lab[i] not in g or i == g[lab[i]]: continue
            s = G @ X[i]; order = [gp[j] for j in np.argsort(-s)]
            rank = order.index(lab[i]) + 1; t1 += rank == 1; t5 += rank <= 5; n += 1
    print(f"{k:5s} top1 {100*t1/n:.1f}%  top5 {100*t5/n:.1f}%  (n={n})")
