#!/usr/bin/env python
"""Command-line entry point for the hidden-benchmark run. Offline, CPU, no network access needed.

  python run.py verify --a img1.jpg --b img2.jpg
  python run.py embed  --images DIR --out embeddings.npz
  python run.py pairs  --csv pairs.csv --out scores.csv [--root DIR]

pairs.csv needs two path columns (default names: img1,img2; any first two columns are used if those are absent).
Output: score = cosine similarity in [-1, 1] (higher = same person); decision = score >= threshold (config.json).
Images with no detectable face get status=no_face, score=-1.0, decision=0 (reject) and never crash the run.
"""
import argparse, csv, json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import Config, FacePipeline  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_pipeline():
    cfgj = json.load(open(os.path.join(HERE, "config.json")))
    rec = cfgj.get("rec_onnx")
    rec = os.path.join(HERE, rec) if rec else None
    return FacePipeline(Config(**cfgj.get("pipeline", {})), rec_onnx=rec), cfgj["threshold"]


def embed_path(pipe, path):
    r = pipe.process(cv2.imread(path))
    return r["status"], r.get("emb")


def score(e1, e2):
    return float(np.dot(e1, e2)) if e1 is not None and e2 is not None else -1.0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--a", required=True); v.add_argument("--b", required=True)
    e = sub.add_parser("embed"); e.add_argument("--images", required=True); e.add_argument("--out", default="embeddings.npz")
    p = sub.add_parser("pairs"); p.add_argument("--csv", required=True); p.add_argument("--out", default="scores.csv")
    p.add_argument("--root", default="")
    a = ap.parse_args()
    pipe, thr = load_pipeline()

    if a.cmd == "verify":
        (sa, ea), (sb, eb) = embed_path(pipe, a.a), embed_path(pipe, a.b)
        s = score(ea, eb)
        print(json.dumps({"score": s, "decision": int(s >= thr), "threshold": thr, "status": [sa, sb]}))
    elif a.cmd == "embed":
        names, embs, status = [], [], []
        for f in sorted(os.listdir(a.images)):
            if f.lower().endswith(IMG_EXT):
                st, em = embed_path(pipe, os.path.join(a.images, f))
                names.append(f); status.append(st); embs.append(em if em is not None else np.zeros(512, np.float32))
        np.savez(a.out, names=np.array(names), emb=np.stack(embs) if embs else np.zeros((0, 512)), status=np.array(status))
        print(f"wrote {len(names)} embeddings to {a.out}; no_face={status.count('no_face')} rejected={status.count('rejected')}")
    else:
        cache = {}

        def get(path):
            if path not in cache:
                cache[path] = embed_path(pipe, os.path.join(a.root, path))
            return cache[path]
        with open(a.csv, newline="") as fi, open(a.out, "w", newline="") as fo:
            rd = csv.DictReader(fi)
            c1, c2 = ("img1", "img2") if {"img1", "img2"} <= set(rd.fieldnames) else rd.fieldnames[:2]
            wr = csv.writer(fo); wr.writerow(["img1", "img2", "score", "decision", "status1", "status2"])
            for row in rd:
                (s1, e1), (s2, e2) = get(row[c1]), get(row[c2])
                s = score(e1, e2)
                wr.writerow([row[c1], row[c2], f"{s:.6f}", int(s >= thr), s1, s2])
        print("wrote", a.out)


if __name__ == "__main__":
    main()
