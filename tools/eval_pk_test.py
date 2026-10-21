"""Small hand-picked Pakistani TEST set: original vs fine-tuned model through the full pipeline (detect -> align -> embed).

  python tools/eval_pk_test.py --dir data/pk_test --ft runs/ft1/finetuned.onnx [--tags data/pk_test/tags.csv]

Layout: data/pk_test/<PersonName>/<photo>.jpg (>= 2 photos per person, >= 8 people to say anything). Optional tags.csv
with columns `file` (e.g. "PersonName/1.jpg") and `covering` (none, dupatta, hijab, turban, cap, beard, mask, ...).
TEST ONLY: never train on these photos and never pick the model or the threshold from them.
The decision threshold is a fixed number from the validation split (config.json by default; pass --thr / --thr-ft for the
fine-tuned model's own val threshold). EER/AUC need no threshold but are optimistic (best cut-off on the test set itself).
Pairs: every same-person pair and every different-person pair. Intervals: bootstrap over people.
"""
import argparse, glob, itertools, json, os, sys
import cv2, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pipeline import Config, FacePipeline  # noqa: E402
from evalkit import eer_and_thr, tar_at_far  # noqa: E402


def imread(p):
    b = np.fromfile(p, np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_COLOR) if b.size else None


def embed_all(rec, files):
    pipe = FacePipeline(Config(), rec_onnx=rec)
    out = []
    for f in files:
        r = pipe.process(imread(f)) if os.path.getsize(f) else {"status": "no_face"}
        out.append(r["emb"] if r["status"] == "ok" else None)
    return out


def score(emb, ids, cov, thr, idx_people=None):
    ok = [i for i, e in enumerate(emb) if e is not None]
    pairs = [(i, j) for i, j in itertools.combinations(ok, 2)]
    y = np.array([ids[i] == ids[j] for i, j in pairs]).astype(int)
    s = np.array([float(np.dot(emb[i], emb[j])) for i, j in pairs])
    cv = np.array([(cov[i] != "none") or (cov[j] != "none") for i, j in pairs])
    return y, s, cv


def summary(y, s, cv, thr):
    pred = (s >= thr).astype(int)
    r = {"pairs": int(len(y)), "same": int(y.sum()), "acc": float((pred == y).mean()),
         "FAR": float(pred[y == 0].mean()) if (y == 0).any() else None, "FRR": float(1 - pred[y == 1].mean()) if y.any() else None}
    if y.any() and (y == 0).any():
        r.update(AUC=float(roc_auc_score(y, s)), EER_optimistic=eer_and_thr(y, s)[0], TAR_at_FAR1=tar_at_far(y, s, 0.01))
    m = (y == 1) & cv
    n = (y == 1) & ~cv
    r["TAR_same_pairs_with_covering"] = float(pred[m].mean()) if m.any() else None
    r["TAR_same_pairs_plain"] = float(pred[n].mean()) if n.any() else None
    r["n_same_with_covering"], r["n_same_plain"] = int(m.sum()), int(n.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/pk_test"); ap.add_argument("--ft", default=None); ap.add_argument("--tags", default=None)
    ap.add_argument("--thr", type=float, default=None); ap.add_argument("--thr-ft", type=float, default=None)
    ap.add_argument("--B", type=int, default=300); ap.add_argument("--out", default="results/pk_test.json")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "..", "config.json")))
    thr = a.thr if a.thr is not None else cfg["threshold"]
    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*", "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    ids = [os.path.basename(os.path.dirname(f)) for f in files]
    keep = [k for k, c in __import__("collections").Counter(ids).items() if c >= 2]
    files, ids = zip(*[(f, i) for f, i in zip(files, ids) if i in keep]) if keep else ([], [])
    cov = ["none"] * len(files)
    if a.tags and os.path.exists(a.tags):
        t = dict(pd.read_csv(a.tags, encoding="utf-8").values[:, :2])
        cov = [str(t.get(os.path.relpath(f, a.dir).replace("\\", "/"), "none")) for f in files]
    print(f"{len(files)} photos, {len(set(ids))} people (people with < 2 photos dropped)")
    res = {"photos": len(files), "people": len(set(ids)), "threshold_original": thr, "models": {}}
    models = {"original": (None, thr)}
    if a.ft:
        models["fine-tuned"] = (a.ft, a.thr_ft if a.thr_ft is not None else thr)
    rng = np.random.default_rng(0)
    for name, (rec, th) in models.items():
        emb = embed_all(rec, files)
        res["models"][name] = {"no_face_rate": float(np.mean([e is None for e in emb])), "threshold_used": th}
        y, s, cv = score(emb, ids, cov, th)
        res["models"][name].update(summary(y, s, cv, th))
        ppl = sorted(set(ids)); accs = []
        for _ in range(a.B):                                  # bootstrap over people
            d = rng.choice(ppl, len(ppl), replace=True)
            sel = [i for p in d for i in range(len(ids)) if ids[i] == p]
            lab = [f"{ids[i]}#{k}" for k, p in enumerate(d) for i in range(len(ids)) if ids[i] == p]
            e2 = [emb[i] for i in sel]; ok = [k for k, e in enumerate(e2) if e is not None]
            pr = [(i, j) for i, j in itertools.combinations(ok, 2) if ids[sel[i]] != ids[sel[j]] or lab[i] == lab[j]]
            if not pr: continue
            yy = np.array([lab[i] == lab[j] for i, j in pr]); ss = np.array([float(np.dot(e2[i], e2[j])) for i, j in pr])
            accs.append(((ss >= th).astype(int) == yy).mean())
        res["models"][name]["acc_95ci"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))] if accs else None
        print(name, json.dumps(res["models"][name], indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
