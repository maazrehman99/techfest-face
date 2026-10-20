"""Evaluate the pipeline on a manifest CSV and write metrics, per-group tables, ablation and ROC plots.

Manifest columns: path, identity, split (train|val|test), plus any optional group columns
(gender, age_bracket, covering, glasses, mask, condition, ...). Identities never cross splits (checked).

  python evaluate.py --manifest data/lfw/manifest.csv --out results/lfw --ablation
"""
import argparse, json, os, time
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import tqdm
from pipeline import Config, FacePipeline
from evalkit import make_pairs, scores_for, eer_and_thr, thr_at_far, full_report

RESERVED = {"path", "identity", "split", "status", "det", "blur", "bright", "size"}


def check_disjoint(df):
    s = {k: set(g.identity) for k, g in df.groupby("split")}
    for a in s:
        for b in s:
            if a < b and s[a] & s[b]:
                raise SystemExit(f"IDENTITY LEAK between {a} and {b}: {sorted(s[a] & s[b])[:5]}")


_PIPE = None


def _init_worker():
    cv2.setNumThreads(1)
    global _PIPE
    _PIPE = FacePipeline()


def _work(path):
    return _PIPE.analyze(cv2.imread(path))


def extract_all(df, root, cache, workers=int(os.environ.get("FACE_WORKERS", "6"))):
    """Single expensive pass: every embedding variant + quality stats per image (pickle-cached).
    Runs `workers` processes with 2 ONNX threads each (one process with all threads was ~10x slower)."""
    import pickle
    from multiprocessing import Pool
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    paths = [os.path.join(root, p) for p in df.path]
    with Pool(workers, initializer=_init_worker) as pool:
        infos = list(tqdm(pool.imap(_work, paths, chunksize=8), total=len(paths), desc="analyze"))
    pickle.dump(infos, open(cache, "wb"))
    return infos


def extract(infos, cfg):
    embs, meta = [], []
    for inf in infos:
        st, e = FacePipeline.decide(inf, cfg)
        embs.append(e if e is not None else np.zeros(512, np.float32))
        meta.append({"status": st, **{k: inf.get(k, np.nan) for k in ("det", "blur", "bright", "size")}})
    return np.stack(embs), pd.DataFrame(meta)


def group_columns(df, meta):
    d = pd.concat([df.reset_index(drop=True), meta], axis=1)
    ok = d[d.status == "ok"]
    if len(ok):
        d["blur_bin"] = pd.qcut(d.blur.rank(method="first"), 3, labels=["blurry", "mid", "sharp"]) if d.blur.notna().sum() > 3 else "na"
        d["bright_bin"] = pd.cut(d.bright, [0, 80, 170, 256], labels=["dark", "normal", "bright"])
        d["size_bin"] = pd.cut(d["size"], [0, 60, 120, 1e5], labels=["small<60px", "mid", "large>120px"])
    cols = [c for c in d.columns if c not in RESERVED and d[c].nunique() <= 12 and d[c].nunique() > 1 and c != "identity"]
    return d, cols


def run_config(df, cfg, infos, seed=0):
    emb, meta = extract(infos, cfg)
    d, gcols = group_columns(df, meta)
    ok = (d.status == "ok").values
    res = {"config": cfg.name(), "n_images": int(len(d)),
           "FTA_no_face": float((d.status == "no_face").mean()), "FTA_rejected": float((d.status == "rejected").mean())}

    def pairs(split, mask=None):
        m = ok & (d.split == split).values
        if mask is not None:
            m &= mask
        idx = np.where(m)[0]
        I, J, y = make_pairs(d.identity.values[idx], max_pos=20000, seed=seed)
        return idx[I], idx[J], y

    vI, vJ, vy = pairs("val")
    vs = scores_for(emb, vI, vJ)
    _, thr_eer = eer_and_thr(vy, vs)
    thr_far = thr_at_far(vy, vs, 1e-2)
    res["thr_val_EER"], res["thr_val_FAR1e-2"] = thr_eer, thr_far
    tI, tJ, ty = pairs("test")
    ts = scores_for(emb, tI, tJ)
    res["test"] = full_report(ty, ts, thr_eer)  # threshold fixed from VALIDATION, never the test set
    res["test_at_FAR1e-2_thr"] = full_report(ty, ts, thr_far)
    res["_roc"] = (ty, ts)
    groups = {}
    for col in gcols:
        for val in d[col].dropna().unique():
            msk = ((d[col] == val).values)
            I, J, y = make_pairs(d.identity.values[np.where(ok & (d.split == "test").values & msk)[0]], max_pos=5000, seed=seed)
            if len(set(y)) < 2 or (y == 1).sum() < 20:
                continue
            idx = np.where(ok & (d.split == "test").values & msk)[0]
            groups[f"{col}={val}"] = full_report(y, scores_for(emb, idx[I], idx[J]), thr_eer)
    res["groups"] = groups
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="folder paths in the manifest are relative to (default: manifest dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.manifest)
    root = a.root or os.path.dirname(a.manifest)
    check_disjoint(df)
    cfgs = [("full pipeline", Config())]
    if a.ablation:
        cfgs = [("full pipeline", Config()),
                ("- quality filter", Config(quality_filter=False)),
                ("- alignment (bbox crop)", Config(align=False)),
                ("+ flip TTA", Config(flip_tta=True)),
                ("minimal (no align, no filter)", Config(align=False, quality_filter=False))]
    infos = extract_all(df, root, os.path.join(a.out, "infos.pkl"))
    results = {}
    for name, cfg in cfgs:
        results[name] = run_config(df, cfg, infos)
    rows = []
    plt.figure(figsize=(5, 5))
    for name, r in results.items():
        ty, ts = r.pop("_roc")
        fpr, tpr, _ = roc_curve(ty, ts)
        plt.plot(fpr, tpr, label=f"{name} (EER {r['test']['EER']*100:.2f}%)")
        t = r["test"]
        rows.append({"config": name, "EER%": t["EER"] * 100, "AUC": t["AUC"], "acc%": t["acc"] * 100, "P": t["precision"],
                     "R": t["recall"], "F1": t["f1"], "FAR%": t["FAR"] * 100, "FRR%": t["FRR"] * 100,
                     "TAR@FAR1e-2": t["TAR@FAR=1e-2"], "TAR@FAR1e-3": t["TAR@FAR=1e-3"],
                     "FTA%": (r["FTA_no_face"] + r["FTA_rejected"]) * 100, "pairs": t["pairs"]})
    plt.xscale("log"); plt.xlabel("FAR"); plt.ylabel("TAR (1-FRR)"); plt.grid(alpha=.3); plt.legend(fontsize=7)
    plt.title("ROC (test, thresholds from val)"); plt.tight_layout(); plt.savefig(os.path.join(a.out, "roc.png"), dpi=150)
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    first = next(iter(results.values()))
    g = pd.DataFrame(first["groups"]).T.reset_index().rename(columns={"index": "group"})
    if len(g):
        g[["group", "pairs", "EER", "acc", "FAR", "FRR", "TAR@FAR=1e-2"]].round(4).to_csv(os.path.join(a.out, "per_group.csv"), index=False)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(tab.to_string(index=False))
    if len(g):
        print(g[["group", "pairs", "EER", "acc", "FAR", "FRR"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
