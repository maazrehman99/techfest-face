"""Turn the raw Wikimedia download into a usable identity-labelled dataset.

  python process_wikimedia.py --root data/pk_wiki
For each image: detect faces, take the largest, reject tiny/low-confidence faces, save the aligned 112x112 crop.
Label cleaning (article images are noisy: other people, group photos, medals, crowds):
 1. the target face must dominate the image: largest face area >= --dominance x the second largest (drops group photos);
 2. ANCHORS: images whose file name contains the person's name (>= 2 name tokens, or the only token) are trusted without
    using the model; a person with no anchor is dropped;
 3. other images are kept only if their face matches the median embedding of the anchors (cosine >= --admit), so
    other people's photos in the article are rejected;
 4. keep people with >= --min-images images left.
Writes <root>/manifest.csv (path = 640px original, crop = aligned crop, identity, split, gender, licence, source) with an
identity-disjoint train/val/test split (70/15/15, stratified by gender). Never splits a person across sets.
"""
import argparse, json, os, re, warnings
import cv2, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pipeline import Config, FacePipeline, detect_all
from insightface.utils.face_align import norm_crop


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data/pk_wiki")
    ap.add_argument("--admit", type=float, default=0.40); ap.add_argument("--dominance", type=float, default=2.0); ap.add_argument("--min-face", type=int, default=40)
    ap.add_argument("--min-images", type=int, default=3)
    ap.add_argument("--max-born", type=int, default=2005, help="adults only: drop people born after this year")
    ap.add_argument("--exclude", default="exclude_people.txt", help="file with one person name per line to leave out (sensitive cases)"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.root, "records.jsonl"), encoding="utf-8")]
    excl = {l.strip().lower() for l in open(a.exclude, encoding="utf-8")} if os.path.exists(a.exclude) else set()
    n0 = len(recs)
    recs = [r for r in recs if int(r["born"]) <= a.max_born and r["name"].lower() not in excl]
    print(f"ethics filter: {n0 - len(recs)} images dropped (born after {a.max_born} or on the exclusion list)")
    pipe = FacePipeline(Config(quality_filter=False))
    rows, embs = [], []
    for k, r in enumerate(recs):
        img = cv2.imread(os.path.join(a.root, r["path"]))
        if img is None:
            continue
        img, fs = detect_all(pipe, img, pad_retry=True)
        fs = [f for f in fs if f["det"] >= 0.6 and min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= a.min_face]
        if not fs:
            continue
        area = sorted(((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in fs), reverse=True)
        if len(area) > 1 and area[0] < a.dominance * area[1]:
            continue   # group photo: cannot tell which face is the person
        f = max(fs, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        crop = norm_crop(img, f["kps"], 112)
        cp = os.path.join("crops", r["qid"], os.path.basename(r["path"]))
        os.makedirs(os.path.join(a.root, "crops", r["qid"]), exist_ok=True)
        cv2.imwrite(os.path.join(a.root, cp), crop)
        rows.append({"path": r["path"], "crop": cp, "identity": r["qid"], "name": r["name"], "gender": r["gender"], "born": r["born"],
                     "file": r["file"], "n_faces": len(fs), "license": r["license"], "author": r["author"], "source": r["page"]})
        embs.append(pipe._embed(crop)[0])
        if k % 100 == 0:
            print(k, "/", len(recs), len(rows), "faces", flush=True)
    df, E = pd.DataFrame(rows), np.stack(embs)

    def tokens(name):
        return [t for t in re.sub(r"[^a-z ]", " ", name.lower()).split() if len(t) >= 3]

    def is_anchor(row):
        fn = re.sub(r"[^a-z ]", " ", row["file"].lower().replace("_", " "))
        t = tokens(row["name"])
        return bool(t) and sum(x in fn for x in t) >= min(2, len(t))
    df["anchor"] = df.apply(is_anchor, axis=1)
    keep = np.zeros(len(df), bool)
    for _, idx in df.groupby("identity").indices.items():
        anc = idx[df.anchor.values[idx]]
        if len(anc) == 0:
            continue
        med = np.median(E[anc], axis=0); med /= np.linalg.norm(med) + 1e-9
        sim = E[idx] @ med
        keep[idx] = df.anchor.values[idx] | (sim >= a.admit)
    print(f"label cleaning: {len(df)} faces -> {int(keep.sum())} kept ({int(df.anchor.sum())} name-anchored)")
    df = df[keep].reset_index(drop=True)
    n = df.groupby("identity").size()
    df = df[df.identity.isin(n[n >= a.min_images].index)].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    split = {}
    for g, ids in df.groupby("gender").identity:
        ids = rng.permutation(ids.unique())
        for j, i in enumerate(ids):
            split[i] = "train" if j < 0.7 * len(ids) else ("val" if j < 0.85 * len(ids) else "test")
    df["split"] = df.identity.map(split)
    df.to_csv(os.path.join(a.root, "manifest.csv"), index=False)
    print(df.groupby("split").agg(images=("path", "size"), people=("identity", "nunique")))
    print(df.drop_duplicates("identity").gender.value_counts().to_dict(), "| licences:", df.license.value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
