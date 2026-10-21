"""Decision B (2026-10-21): build the fine-tuning data from halfFace (split BY PERSON) + the Pakistani Wikimedia crops.

  python build_ft_data.py                 # needs data/halfface/extracted/... (the Mendeley zip, unpacked)
  python build_ft_data.py --skip-crops    # only write the person split (no images needed)

Why: the Wikimedia set alone is 13 people / 48 images, too small to fine-tune on. halfFace has 166 South Asian people
(real beards, masks), so some of its people are moved into training. Its other people stay held out, and the
baseline MUST be re-measured on them (results/halfface_ft_baseline), never compared with the old 126-person test numbers.

Writes
  data/halfface/manifest_ft.csv  path, identity, mask, pose, gender, split   (for evaluate.py: baseline vs fine-tuned)
  data/halfface/crops/<id>/<file>.jpg   aligned 112x112 crops
  data/ft_manifest.csv           crop (relative to data/), identity, split, gender, source   (for finetune.py --root data)
Split: per person, stratified by gender, 60% train / 15% val / 25% test (seed 0). Wikimedia people keep their own split.
"""
import argparse, os
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def person_split(df, seed=0, train=0.60, val=0.15):
    rng = np.random.default_rng(seed)
    split = {}
    people = df.drop_duplicates("identity")
    for _, ids in people.groupby("gender").identity:
        ids = rng.permutation(ids.values)
        for j, i in enumerate(ids):
            split[i] = "train" if j < train * len(ids) else ("val" if j < (train + val) * len(ids) else "test")
    return df.identity.map(split)


def imread(path):
    """cv2.imread fails on non-ASCII Windows paths; decoding the bytes works everywhere."""
    buf = np.fromfile(path, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crops", action="store_true", help="only write manifest_ft.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    hf = pd.read_csv(os.path.join(DATA, "halfface", "manifest.csv"))
    hf["split"] = person_split(hf, a.seed)
    hf.to_csv(os.path.join(DATA, "halfface", "manifest_ft.csv"), index=False)
    print("halfFace person split:")
    print(hf.groupby("split").agg(images=("path", "size"), people=("identity", "nunique"),
                                  women=("gender", lambda g: int((g == "female").sum()))))
    if a.skip_crops:
        return

    from pipeline import Config, FacePipeline
    pipe = FacePipeline(Config(quality_filter=False))
    rows, missing, noface = [], 0, 0
    for k, r in enumerate(hf.itertuples()):
        img = imread(os.path.join(DATA, "halfface", r.path))
        if img is None:
            missing += 1; continue
        crop = pipe.align_crop(img)
        if crop is None:
            noface += 1; continue
        cp = os.path.join("halfface", "crops", r.identity, os.path.splitext(os.path.basename(r.path))[0] + ".jpg")
        os.makedirs(os.path.dirname(os.path.join(DATA, cp)), exist_ok=True)
        cv2.imencode(".jpg", crop)[1].tofile(os.path.join(DATA, cp))
        rows.append({"crop": cp.replace("\\", "/"), "identity": r.identity, "split": r.split, "gender": r.gender,
                     "source": "halfface", "mask": r.mask})
        if k % 100 == 0:
            print(k, "/", len(hf), flush=True)
    if missing == len(hf):
        raise SystemExit("no halfFace images found under data/halfface/extracted: download + unzip the Mendeley zip first")
    print(f"halfFace crops: {len(rows)} ok, {noface} no face, {missing} unreadable")

    pk = pd.read_csv(os.path.join(DATA, "pk_wiki", "manifest.csv"))
    pk = pd.DataFrame({"crop": "pk_wiki/" + pk.crop, "identity": pk.identity, "split": pk.split, "gender": pk.gender,
                       "source": "pk_wiki", "mask": "unmasked"})
    ft = pd.concat([pd.DataFrame(rows), pk], ignore_index=True)
    s = {k: set(g.identity) for k, g in ft.groupby("split")}
    assert not (s["train"] & s.get("val", set()) or s["train"] & s.get("test", set())), "identity leak"
    ft.to_csv(os.path.join(DATA, "ft_manifest.csv"), index=False)
    print("fine-tune manifest data/ft_manifest.csv:")
    print(ft.groupby(["split", "source"]).agg(images=("crop", "size"), people=("identity", "nunique")))


if __name__ == "__main__":
    main()
