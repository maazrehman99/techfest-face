"""Add IMFDB (Indian Movie Face Database, 100 actors) to the fine-tune data. Run build_ft_data.py first.

  python build_imfdb.py                 # needs data/imfdb/extracted/IMFDB_final (official zip, unpacked)
  python build_imfdb.py --per-actor 40

Why: halfFace + Wikimedia give only ~600 training images. IMFDB has 100 identities x 200-620 images (about 32% women),
so a capped sample of it is the largest South Asian identity set we can get quickly. It is Indian film actors, a South
Asian PROXY, not Pakistani data. The site states no licence: cite it and keep it local (ATTRIBUTION_AND_USE.md).

Split: per actor, stratified by gender, 60% train / 15% val / 25% test (seed 0), same rule as halfFace. Identities are
prefixed "imfdb_" so they never collide with halfFace ids (p001...).
Writes
  data/imfdb/crops/<actor>/<file>.jpg   aligned 112x112 crops of the sampled images
  data/imfdb/manifest_eval.csv          path (original image, relative to data/imfdb), identity, mask, pose, gender, split
                                        -> evaluate.py --root data/imfdb (held-out actors = an extra South Asian test)
  data/ft_manifest.csv                  imfdb rows appended (existing imfdb rows replaced, so the script can be re-run)
"""
import argparse, glob, os
from collections import Counter
import cv2, numpy as np, pandas as pd
from insightface.utils.face_align import norm_crop

from build_ft_data import DATA, imread, person_split

ROOT = os.path.join(DATA, "imfdb")
SRC = os.path.join(ROOT, "extracted", "IMFDB_final")


def actor_gender(actor):
    """Majority of the MALE/FEMALE column over the actor's annotation files (a few rows are shifted or blank)."""
    c = Counter()
    for t in glob.glob(os.path.join(SRC, actor, "*", "*.txt")):
        for line in open(t, encoding="utf-8", errors="ignore"):
            f = line.rstrip("\n").split("\t")
            if len(f) > 10 and f[10] in ("MALE", "FEMALE"):
                c[f[10].lower()] += 1
    return c.most_common(1)[0][0] if c else "unknown"


def aligned(pipe, bgr, min_det):
    """Tiny movie-frame face crops give SCRFD no context: upscale 2x and pad, like FacePipeline.analyze does."""
    bgr = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    p = int(0.35 * max(bgr.shape[:2]))
    bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REFLECT)
    bgr, faces = pipe._detect(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return norm_crop(bgr, f.kps, 112) if f.det_score >= min_det else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-actor", type=int, default=40, help="max images kept per actor (balances identities)")
    ap.add_argument("--min-det", type=float, default=0.5, help="drop faces the detector is unsure about")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ft_path = os.path.join(DATA, "ft_manifest.csv")
    if not os.path.exists(ft_path):
        raise SystemExit("run build_ft_data.py first (it writes data/ft_manifest.csv)")
    if not os.path.isdir(SRC):
        raise SystemExit(f"{SRC} not found: unpack the official IMFDB_final.zip into data/imfdb/extracted")

    actors = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    people = pd.DataFrame({"identity": actors, "gender": [actor_gender(x) for x in actors]})
    people = people[people.gender != "unknown"]
    people["split"] = person_split(people, a.seed)
    print("IMFDB people:", people.groupby(["split", "gender"]).size().to_dict())

    from pipeline import Config, FacePipeline
    pipe = FacePipeline(Config(quality_filter=False))
    rng = np.random.default_rng(a.seed)
    crops, evals, nodet = [], [], 0
    for k, r in enumerate(people.itertuples()):
        files = sorted(glob.glob(os.path.join(SRC, r.identity, "*", "images", "*.jp*g")))
        kept = 0
        for f in rng.permutation(files):
            if kept >= a.per_actor:
                break
            img = imread(f)
            crop = aligned(pipe, img, a.min_det) if img is not None else None
            if crop is None:
                nodet += 1; continue
            name = os.path.splitext(os.path.basename(f))[0] + "_" + os.path.basename(os.path.dirname(os.path.dirname(f))) + ".jpg"
            cp = os.path.join("imfdb", "crops", r.identity, name)
            os.makedirs(os.path.dirname(os.path.join(DATA, cp)), exist_ok=True)
            cv2.imencode(".jpg", crop)[1].tofile(os.path.join(DATA, cp))
            ident = "imfdb_" + r.identity
            crops.append({"crop": cp.replace("\\", "/"), "identity": ident, "split": r.split, "gender": r.gender,
                          "source": "imfdb", "mask": "unmasked"})
            evals.append({"path": os.path.relpath(f, ROOT).replace("\\", "/"), "identity": ident, "mask": "unmasked",
                          "pose": "front", "gender": r.gender, "split": r.split})
            kept += 1
        print(f"{k + 1}/{len(people)} {r.identity}: {kept} kept", flush=True)
    print(f"IMFDB: {len(crops)} crops kept, {nodet} sampled images skipped (no confident face)")

    new = pd.DataFrame(crops)
    pd.DataFrame(evals).to_csv(os.path.join(ROOT, "manifest_eval.csv"), index=False)
    ft = pd.read_csv(ft_path)
    ft = pd.concat([ft[ft.source != "imfdb"], new], ignore_index=True)
    s = {k: set(g.identity) for k, g in ft.groupby("split")}
    assert not (s["train"] & s.get("val", set()) or s["train"] & s.get("test", set())), "identity leak"
    ft.to_csv(ft_path, index=False)
    print("fine-tune manifest data/ft_manifest.csv:")
    print(ft.groupby(["split", "source"]).agg(images=("crop", "size"), people=("identity", "nunique")))


if __name__ == "__main__":
    main()
