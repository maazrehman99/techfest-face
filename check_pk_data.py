"""Sanity checks on the processed Pakistani set: eyeball sheets + how much each augmentation hurts on REAL Pakistani faces.
  python check_pk_data.py --root data/pk_wiki --out results/pk_wiki
Writes contact_<split>.jpg (people x crops, to eyeball label noise), aug_real_samples.jpg (gender-aware augmentation on real
crops) and aug_real_eval.csv (per-augmentation mean same-person cosine, rank-1 and EER on val+test people)."""
import argparse, os, sys, warnings
import cv2, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A
from pipeline import Config, FacePipeline
from sklearn.metrics import roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data/pk_wiki"); ap.add_argument("--out", default="results/pk_wiki")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.root, "manifest.csv"))
    img = lambda p: cv2.imread(os.path.join(a.root, p))
    for sp in ("train", "val", "test"):
        d = df[df.split == sp]; people = list(d.groupby("identity"))[:30]
        if not people:
            continue
        fig, ax = plt.subplots(len(people), 5, figsize=(7.5, 1.55 * len(people)))
        ax = np.atleast_2d(ax)
        for i, (_, g) in enumerate(people):
            for j in range(5):
                ax[i, j].axis("off")
                if j < len(g):
                    ax[i, j].imshow(cv2.cvtColor(img(g.crop.iloc[j]), cv2.COLOR_BGR2RGB))
            ax[i, 0].set_title(f"{g.name.iloc[0]} ({g.gender.iloc[0]}, {len(g)} imgs)", fontsize=7, loc="left")
        plt.tight_layout(); plt.savefig(os.path.join(a.out, f"contact_{sp}.jpg"), dpi=60); plt.close()
    # augmentation samples on real crops
    rng = np.random.default_rng(3); pick = df.drop_duplicates("identity").sample(min(8, df.identity.nunique()), random_state=1)
    fig, ax = plt.subplots(len(pick), 6, figsize=(9, 1.6 * len(pick))); ax = np.atleast_2d(ax)
    for i, (_, r) in enumerate(pick.iterrows()):
        c = img(r.crop)
        for j in range(6):
            x, tags = (c, ["original"]) if j == 0 else A.random_augment(c, rng, p_occlusion=.9, gender=r.gender)
            ax[i, j].imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)); ax[i, j].axis("off"); ax[i, j].set_title(",".join(tags)[:28], fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "aug_real_samples.jpg"), dpi=70); plt.close()
    # per-augmentation effect on val+test people with >= 2 images
    pipe = FacePipeline(Config(quality_filter=False)); emb = lambda c: pipe._embed(c)[0]
    d = df[df.split.isin(["val", "test"])]; P = []
    for _, g in d.groupby("identity"):
        if len(g) >= 2:
            P.append((img(g.crop.iloc[0]), img(g.crop.iloc[1])))
    print(len(P), "people for the augmentation check")
    if len(P) < 5:
        return
    gal = np.stack([emb(b) for _, b in P]); rows = []
    for name in ["clean"] + list(A.OCCLUSIONS) + list(A.ACCESSORIES) + list(A.DEGRADATIONS):
        pr = np.stack([emb(x if name == "clean" else A.apply(x, name, rng)) for x, _ in P])
        S = pr @ gal.T; n = len(P); y = np.eye(n).ravel()
        fpr, tpr, _ = roc_curve(y, S.ravel())
        rows.append((name, np.diag(S).mean(), float(np.mean(S.argmax(1) == np.arange(n))), float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])))
    out = pd.DataFrame(rows, columns=["aug", "mean_genuine_cos", "rank1", "EER"]).round(3)
    out.to_csv(os.path.join(a.out, "aug_real_eval.csv"), index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
