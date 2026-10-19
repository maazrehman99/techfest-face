"""Dump sklearn's LFW subset (people with >=20 images) to jpgs with an identity-disjoint train/val/test manifest.
LFW is a Western-skewed sanity check for the harness and pipeline, NOT evidence for Pakistani performance."""
import os, numpy as np, pandas as pd, cv2
from sklearn.datasets import fetch_lfw_people

out = "data/lfw"
os.makedirs(f"{out}/img", exist_ok=True)
d = fetch_lfw_people(data_home="data/lfw_raw", min_faces_per_person=20, resize=1.0, color=True)
rng = np.random.default_rng(0)
ids = rng.permutation(len(d.target_names))
split = {int(i): ("train" if k < 30 else "val" if k < 46 else "test") for k, i in enumerate(ids)}
rows = []
for n, (im, t) in enumerate(zip(d.images, d.target)):
    p = f"img/{n:05d}.jpg"
    cv2.imwrite(f"{out}/{p}", np.clip(im * 255, 0, 255).astype(np.uint8)[:, :, ::-1])
    rows.append({"path": p, "identity": d.target_names[t], "split": split[int(t)]})
pd.DataFrame(rows).to_csv(f"{out}/manifest.csv", index=False)
print(pd.DataFrame(rows).groupby("split").agg(images=("path", "count"), ids=("identity", "nunique")))
