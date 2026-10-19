"""halfFace (Mendeley pk44mkx9vm, CC BY 4.0, 166 South Asian participants) -> eval-only manifest.
Filename: pXXX_<mask 0/1><pose 0 front/1 left/2 right><gender m/f>. Split by participant: 40 val ids, rest test (no train;
this set is held out from any fine-tuning). Odd filenames are skipped and counted. Gender is ~7 women, so gender slices are thin."""
import os, re, numpy as np, pandas as pd
root = "data/halfface"
D = [os.path.join(dp) for dp, dn, fn in os.walk(root + "/extracted") if os.path.basename(dp) == "Dataset1_Full Facial Image"][0]
pat = re.compile(r"^(p\d+)_([01])([012])([mf])\.(jpe?g|png)$", re.I)
rows, skipped = [], 0
for p in sorted(os.listdir(D)):
    for f in sorted(os.listdir(os.path.join(D, p))):
        m = pat.match(f)
        if not m:
            skipped += 1; continue
        rows.append({"path": os.path.relpath(os.path.join(D, p, f), root), "identity": m[1].lower(),
                     "mask": "masked" if m[2] == "1" else "unmasked", "pose": ["front", "left", "right"][int(m[3])],
                     "gender": "male" if m[4].lower() == "m" else "female"})
df = pd.DataFrame(rows)
ids = np.random.default_rng(0).permutation(df.identity.unique())
val = set(ids[:40])
df["split"] = np.where(df.identity.isin(val), "val", "test")
df.to_csv(f"{root}/manifest.csv", index=False)
print("skipped", skipped); print(df.groupby("split").agg(images=("path", "count"), ids=("identity", "nunique")))
print(df.drop_duplicates("identity").gender.value_counts().to_dict(), df[df.split == "test"].gender.value_counts().to_dict())
