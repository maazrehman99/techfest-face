"""Build data/pk/manifest.csv from data/pk/raw/<ID>/<ID>_<tag>_<n>.jpg and data/pk/people.csv.
Splits by PERSON (identity-disjoint), stratified roughly by gender so test has both. Condition tag comes from the filename."""
import os, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "data/pk"
people = pd.read_csv(f"{root}/people.csv").set_index("person_id")
rng = np.random.default_rng(42)
split = {}
for _, grp in people.groupby("gender") if "gender" in people else [(None, people)]:
    ids = rng.permutation(grp.index.values)
    n = len(ids)
    for k, i in enumerate(ids):
        split[i] = "train" if k < 0.60 * n else "val" if k < 0.75 * n else "test"
rows = []
for pid in sorted(os.listdir(f"{root}/raw")):
    if pid not in people.index:
        print("WARN: no people.csv row for", pid); continue
    for f in sorted(os.listdir(f"{root}/raw/{pid}")):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = f.rsplit(".", 1)[0].split("_")
        rows.append({"path": f"raw/{pid}/{f}", "identity": pid, "split": split[pid],
                     "condition": parts[1] if len(parts) > 2 else "std", **people.loc[pid].to_dict()})
df = pd.DataFrame(rows)
df.to_csv(f"{root}/manifest.csv", index=False)
print(df.groupby("split").agg(images=("path", "count"), people=("identity", "nunique")))
for c in ("gender", "age_bracket", "region", "condition"):
    if c in df:
        print(df.drop_duplicates("identity").groupby(c).size().to_dict() if c != "condition" else df[c].value_counts().to_dict())
