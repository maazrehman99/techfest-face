"""Checks data/pk_test against report/PK_TEST_PROTOCOL.md before any model is run on it.
  python tools/check_pk_test.py [--dir data/pk_test]
Checks: 2 photos per person, every photo has a source row (url, licence, author), tags.csv covers every photo,
no overlap with the pk_wiki people, and the mix (women, coverings). Never runs a model."""
import argparse, glob, os, re
import pandas as pd

def norm(s): return re.sub(r"[^a-z]", "", str(s).lower())

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/pk_test"); d = ap.parse_args().dir
files = sorted(f for f in glob.glob(f"{d}/*/*") if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
rel = [os.path.relpath(f, d).replace(os.sep, "/") for f in files]
people = {}
for r in rel: people.setdefault(r.split("/")[0], []).append(r)
bad = []
for p, fs in people.items():
    if len(fs) != 2: bad.append(f"{p}: {len(fs)} photos (need exactly 2)")
src = pd.read_csv(f"{d}/sources.csv"); src = src[src.file != "Example_Person/1.jpg"]
have = set(src.file)
for r in rel:
    if r not in have: bad.append(f"{r}: no row in sources.csv")
for _, x in src.iterrows():
    for c in ("page_url", "licence", "author"):
        if pd.isna(x[c]) or not str(x[c]).strip(): bad.append(f"{x.file}: empty {c}")
tags = pd.read_csv(f"{d}/tags.csv"); tg = set(tags.file)
for r in rel:
    if r not in tg: bad.append(f"{r}: not in tags.csv")
wiki = pd.read_csv("data/pk_wiki/manifest.csv")
wn = {norm(n) for n in wiki["name"].dropna()}
for p in people:
    if norm(p.replace("_", " ")) in wn: bad.append(f"{p}: already in the pk_wiki set (overlap)")
g = src.drop_duplicates("person").gender.str.lower().value_counts().to_dict() if len(src) else {}
cov = tags[tags.file.isin(rel)].covering.value_counts().to_dict()
print(f"people {len(people)}, photos {len(rel)}, gender {g}, covering {cov}")
if len(people) < 40: bad.append(f"only {len(people)} people (target about 50)")
if g.get("female", 0) < 15: bad.append("fewer than 15 women (target in protocol)")
print("\n".join(bad) if bad else "OK: set is ready, freeze the list, then run eval_pk_test.py once")
raise SystemExit(1 if bad else 0)
