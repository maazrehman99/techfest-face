"""Is the fine-tuned model really better than the baseline on the held-out crops, or is it luck?

  python tools/boot_ci.py --manifest data/ft_manifest.csv --root data --base models/buffalo_l/w600k_r50.onnx --ft runs/ft1/finetuned.onnx

Bootstrap over PEOPLE: redraw the test people with replacement B times (within each source), rebuild same/different
pairs, and compute the EER of both models on exactly the same pairs. Reports the mean EER difference (fine-tuned minus
baseline; negative = fine-tuned better), its 95% interval and the share of draws where the fine-tuned model wins.
Copies of one person drawn twice are never paired as impostors with each other.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_crops import load_embedder, embed  # noqa: E402
from evalkit import make_pairs, eer_and_thr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True); ap.add_argument("--ft", required=True)
    ap.add_argument("--B", type=int, default=200); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest); df = df[df.split == "test"].reset_index(drop=True)
    E = {}
    for name, path in (("base", a.base), ("ft", a.ft)):
        run, _ = load_embedder(path)
        E[name] = embed(run, list(df.crop), a.root)
    ids, src = df.identity.values, df.source.values
    people = {s: sorted(set(ids[src == s])) for s in sorted(set(src))}
    rows = {s: {p: np.where(ids == p)[0] for p in ps} for s, ps in people.items()}
    rng = np.random.default_rng(0)
    out = {}
    for group in ["all_test"] + [s for s in people if len(people[s]) >= 5]:
        srcs = list(people) if group == "all_test" else [group]
        diffs, b_eer, f_eer = [], [], []
        for _ in range(a.B):
            idx, lab, orig = [], [], []
            for s in srcs:
                draw = rng.choice(people[s], len(people[s]), replace=True)
                for k, p in enumerate(draw):
                    r = rows[s][p]; idx += list(r); lab += [f"{s}{k}"] * len(r); orig += [p] * len(r)
            idx, lab, orig = np.array(idx), np.array(lab), np.array(orig)
            I, J, y = make_pairs(lab, max_pos=4000, seed=int(rng.integers(1e9)))
            keep = (y == 1) | (orig[I] != orig[J])           # drop impostor pairs between two copies of one person
            I, J, y = I[keep], J[keep], y[keep]
            res = []
            for n in ("base", "ft"):
                e = E[n][idx]; s_ = np.einsum("ij,ij->i", e[I], e[J]); res.append(eer_and_thr(y, s_)[0])
            b_eer.append(res[0]); f_eer.append(res[1]); diffs.append(res[1] - res[0])
        d = np.array(diffs)
        out[group] = {"people": int(sum(len(people[s]) for s in srcs)), "B": a.B,
                      "EER_base": float(np.mean(b_eer)), "EER_ft": float(np.mean(f_eer)),
                      "diff_mean": float(d.mean()), "diff_95ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      "share_ft_better": float((d < 0).mean())}
        print(group, json.dumps(out[group]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
