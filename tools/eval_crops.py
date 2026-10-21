"""Baseline vs fine-tuned recognizer on the aligned 112x112 crops of the held-out (test) people.

  python tools/eval_crops.py --manifest data/ft_manifest.csv --root data --rec models/w600k_r50.onnx --name baseline --out runs/eval
  python tools/eval_crops.py ... --rec runs/ft1/finetuned.onnx --name finetuned --out runs/eval

Embeds each crop (image + mirror, summed, like the pipeline), builds same/different-person pairs INSIDE each
group of the test split, and scores them. The threshold is chosen on the val split at FAR 1% (never on test) and
applied to test. Groups: halfface (all / masked / unmasked), imfdb, pk_wiki (reported only if it has >=2 people).
This skips detection and alignment (crops are already aligned), so it measures the recognizer, not the pipeline;
the full-pipeline evaluation stays evaluate.py on the original photos.
"""
import argparse, json, os, sys
import cv2, numpy as np, pandas as pd, onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evalkit import make_pairs, eer_and_thr, thr_at_far, tar_at_far, metrics_at  # noqa: E402


def load_embedder(path):
    """GPU through onnx2torch when torch+CUDA exist (same weights, verified to 1e-6 by finetune.py); else ONNX Runtime CPU."""
    try:
        import torch, onnx, onnx2torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        m = onnx2torch.convert(onnx.load(path)).eval().cuda()
        def run(x):
            with torch.no_grad():
                return m(torch.from_numpy(np.ascontiguousarray(x)).cuda()).float().cpu().numpy()
        return run, "torch-cuda"
    except Exception as e:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return (lambda x: sess.run(None, {name: np.ascontiguousarray(x)})[0]), f"onnxruntime-cpu ({e})"


def embed(run, paths, root, bs=128):
    out = []
    for k in range(0, len(paths), bs):
        imgs = [cv2.imdecode(np.fromfile(os.path.join(root, p), np.uint8), cv2.IMREAD_COLOR) for p in paths[k:k + bs]]
        x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = run(x) + run(x[:, :, :, ::-1])
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out)


def scores(emb, ids, max_pos=20000, seed=0):
    I, J, y = make_pairs(ids, max_pos=max_pos, seed=seed)
    return y, np.einsum("ij,ij->i", emb[I], emb[J])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--rec", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, backend = load_embedder(a.rec)
    print(a.name, "backend:", backend, flush=True)
    df = pd.read_csv(a.manifest)
    if "mask" not in df:
        df["mask"] = "unmasked"
    df = df[df.split.isin(["val", "test"])].reset_index(drop=True)
    emb = embed(run, list(df.crop), a.root)

    va = (df.split == "val").values
    y, s = scores(emb[va], df.identity.values[va])
    thr = thr_at_far(y, s, 0.01)
    res = {"model": a.name, "val_threshold_far1pct": float(thr), "val_pairs": int(len(y)), "groups": {}}

    te = df[df.split == "test"]
    groups = {"all_test": te.index, "halfface": te[te.source == "halfface"].index,
              "halfface_masked": te[(te.source == "halfface") & (te["mask"] == "masked")].index,
              "halfface_unmasked": te[(te.source == "halfface") & (te["mask"] == "unmasked")].index,
              "imfdb": te[te.source == "imfdb"].index, "pk_wiki": te[te.source == "pk_wiki"].index}
    for g, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) == 0 or len(set(df.identity.values[idx])) < 2:
            continue   # impostor pairs need >= 2 people
        y, s = scores(emb[idx], df.identity.values[idx])
        eer, _ = eer_and_thr(y, s)
        m = metrics_at(y, s, thr)
        res["groups"][g] = {"people": int(len(set(df.identity.values[idx]))), "images": int(len(idx)), "pairs": int(len(y)),
                            "EER": eer, "TAR@FAR1%": tar_at_far(y, s, 0.01), "acc@val_thr": m["acc"],
                            "FAR@val_thr": m["FAR"], "FRR@val_thr": m["FRR"]}
        print(g, json.dumps(res["groups"][g]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, a.name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
