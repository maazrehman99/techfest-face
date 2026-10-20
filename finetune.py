"""Fine-tune the ArcFace R50 (w600k_r50) on identity-labelled face crops, keep it only if it beats the baseline.

  python finetune.py --manifest data/pk_wiki/manifest.csv --root data/pk_wiki --out runs/ft1

Manifest columns: crop (aligned 112x112 image path, relative to --root), identity, split (train|val|test).
Identities never cross splits. Only `train` is trained on; `val` picks the best epoch; `test` is untouched here
(evaluate it afterwards with evaluate.py). halfFace must NEVER be in this manifest (it is the held-out benchmark).

Recipe (conservative, the dataset is small):
 - start from the pretrained network converted ONNX -> PyTorch (onnx2torch)
 - BatchNorm layers frozen (running statistics kept), only the last --trainable fraction of parameters is trained
 - ArcFace margin head whose class weights are initialised from each class's mean pretrained embedding
 - augmentation from augment.py (coverings, accessories, blur/low-res/lighting) + horizontal flip
 - mixed precision on CUDA; best epoch chosen by validation EER; if no epoch beats epoch 0 (the baseline) the
   baseline is exported unchanged and the log says so.
Outputs in --out: finetuned.onnx (drop-in for config.json "rec_onnx"), history.csv, log.txt.
"""
import argparse, os, sys, time
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import augment as A  # noqa: E402
from evalkit import make_pairs  # noqa: E402


def to_tensor(bgr):
    """112x112 BGR uint8 -> normalised RGB CHW float (same as insightface: (x-127.5)/127.5)."""
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(((x - 127.5) / 127.5).transpose(2, 0, 1).copy())


class Crops(torch.utils.data.Dataset):
    def __init__(self, df, root, label_map, augment, seed=0):
        self.df, self.root, self.map, self.aug = df.reset_index(drop=True), root, label_map, augment
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.root, r.crop))
        if img is None or img.shape[:2] != (112, 112):
            img = cv2.resize(img if img is not None else np.zeros((112, 112, 3), np.uint8), (112, 112))
        if self.aug:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            img, _ = A.random_augment(img, rng, gender=r.get("gender"))
        return to_tensor(img), self.map.get(r.identity, -1)


class ArcHead(nn.Module):
    def __init__(self, init_w, s=32.0, m=0.30):
        super().__init__()
        self.w, self.s, self.m = nn.Parameter(init_w.clone()), s, m

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.w)).clamp(-1 + 1e-6, 1 - 1e-6)
        th = torch.acos(cos.gather(1, y[:, None]))
        tgt = torch.cos(th + self.m)
        logits = cos.scatter(1, y[:, None], tgt)
        return F.cross_entropy(self.s * logits, y)


def freeze(model, trainable):
    """Freeze BatchNorm (eval mode, no grads) and every parameter except the last `trainable` fraction."""
    params = [(n, p) for n, p in model.named_parameters()]
    cut = int(len(params) * (1 - trainable))
    for k, (_, p) in enumerate(params):
        p.requires_grad = k >= cut
    return sum(p.numel() for _, p in params if p.requires_grad), sum(p.numel() for _, p in params)


def set_train(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()   # keep pretrained statistics; tiny batches would wreck them


@torch.no_grad()
def embed_all(model, df, root, dev, bs=128):
    model.eval()
    ds = Crops(df, root, {}, False)
    out = []
    for i in range(0, len(ds), bs):
        x = torch.stack([ds[j][0] for j in range(i, min(i + bs, len(ds)))]).to(dev)
        out.append(F.normalize(model(x).float()).cpu())
    return torch.cat(out).numpy()


def val_eer(emb, ids):
    I, J, y = make_pairs(ids, max_pos=20000)
    s = np.einsum("ij,ij->i", emb[I], emb[J])
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "w600k_r50.onnx"))
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--trainable", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=0, help="debug: cap number of training images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8")

    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    df = pd.read_csv(a.manifest)
    assert not any("halfface" in str(p).lower() for p in df.get("path", df.crop)), "halfFace must not be in the training manifest"
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if a.max_train:
        tr = tr.sample(min(a.max_train, len(tr)), random_state=0)
    assert not set(tr.identity) & set(va.identity), "identity leak between train and val"
    P(f"train {len(tr)} imgs / {tr.identity.nunique()} people; val {len(va)} imgs / {va.identity.nunique()} people")

    from onnx2torch import convert
    model = convert(a.onnx).to(dev)
    n_tr, n_all = freeze(model, a.trainable)
    P(f"trainable params {n_tr:,} of {n_all:,}")

    # head init from pretrained class-mean embeddings
    classes = sorted(tr.identity.unique()); cmap = {c: i for i, c in enumerate(classes)}
    e_tr = embed_all(model, tr, a.root, dev)
    init = torch.from_numpy(np.stack([F.normalize(torch.from_numpy(e_tr[(tr.identity == c).values].mean(0)), dim=0).numpy() for c in classes])).to(dev)
    head = ArcHead(init).to(dev)
    base = val_eer(embed_all(model, va, a.root, dev), va.identity.values)
    P(f"epoch 0 (baseline) val EER {base * 100:.2f}%")

    dl = torch.utils.data.DataLoader(Crops(tr, a.root, cmap, True), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                     drop_last=len(tr) > a.bs, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": head.parameters(), "lr": a.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(1, len(dl)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, best_state, bad, hist = base, None, 0, [(0, float("nan"), base)]
    for ep in range(1, a.epochs + 1):
        set_train(model); t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = head(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        eer = val_eer(embed_all(model, va, a.root, dev), va.identity.values)
        hist.append((ep, tot / n, eer))
        P(f"epoch {ep}: loss {tot / n:.3f}  val EER {eer * 100:.2f}%  ({time.time() - t0:.0f}s)" + ("  *best*" if eer < best else ""))
        if eer < best - 1e-4:
            best, bad = eer, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                P("early stopping"); break
    pd.DataFrame(hist, columns=["epoch", "loss", "val_EER"]).to_csv(os.path.join(a.out, "history.csv"), index=False)

    if best_state is None:
        P(f"NO epoch beat the baseline val EER ({base * 100:.2f}%): exporting the unchanged baseline. Report this honestly.")
        model = convert(a.onnx).to(dev)
    else:
        model.load_state_dict(best_state)
        P(f"best val EER {best * 100:.2f}% vs baseline {base * 100:.2f}%")
    model.eval()
    out = os.path.join(a.out, "finetuned.onnx")
    kw = dict(input_names=["input.1"], output_names=["output"], dynamic_axes={"input.1": {0: "batch"}, "output": {0: "batch"}},
              opset_version=17)
    try:   # classic exporter: the new default (dynamo) needs the extra `onnxscript` package
        torch.onnx.export(model.cpu(), torch.randn(1, 3, 112, 112), out, dynamo=False, **kw)
    except TypeError:   # older PyTorch has no `dynamo` argument and always uses the classic exporter
        torch.onnx.export(model.cpu(), torch.randn(1, 3, 112, 112), out, **kw)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, 112, 112)
    d = np.abs(s.run(None, {s.get_inputs()[0].name: x.numpy()})[0] - model(x).detach().numpy()).max()
    P(f"exported {out}; max |onnx - torch| = {d:.2e}")
    P("next: set config.json rec_onnx to this file (path relative to the project), then run evaluate.py on halfFace and the held-out test split")


if __name__ == "__main__":
    main()
