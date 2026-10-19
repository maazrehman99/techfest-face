"""Verification metrics: ROC/AUC, EER, TAR@FAR, accuracy/precision/recall/F1/FAR/FRR at a threshold,
plus identity-disjoint pair generation and per-group slicing."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def make_pairs(ids, groups=None, max_pos=20000, neg_per_pos=1, seed=0):
    """ids: identity label per item. If groups is given, impostor pairs are drawn within the same group
    (hardest, fairness-style protocol). Returns (i, j, y) arrays."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    by_id = {}
    for k, v in enumerate(ids):
        by_id.setdefault(v, []).append(k)
    pos = [(a[x], a[y]) for a in by_id.values() for x in range(len(a)) for y in range(x + 1, len(a))]
    if len(pos) > max_pos:
        pos = [pos[k] for k in rng.choice(len(pos), max_pos, replace=False)]
    pos = np.array(pos).reshape(-1, 2)
    n = len(ids)
    neg = []
    g = None if groups is None else np.asarray(groups)
    tries = 0
    target = len(pos) * neg_per_pos
    while len(neg) < target and tries < target * 50:
        i, j = rng.integers(0, n, 2)
        tries += 1
        if ids[i] == ids[j] or (g is not None and g[i] != g[j]):
            continue
        neg.append((i, j))
    neg = np.array(neg).reshape(-1, 2)
    I = np.concatenate([pos[:, 0], neg[:, 0]])
    J = np.concatenate([pos[:, 1], neg[:, 1]])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    return I, J, y


def scores_for(emb, I, J):
    return np.einsum("ij,ij->i", emb[I], emb[J])  # cosine; embeddings are L2-normalised


def eer_and_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[k] + fnr[k]) / 2), float(thr[k])


def thr_at_far(y, s, far):
    """Smallest-FRR threshold whose FAR on this (validation) set is <= far."""
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return float("nan")
    k = int(np.ceil((1 - far) * len(neg))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)]) + 1e-9


def tar_at_far(y, s, far):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(far, fpr, tpr))


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"thr": float(thr), "acc": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "FAR": fp / (fp + tn) if fp + tn else 0.0, "FRR": fn / (fn + tp) if fn + tp else 0.0,
            "pairs": int(len(y))}


def full_report(y, s, thr):
    """Everything for one score set, with `thr` fixed beforehand (from validation)."""
    eer, _ = eer_and_thr(y, s)
    r = metrics_at(y, s, thr)
    r.update({"AUC": float(roc_auc_score(y, s)), "EER": eer,
              "TAR@FAR=1e-1": tar_at_far(y, s, 1e-1), "TAR@FAR=1e-2": tar_at_far(y, s, 1e-2),
              "TAR@FAR=1e-3": tar_at_far(y, s, 1e-3)})
    return r
