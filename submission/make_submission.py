"""Fill in embed(), then run:
    python make_submission.py <group> <data_dir>
e.g. practice:  python make_submission.py group05 practice
     lab day:   python make_submission.py group05 lab_pack     -> upload group05.npz
<data_dir> must contain train/ and test/."""
import os, sys, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import cv2
from pipeline import FacePipeline, _unit, PROVIDERS
from insightface.utils.face_align import norm_crop
from insightface.model_zoo import get_model

_P = {}
USE_R100 = os.environ.get("SUB_R100", "1") == "1"      # average ArcFace R50 + R100 (better on IMFDB: 75.5 -> 78.1%)
MULTI = os.environ.get("SUB_MULTI", "1") == "1"        # test images with several faces: pick the face closest to a train face

def _models():
    if not _P:
        _P["pipe"] = FacePipeline()
        if USE_R100:
            m = get_model(os.path.join(HERE, "..", "models", "glintr100.onnx"), providers=PROVIDERS)       # GPU when FACE_PROVIDER=cuda
            m.prepare(ctx_id=0 if PROVIDERS[0].startswith("CUDA") else -1); _P["r100"] = m
    return _P

def _vec(crop):
    P = _models()
    e = P["pipe"].rec.get_feat([crop, crop[:, ::-1].copy()]); v = _unit(e[0] + e[1])
    if USE_R100:
        e = P["r100"].get_feat([crop, crop[:, ::-1].copy()]); v = np.concatenate([v, _unit(e[0] + e[1])]) / np.sqrt(2)
    return v.astype(np.float32)

def faces(img: Image.Image, multi=False):
    """All faces of an image as [(area, embedding)], largest first. No face -> one centre-crop guess."""
    pipe = _models()["pipe"]
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if max(bgr.shape[:2]) < 300: bgr = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if max(bgr.shape[:2]) < 700:
        p = int(0.35 * max(bgr.shape[:2])); bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REFLECT)
    bgr, fs = pipe._detect(bgr)
    if not fs:
        h, w = bgr.shape[:2]; s = min(h, w)
        return [(0.0, _vec(cv2.resize(bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (112, 112))))]
    fs = sorted(((float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), f) for f in fs), key=lambda t: -t[0])
    top = fs[0][0]
    keep = [t for t in fs if t[0] >= 0.15 * top][: (5 if multi else 1)]     # tiny background faces are never used
    return [(a, _vec(norm_crop(bgr, f.kps, 112))) for a, f in keep]

def embed(img: Image.Image) -> np.ndarray:
    """Largest face, aligned, ArcFace (R50 + R100 averaged), unit length."""
    return faces(img)[0][1]

def run(folder, ref=None):
    ids = sorted(os.listdir(folder)); embs = []
    for k, i in enumerate(ids):
        if k % 10 == 0: print(f"{folder}: {k}/{len(ids)}", flush=True)
        fs = faces(Image.open(f"{folder}/{i}").convert("RGB"), multi=ref is not None and MULTI)
        e = fs[0][1]
        if len(fs) > 1:
            e = max(fs, key=lambda f: (ref @ f[1]).max())[1]
        embs.append(np.asarray(e, dtype=np.float32))
    return np.array(ids), np.stack(embs)

if __name__ == "__main__":
    group, data = sys.argv[1], sys.argv[2]
    tr_ids, tr = run(f"{data}/train"); te_ids, te = run(f"{data}/test", ref=tr)
    np.savez(f"{group}.npz", train_ids=tr_ids, train_emb=tr, test_ids=te_ids, test_emb=te)
    print("saved", f"{group}.npz", tr.shape, te.shape)
