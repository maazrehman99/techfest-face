"""Fill in embed(), then run:
    python make_submission.py <group> <data_dir>
e.g. practice:  python make_submission.py group05 practice
     lab day:   python make_submission.py group05 lab_pack     -> upload group05.npz
<data_dir> must contain train/ and test/."""
import os, sys
os.environ.setdefault("FACE_THREADS", "2")   # several processes x 2 threads is much faster than one process x all threads
import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import cv2
from pipeline import FacePipeline, _unit, PROVIDERS
from insightface.utils.face_align import norm_crop
from insightface.model_zoo import get_model

_P = {}
USE_R100 = os.environ.get("SUB_R100", "1") == "1"      # average ArcFace R50 + R100 (better on IMFDB: 75.5 -> 78.1%)
MULTI = os.environ.get("SUB_MULTI", "1") == "1"        # test images with several faces: pick the face closest to a train face
CENTER = os.environ.get("SUB_CENTER", "1") == "1"      # subtract the pack's mean embedding (IMFDB: +0.4 val, +1.4 test top-1)
CENTER_MIN = int(os.environ.get("SUB_CENTER_MIN", "40"))   # ... only for packs this big (the mean is noisy for tiny packs)
WORKERS = int(os.environ.get("SUB_WORKERS", str(max(1, min(4, (os.cpu_count() or 2) // 2)))))

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

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

def _detect_hard(pipe, bgr):
    """Only used when the normal detection found nothing: lower cut-off (0.5 -> 0.3), bigger detector size, rotated image.
    -> (image the faces refer to, faces)"""
    det = pipe.app.det_model; old = det.det_thresh; det.det_thresh = 0.3
    try:
        for rot in (0, 90, 270, 180):
            img = bgr if rot == 0 else cv2.rotate(bgr, _ROT[rot])
            for sz in (640, 1024):
                det.input_size = (sz, sz)
                fs = pipe.app.get(img)
                if fs: return img, fs
    finally:
        det.det_thresh = old
    return bgr, []

def faces(img: Image.Image, multi=False):
    """All faces of an image as ([(area, embedding)], status), largest first.
    status: ok | retry (found only by the hard retry) | fallback (no face anywhere: centre-crop guess)."""
    pipe = _models()["pipe"]
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if max(bgr.shape[:2]) < 300: bgr = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if max(bgr.shape[:2]) < 700:
        p = int(0.35 * max(bgr.shape[:2])); bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REFLECT)
    bgr, fs = pipe._detect(bgr)
    status = "ok"
    if not fs:
        bgr, fs = _detect_hard(pipe, bgr); status = "retry"
    if not fs:
        h, w = bgr.shape[:2]; s = min(h, w)
        return [(0.0, _vec(cv2.resize(bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (112, 112))))], "fallback"
    fs = sorted(((float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), f) for f in fs), key=lambda t: -t[0])
    top = fs[0][0]
    keep = [t for t in fs if t[0] >= 0.15 * top][: (5 if multi else 1)]     # tiny background faces are never used
    return [(a, _vec(norm_crop(bgr, f.kps, 112))) for a, f in keep], status

def embed(img: Image.Image) -> np.ndarray:
    """Largest face, aligned, ArcFace (R50 + R100 averaged), unit length."""
    return faces(img)[0][0][1]

def _load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")     # phones store rotation in EXIF; PIL ignores it by default

def _job(args):
    path, multi = args
    return faces(_load(path), multi=multi)

def _map(paths, multi):
    jobs = [(p, multi) for p in paths]
    if WORKERS <= 1:
        return [_job(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(WORKERS, initializer=_models) as pool:
        out = []
        for k, r in enumerate(pool.imap(_job, jobs, chunksize=1)):
            out.append(r)
            if k % 10 == 0: print(f"  {k}/{len(jobs)}", flush=True)
        return out

def run(folder, ref=None):
    ids = sorted(os.listdir(folder)); print(f"{folder}: {len(ids)} images, {WORKERS} workers", flush=True)
    res = _map([f"{folder}/{i}" for i in ids], multi=ref is not None and MULTI)
    embs = []
    for i, (fs, status) in zip(ids, res):
        if status != "ok": print(f"  NOTE {i}: {status}", flush=True)
        e = fs[0][1]
        if len(fs) > 1:
            e = max(fs, key=lambda f: (ref @ f[1]).max())[1]
        embs.append(np.asarray(e, dtype=np.float32))
    return np.array(ids), np.stack(embs)

if __name__ == "__main__":
    group, data = sys.argv[1], sys.argv[2]
    tr_ids, tr = run(f"{data}/train"); te_ids, te = run(f"{data}/test", ref=tr)
    if CENTER and len(tr) + len(te) >= CENTER_MIN:
        m = np.vstack([tr, te]).mean(0)       # both are unit length already
        tr, te = [(x - m) / np.linalg.norm(x - m, axis=1, keepdims=True) for x in (tr, te)]
        print("centered embeddings on the pack mean")
    np.savez(f"{group}.npz", train_ids=tr_ids, train_emb=tr, test_ids=te_ids, test_emb=te)
    print("saved", f"{group}.npz", tr.shape, te.shape)
