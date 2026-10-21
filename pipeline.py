"""Face recognition pipeline: detect -> quality check -> align -> embed.

Every stage can be toggled so the ablation in evaluate.py can measure its contribution.
Detector (SCRFD) and recognizer (ArcFace) come from InsightFace's buffalo_l pack.

analyze() does the expensive work once per image (detect, both crops, all embedding variants);
decide() applies a Config to that result. process() = decide(analyze()) for deployment.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis

_THREADS = int(os.environ.get("FACE_THREADS", "0"))
if _THREADS:  # cap ONNX threads per process so several workers don't oversubscribe the CPU
    _orig_session = ort.InferenceSession

    def _capped(path, sess_options=None, providers=None, provider_options=None, **kw):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = _THREADS, 1
        return _orig_session(path, sess_options=so, providers=providers, provider_options=provider_options, **kw)

    ort.InferenceSession = _capped
from insightface.utils.face_align import norm_crop


HERE = os.path.dirname(os.path.abspath(__file__))
# FACE_PROVIDER=cuda on Colab/Kaggle GPUs; default CPU. Falls back to CPU if CUDA is unavailable.
if os.environ.get("FACE_PROVIDER") == "cuda" and hasattr(ort, "preload_dlls"):
    try: ort.preload_dlls()   # find CUDA/cuDNN libs shipped as pip packages
    except Exception: pass
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if os.environ.get("FACE_PROVIDER") == "cuda" else ["CPUExecutionProvider"]


@dataclass
class Config:
    align: bool = True          # 5-point similarity alignment (else plain bbox crop)
    quality_filter: bool = True  # reject unusable images (counted as failure-to-acquire)
    flip_tta: bool = False      # average embedding of image and its mirror
    min_det_score: float = 0.5
    min_face: int = 24          # shorter bbox side, pixels (in original image scale)
    min_blur: float = 20.0      # variance of Laplacian on the 112x112 aligned crop
    min_bright: float = 35.0    # mean gray
    max_bright: float = 225.0

    def name(self):
        return "|".join(f"{k}={v}" for k, v in asdict(self).items() if k in ("align", "quality_filter", "flip_tta"))


def imread(path):
    """cv2.imread returns None for non-ASCII paths on Windows (e.g. Urdu folder names); decoding the bytes works everywhere.
    Missing or unreadable files give None, which the pipeline reports as no_face."""
    try:
        buf = np.fromfile(path, np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, ValueError):
        return None


def _unit(e):
    e = np.asarray(e, dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


class FacePipeline:
    def __init__(self, cfg: Config = None, root=HERE, rec_onnx=None):
        """root must contain models/buffalo_l/ (detector + recognizer). rec_onnx optionally replaces the
        recognizer with a fine-tuned ArcFace ONNX file (same 112x112 -> 512-d interface)."""
        self.cfg = cfg or Config()
        self.app = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "recognition"],
                                providers=PROVIDERS)
        self.app.prepare(ctx_id=0 if PROVIDERS[0].startswith("CUDA") else -1, det_size=(640, 640))
        self.rec = self.app.models["recognition"]
        if PROVIDERS[0].startswith("CUDA"):
            self._require_gpu()
        if rec_onnx:
            from insightface.model_zoo import get_model
            self.rec = get_model(rec_onnx, providers=PROVIDERS)
            self.rec.prepare(ctx_id=0 if PROVIDERS[0].startswith("CUDA") else -1)

    def _require_gpu(self):
        """FACE_PROVIDER=cuda means "GPU or fail": never silently run on CPU."""
        for name, m in (("detector", self.app.det_model), ("recognizer", self.rec)):
            used = m.session.get_providers()
            if not used or used[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"FACE_PROVIDER=cuda but the {name} runs on {used}. "
                                   "Fix: pip uninstall -y onnxruntime && pip install onnxruntime-gpu (plain onnxruntime shadows the GPU build).")

    def align_crop(self, bgr):
        """Largest face aligned to the 112x112 ArcFace template, or None. Used to build training crops."""
        bgr, faces = self._detect(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return norm_crop(bgr, f.kps, 112)

    def _detect(self, bgr):
        """-> (image the faces refer to, faces). Retries with background padding for extreme close-ups
        (face fills the frame, e.g. halfFace), which SCRFD otherwise misses. Landmarks stay valid
        because the returned image is the padded one."""
        for pad in (0.0, 0.5):
            img = bgr
            if pad:
                p = int(pad * max(bgr.shape[:2]))
                img = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=[int(v) for v in bgr.reshape(-1, 3).mean(0)])
            sz = 320 if max(img.shape[:2]) <= 480 else 640   # small images need no 640 pass
            self.app.det_model.input_size = (sz, sz)
            faces = self.app.get(img)
            if faces:
                return img, faces
        return bgr, []

    def _embed(self, crop):
        e = self.rec.get_feat([crop, crop[:, ::-1].copy()])  # one batch: image + mirror
        return _unit(e[0]), _unit(np.asarray(e[0]) + np.asarray(e[1]))

    def analyze(self, bgr, scale_pad=True):
        if bgr is None:
            return {"status": "no_face"}
        s = 1.0
        if scale_pad and max(bgr.shape[:2]) < 300:   # tight crops (e.g. LFW) give SCRFD no context
            bgr = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); s = 2.0
        if scale_pad and max(bgr.shape[:2]) < 700:
            p = int(0.35 * max(bgr.shape[:2]))
            bgr = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REFLECT)
        bgr, faces = self._detect(bgr)
        if not faces:
            return {"status": "no_face"}
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        al = norm_crop(bgr, f.kps, 112)
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        h, w = bgr.shape[:2]
        bb = cv2.resize(bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)], (112, 112))
        gray = cv2.cvtColor(al, cv2.COLOR_BGR2GRAY)
        info = {"status": "ok", "det": float(f.det_score),
                "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "bright": float(gray.mean()),
                "size": float(min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) / s)}
        info["emb_a"], info["emb_a_flip"] = self._embed(al)
        info["emb_b"], info["emb_b_flip"] = self._embed(bb)
        return info

    @staticmethod
    def decide(info, cfg: Config):
        """-> (status, embedding or None) under cfg."""
        if info.get("status") != "ok":
            return "no_face", None
        if cfg.quality_filter and (info["det"] < cfg.min_det_score or info["size"] < cfg.min_face
                                   or info["blur"] < cfg.min_blur or not (cfg.min_bright <= info["bright"] <= cfg.max_bright)):
            return "rejected", None
        key = ("emb_a" if cfg.align else "emb_b") + ("_flip" if cfg.flip_tta else "")
        return "ok", info[key]

    def process(self, bgr):
        info = self.analyze(bgr)
        st, emb = self.decide(info, self.cfg)
        return {"status": st, "emb": emb, **{k: info.get(k) for k in ("det", "blur", "bright", "size")}}


def detect_all(pipe: FacePipeline, bgr, size=640, pad_retry=False):
    """All faces in a frame, no embedding yet (live use embeds only when needed).
    -> (image the boxes refer to, [{bbox:[x1,y1,x2,y2], kps:(5,2), det:float}]).
    pad_retry: if nothing is found, retry with background padding (extreme close-up selfies / ID photos);
    the returned image is then the padded one, so always use it for cropping."""
    for pad in ((0.0, 0.5) if pad_retry else (0.0,)):
        img = bgr
        if pad:
            p = int(pad * max(bgr.shape[:2]))
            img = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=[int(v) for v in bgr.reshape(-1, 3).mean(0)])
        pipe.app.det_model.input_size = (size, size)
        bboxes, kpss = pipe.app.det_model.detect(img, max_num=0, metric="default")
        if bboxes is not None and len(bboxes):
            return img, [{"bbox": b[:4].tolist(), "kps": k, "det": float(b[4])} for b, k in zip(bboxes, kpss)]
    return bgr, []


def embed_face(pipe: FacePipeline, bgr, kps, cfg: Config = None):
    """-> (embedding or None, aligned 112x112 crop, quality dict). None embedding = failed the quality filter."""
    cfg = cfg or pipe.cfg
    crop = norm_crop(bgr, kps, 112)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    q = {"blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "bright": float(gray.mean())}
    if cfg.quality_filter and (q["blur"] < cfg.min_blur or not (cfg.min_bright <= q["bright"] <= cfg.max_bright)):
        return None, crop, q
    e = pipe._embed(crop)
    return (e[1] if cfg.flip_tta else e[0]), crop, q
