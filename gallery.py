"""Open-set face gallery: enroll people once, recognise them later with NO retraining.

The embedding model stays frozen; "remembering" a person = storing their face embeddings under a name.
Unknown faces (best score < threshold) are reported as unknown, never forced onto a name.

  python gallery.py enroll --name "Ali Khan" --images DIR_OR_FILE... --consent
  python gallery.py list
  python gallery.py remove --name "Ali Khan"            # right-to-erasure
  python gallery.py identify --images a.jpg b.jpg       # who is in these images
  python gallery.py scan --input video.mp4|DIR --out report.csv [--every 1.0]   # who appears where/when
  python gallery.py reindex                              # recompute embeddings from stored crops (after swapping in a fine-tuned model)

Storage (default ./gallery): gallery.npz (names, embeddings), meta.json (consent record per person), faces/<name>/*.jpg (aligned crops).
"""
import argparse, csv, json, os, re, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import Config, FacePipeline, detect_all, embed_face  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DEFAULT_THR = 0.40   # placeholder until tuned on a Pakistani eval set (see config.json "identify_threshold")


def _slug(name):
    return re.sub(r"[^\w\-]+", "_", name.strip()) or "person"


def load_settings():
    p = os.path.join(HERE, "config.json")
    j = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rec = j.get("rec_onnx")
    return (Config(**j.get("pipeline", {})), os.path.join(HERE, rec) if rec else None,
            j.get("identify_threshold", DEFAULT_THR))


class Gallery:
    def __init__(self, path=os.path.join(HERE, "gallery")):
        self.path = path
        self.npz, self.metaf = os.path.join(path, "gallery.npz"), os.path.join(path, "meta.json")
        os.makedirs(os.path.join(path, "faces"), exist_ok=True)
        self.names, self.embs = [], np.zeros((0, 512), np.float32)
        self.meta = {}
        if os.path.exists(self.npz):
            d = np.load(self.npz)
            self.names, self.embs = list(d["names"]), d["emb"].astype(np.float32)
        if os.path.exists(self.metaf):
            self.meta = json.load(open(self.metaf, encoding="utf-8"))

    def save(self):
        np.savez(self.npz, names=np.array(self.names), emb=self.embs)
        json.dump(self.meta, open(self.metaf, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def people(self):
        out = {}
        for n in self.names:
            out[n] = out.get(n, 0) + 1
        return out

    def match(self, emb):
        """-> (name or None, score, runner_up_score). Score of a person = best cosine over their samples."""
        if not len(self.names):
            return None, -1.0, -1.0
        sims = self.embs @ emb
        best = {}
        for n, s in zip(self.names, sims):
            if s > best.get(n, -2):
                best[n] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return ranked[0][0], ranked[0][1], (ranked[1][1] if len(ranked) > 1 else -1.0)

    def add(self, name, emb, crop, consent_note, thr, force=False):
        """Enroll one sample. Refuses a face that already matches a DIFFERENT person (likely mislabel / duplicate identity)."""
        who, s, _ = self.match(emb)
        if who and who != name and s >= thr and not force:
            return False, f"looks like existing person '{who}' (score {s:.2f}); not enrolled"
        d = os.path.join(self.path, "faces", _slug(name))
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{int(time.time() * 1000)}_{len(self.names)}.jpg"), crop)
        self.names.append(name)
        self.embs = np.vstack([self.embs, emb[None]])
        m = self.meta.setdefault(name, {"consent": consent_note, "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        m["samples"] = self.people()[name]
        self.save()
        return True, "ok"

    def remove(self, name):
        keep = [i for i, n in enumerate(self.names) if n != name]
        removed = len(self.names) - len(keep)
        self.names = [self.names[i] for i in keep]
        self.embs = self.embs[keep] if keep else np.zeros((0, 512), np.float32)
        self.meta.pop(name, None)
        d = os.path.join(self.path, "faces", _slug(name))
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            os.rmdir(d)
        self.save()
        return removed

    def reindex(self, pipe):
        """Re-embed every stored crop with pipe's recognizer (the gallery is tied to the model that made it)."""
        names, embs = [], []
        for n in self.meta:
            d = os.path.join(self.path, "faces", _slug(n))
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                crop = cv2.imread(os.path.join(d, f))
                if crop is not None:
                    names.append(n); embs.append(pipe._embed(crop)[0])
        self.names, self.embs = names, (np.stack(embs) if embs else np.zeros((0, 512), np.float32))
        self.save()
        return len(names)


def _images_in(paths):
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(IMG_EXT):
                    yield os.path.join(p, f)
        elif os.path.isfile(p):
            yield p


def identify_frame(pipe, gal, bgr, thr, size=640, pad_retry=False):
    """-> list of {bbox, name|None, score, det} for every face in the frame (boxes refer to bgr, which is
    padded in place only if pad_retry had to kick in - callers drawing boxes should use identify_image)."""
    bgr, faces = detect_all(pipe, bgr, size, pad_retry)
    out = []
    for f in faces:
        emb, _, _ = embed_face(pipe, bgr, f["kps"])
        if emb is None:
            out.append({**f, "name": None, "score": -1.0, "low_quality": True}); continue
        who, s, _ = gal.match(emb)
        out.append({**f, "name": who if s >= thr else None, "score": s})
    return bgr, out


def draw(bgr, results):
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        ok = r["name"] is not None
        col = (60, 200, 60) if ok else (60, 60, 230)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
        label = f"{r['name']} {r['score']:.2f}" if ok else ("low quality" if r.get("low_quality") else "Unknown")
        cv2.putText(bgr, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=os.path.join(HERE, "gallery"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--name", required=True); e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--consent", action="store_true", help="confirm this person consented to being enrolled")
    e.add_argument("--force", action="store_true")
    sub.add_parser("list")
    r = sub.add_parser("remove"); r.add_argument("--name", required=True)
    i = sub.add_parser("identify"); i.add_argument("--images", nargs="+", required=True); i.add_argument("--save", help="dir for annotated copies")
    s = sub.add_parser("scan"); s.add_argument("--input", required=True); s.add_argument("--out", default="appearances.csv")
    s.add_argument("--every", type=float, default=1.0, help="seconds between analysed video frames")
    sub.add_parser("reindex")
    a = ap.parse_args()

    cfg, rec, thr = load_settings()
    gal = Gallery(a.gallery)
    if a.cmd == "list":
        for n, c in gal.people().items():
            print(f"{n}: {c} samples, consent={gal.meta.get(n, {}).get('consent')}, since {gal.meta.get(n, {}).get('enrolled_at')}")
        return
    if a.cmd == "remove":
        print(f"removed {gal.remove(a.name)} samples of {a.name}"); return
    pipe = FacePipeline(cfg, rec_onnx=rec)

    if a.cmd == "enroll":
        if not a.consent:
            sys.exit("refusing: pass --consent to confirm this person agreed to be enrolled")
        ok = 0
        for p in _images_in(a.images):
            img = cv2.imread(p)
            img, faces = detect_all(pipe, img, pad_retry=True) if img is not None else (None, [])
            if len(faces) != 1:
                print(f"skip {p}: {len(faces)} faces (need exactly 1)"); continue
            emb, crop, _ = embed_face(pipe, img, faces[0]["kps"])
            if emb is None:
                print(f"skip {p}: low quality"); continue
            done, why = gal.add(a.name, emb, crop, "confirmed via --consent flag", thr, a.force)
            ok += done
            if not done:
                print(f"skip {p}: {why}")
        print(f"enrolled {ok} samples for {a.name}")
    elif a.cmd == "identify":
        for p in _images_in(a.images):
            img = cv2.imread(p)
            img, res = identify_frame(pipe, gal, img, thr, pad_retry=True) if img is not None else (None, [])
            print(p, "->", [(r["name"] or "Unknown", round(r["score"], 2)) for r in res] or "no face")
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                cv2.imwrite(os.path.join(a.save, os.path.basename(p)), draw(img, res))
    elif a.cmd == "reindex":
        print("re-embedded", gal.reindex(pipe), "samples")
    elif a.cmd == "scan":
        rows = []
        if os.path.isfile(a.input) and a.input.lower().endswith(VID_EXT):
            cap = cv2.VideoCapture(a.input); fps = cap.get(cv2.CAP_PROP_FPS) or 25
            step, n = max(1, int(round(fps * a.every))), 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if n % step == 0:
                    for r in identify_frame(pipe, gal, fr, thr)[1]:
                        rows.append((a.input, f"{n / fps:.1f}s", r["name"] or "Unknown", f"{r['score']:.3f}"))
                n += 1
        else:
            for p in _images_in([a.input]):
                img = cv2.imread(p)
                for r in (identify_frame(pipe, gal, img, thr, pad_retry=True)[1] if img is not None else []):
                    rows.append((p, "", r["name"] or "Unknown", f"{r['score']:.3f}"))
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["source", "time", "person", "score"]); w.writerows(rows)
        seen = {}
        for _, _, n, _ in rows:
            seen[n] = seen.get(n, 0) + 1
        print(f"{len(rows)} detections -> {a.out}; per person: {seen}")


if __name__ == "__main__":
    main()
