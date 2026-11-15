"""Turn a video into a demo video: faces get a box and a name (or "Unknown").

  python demo/facevideo.py --people PEOPLE_DIR --video in.mp4 --out out.mp4 --consent

PEOPLE_DIR has one folder per person, named after them, holding 1-5 clear photos:
  PEOPLE_DIR/Ali Khan/1.jpg  PEOPLE_DIR/Ali Khan/2.jpg  PEOPLE_DIR/Sara/1.jpg ...

Runs offline in two passes:
  1. analyse: find faces, follow each one over time (a "track"), decide ONE name per track from all its frames.
     Deciding after seeing the whole track is what keeps labels stable, but it also means this cannot run live.
  2. render: draw boxes, name labels and the people panel, re-encode as H.264, keep the original audio.

Nothing is saved except the output video and an optional report: the enrolled faces stay in memory.
Uses the repo's detector + ArcFace R50 (mirror-averaged), not the R50 + R100 ensemble of the competition submission.
"""
import argparse, json, os, re, subprocess, sys
import cv2, imageio_ffmpeg, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pipeline import Config, FacePipeline, detect_all, embed_face  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KNOWN, UNKNOWN = (166, 230, 46), (70, 170, 255)      # BGR: green for named faces, amber for unknown ones
CUT_DIFF = 28.0        # mean pixel change between frames (on a tiny copy) that counts as a hard cut
MIN_FACE = 28          # pixels: smaller faces are too unreliable to name


# ---------------------------------------------------------------- video in / out
def probe(path, max_fps=30.0, max_side=1920):
    """-> (width, height, fps, approx frame count, ffmpeg filter). Size comes from a real decoded frame, so phone rotation is included."""
    gen = imageio_ffmpeg.read_frames(path)
    meta = next(gen); gen.close()
    fps = min(float(meta.get("fps") or 30.0), max_fps)
    vf = (f"fps={fps:.5f},scale=w='if(gt(iw,ih),min({max_side},iw),-2)':h='if(gt(iw,ih),-2,min({max_side},ih))',"
          "crop=trunc(iw/2)*2:trunc(ih/2)*2")
    png = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", vf, "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
                         capture_output=True).stdout
    first = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) if png else None
    if first is None:
        raise SystemExit(f"could not read a frame from {path}")
    h, w = first.shape[:2]
    return w, h, fps, int(float(meta.get("duration") or 0) * fps), vf


def frames(path, w, h, vf):
    """Constant-frame-rate BGR frames (phone videos are often variable-rate, which would drift against the audio)."""
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        p.stdout.close(); p.kill(); p.wait()


def encoder(path_in, path_out, w, h, fps):
    return subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.5f}",
                             "-i", "-", "-i", path_in, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", path_out],
                            stdin=subprocess.PIPE)


# ---------------------------------------------------------------- enrolment
def load_photo(path):
    with Image.open(path) as im:   # exif_transpose: phones store rotation in EXIF
        return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert("RGB")), cv2.COLOR_RGB2BGR)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def enroll(pipe, people_dir, log=print):
    """-> {name: {"embs": array, "thumb": aligned crop}}. A photo is skipped if it has no face or no clearly main face."""
    people = {}
    for name in sorted(os.listdir(people_dir)):
        d = os.path.join(people_dir, name)
        if not os.path.isdir(d):
            continue
        embs, thumb = [], None
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(IMG_EXT):
                continue
            try:
                img = load_photo(os.path.join(d, f))
            except OSError:
                log(f"  skip {name}/{f}: not a readable image"); continue
            img, fs = detect_all(pipe, img, pad_retry=True)
            fs = sorted(fs, key=lambda x: -area(x["bbox"]))
            if not fs:
                log(f"  skip {name}/{f}: no face found"); continue
            if len(fs) > 1 and area(fs[1]["bbox"]) > 0.45 * area(fs[0]["bbox"]):
                log(f"  skip {name}/{f}: {len(fs)} faces, cannot tell which one is {name}"); continue
            e, crop, _ = embed_face(pipe, img, fs[0]["kps"])
            embs.append(e)
            thumb = crop if thumb is None else thumb
        if embs:
            people[name] = {"embs": np.stack(embs), "thumb": thumb}
            log(f"  {name}: {len(embs)} photo(s) enrolled")
        else:
            log(f"  {name}: NOT enrolled, no usable photo")
    return people


# ---------------------------------------------------------------- pass 1: analyse
def match_faces(open_tracks, faces):
    """Greedy: each detection continues the nearest open track it overlaps. -> list of (track or None, face)."""
    pairs = []
    for ti, t in enumerate(open_tracks):
        a = t["boxes"][-1]
        ca, sa = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), np.sqrt(area(a))
        for fi, f in enumerate(faces):
            b = f["bbox"]
            d = np.hypot((b[0] + b[2]) / 2 - ca[0], (b[1] + b[3]) / 2 - ca[1]) / sa
            ratio = np.sqrt(area(b)) / sa
            if d < 0.9 and 0.6 < ratio < 1.6:
                pairs.append((d, ti, fi))
    used_t, used_f, out = set(), set(), []
    for _, ti, fi in sorted(pairs):
        if ti not in used_t and fi not in used_f:
            used_t.add(ti); used_f.add(fi); out.append((open_tracks[ti], faces[fi]))
    out += [(None, f) for i, f in enumerate(faces) if i not in used_f]
    return out


def analyse(pipe, path, w, h, fps, total, vf, people, det_size=640, det_per_sec=6.0, embed_per_sec=2.5, max_samples=30, progress=None):
    """Find and follow faces. A track ends at a hard cut or after 1 s unseen. Returns tracks with their sampled gallery scores."""
    names = list(people)
    step = max(1, int(round(fps / det_per_sec)))
    embed_gap = max(1, int(round(fps / embed_per_sec)))
    max_gap = int(fps)
    tracks, open_tracks, prev, cuts = [], [], None, []
    for n, fr in enumerate(frames(path, w, h, vf)):
        tiny = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = prev is not None and float(np.abs(tiny - prev).mean()) > CUT_DIFF
        prev = tiny
        if cut:
            cuts.append(n); open_tracks = []
        if n % step and not cut:
            continue
        faces = [f for f in detect_all(pipe, fr, det_size)[1]
                 if f["det"] >= 0.6 and min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= MIN_FACE]
        for t, f in match_faces(open_tracks, faces):
            if t is None:
                t = {"frames": [], "boxes": [], "scores": [], "last_embed": -10 ** 9, "det": []}
                tracks.append(t); open_tracks.append(t)
            t["frames"].append(n); t["boxes"].append(f["bbox"]); t["det"].append(f["det"])
            if n - t["last_embed"] >= embed_gap and len(t["scores"]) < max_samples:
                e, _, _ = embed_face(pipe, fr, f["kps"])
                t["scores"].append([float((people[p]["embs"] @ e).max()) for p in names])
                t["last_embed"] = n
        open_tracks = [t for t in open_tracks if n - t["frames"][-1] <= max_gap]
        if progress and total:
            progress("analyse", min(1.0, n / total))
    return tracks, cuts, n + 1 if tracks or prev is not None else 0


def decide(tracks, names, thr, margin=0.04):
    """One name per track: for each person take the mean of the track's best half of samples (ignores profile/blurred frames).
    Named only if the best person clears the threshold and is clearly ahead of the runner-up."""
    for t in tracks:
        t["name"], t["closest"], t["score"], t["runner_up"] = None, None, -1.0, -1.0
        if not t["scores"] or not names:
            continue
        s = np.array(t["scores"])
        k = max(1, int(np.ceil(len(s) / 2)))
        per = np.sort(s, axis=0)[-k:].mean(0)
        order = np.argsort(-per)
        t["closest"], t["score"] = names[order[0]], float(per[order[0]])
        t["runner_up"] = float(per[order[1]]) if len(order) > 1 else -1.0
        if t["score"] >= thr and t["score"] - t["runner_up"] >= margin:
            t["name"] = names[order[0]]


def finish_tracks(tracks, cuts, n_frames, fps):
    """Drop one-off detections, smooth the boxes, set the frame range each box is shown for, and stop one person being named twice at once."""
    hold = int(fps / 4)
    keep = []
    for t in tracks:
        if len(t["frames"]) < 2:
            continue
        b = np.array(t["boxes"], np.float32)
        if len(b) >= 3:     # light smoothing against detector jitter
            pad = np.vstack([b[0], b, b[-1]])
            b = (pad[:-2] + 2 * pad[1:-1] + pad[2:]) / 4
        t["boxes"] = b
        nxt = next((c for c in cuts if c > t["frames"][-1]), n_frames)
        t["start"], t["end"] = t["frames"][0], min(t["frames"][-1] + hold, nxt - 1)
        keep.append(t)
    for a in keep:
        for b in keep:
            if a is not b and a["name"] and a["name"] == b["name"] and a["start"] <= b["end"] and b["start"] <= a["end"] and a["score"] < b["score"]:
                a["clash"] = True
    for t in keep:
        if t.pop("clash", False):
            t["name"] = None
    return keep


# ---------------------------------------------------------------- pass 2: render
def font(px, bold=False):
    for f in (["seguisb.ttf", "segoeui.ttf"] if bold else ["segoeui.ttf"]) + ["arial.ttf", "DejaVuSans.ttf"]:
        for p in (f, os.path.join("C:/Windows/Fonts", f)):
            try:
                return ImageFont.truetype(p, px)
            except OSError:
                pass
    return ImageFont.load_default()


def blend(frame, sprite, x, y, opacity=1.0):
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:].astype(np.float32) / 255 * opacity
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + s[..., :3] * a).astype(np.uint8)


def to_sprite(im):
    return np.asarray(im)[..., [2, 1, 0, 3]]


class Overlay:
    """Everything drawn on top of the video. Sizes scale with the video so it looks the same at 720p, 1080p or vertical."""

    def __init__(self, w, h, people, show_panel=True):
        self.u = max(0.55, min(w, h) / 1080)
        self.f_name, self.f_score = font(int(30 * self.u), True), font(int(22 * self.u))
        self.labels = {}
        self.panel = self._panel(people) if show_panel and 0 < len(people) <= 8 else {}
        self.w, self.h = w, h

    def label(self, name, score):
        key = (name, round(score, 2))
        if key not in self.labels:
            u = self.u
            text, sub = name or "Unknown", f"{score:.2f}" if name else ""
            d = ImageDraw.Draw(Image.new("L", (1, 1)))
            tw, sw = d.textlength(text, font=self.f_name), d.textlength(sub, font=self.f_score) if sub else 0
            pw, ph = int(tw + sw + (24 + (16 if sub else 0)) * u), int(44 * u)
            im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            col = KNOWN if name else UNKNOWN
            d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(9 * u), fill=(col[2], col[1], col[0], 240))
            d.text((12 * u, ph / 2), text, font=self.f_name, fill=(12, 20, 16, 255), anchor="lm")
            if sub:
                d.text((pw - 12 * u, ph / 2), sub, font=self.f_score, fill=(12, 20, 16, 200), anchor="rm")
            self.labels[key] = to_sprite(im)
        return self.labels[key]

    def _panel(self, people):
        u, out = self.u, {}
        size, f = int(56 * u), font(int(24 * u), True)
        d = ImageDraw.Draw(Image.new("L", (1, 1)))
        width = int(max(d.textlength(n, font=f) for n in people) + size + 40 * u)
        for name, p in people.items():
            th = cv2.cvtColor(cv2.resize(p["thumb"], (size * 3, size * 3), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
            mask = Image.new("L", (size * 3, size * 3), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, size * 3 - 1, size * 3 - 1], fill=255)
            face = Image.fromarray(th).convert("RGBA"); face.putalpha(mask)
            face = face.resize((size, size), Image.LANCZOS)
            pair = []
            for on in (False, True):
                im = Image.new("RGBA", (width, size + int(12 * u)), (0, 0, 0, 0))
                dr = ImageDraw.Draw(im)
                dr.rounded_rectangle([0, 0, width - 1, im.height - 1], radius=int(14 * u),
                                     fill=(KNOWN[2], KNOWN[1], KNOWN[0], 235) if on else (16, 20, 24, 170))
                ph = face
                if not on:
                    g = ImageOps.grayscale(face.convert("RGB")).convert("RGBA"); g.putalpha(face.getchannel("A").point(lambda v: int(v * .75))); ph = g
                im.alpha_composite(ph, (int(6 * u), int(6 * u)))
                dr.text((size + 18 * u, im.height / 2), name, font=f, fill=(12, 20, 16, 255) if on else (235, 238, 240, 210), anchor="lm")
                pair.append(to_sprite(im))
            out[name] = pair
        return out

    def box(self, fr, b, col):
        u, (x1, y1, x2, y2) = self.u, [int(v) for v in b]
        ln, th = max(10, int(0.22 * min(x2 - x1, y2 - y1))), max(2, int(3 * u))
        for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(fr, (x, y), (x + dx * ln, y), col, th, cv2.LINE_AA)
            cv2.line(fr, (x, y), (x, y + dy * ln), col, th, cv2.LINE_AA)

    def draw_panel(self, fr, present):
        if not self.panel:
            return
        u = self.u
        y = self.h - int(20 * u)
        for name, pair in reversed(list(self.panel.items())):
            s = pair[1 if name in present else 0]
            y -= s.shape[0]
            blend(fr, s, int(20 * u), y)
            y -= int(8 * u)


def render(path, out, w, h, fps, total, vf, tracks, people, blur_unknown=False, show_panel=True, progress=None):
    ov = Overlay(w, h, people, show_panel)
    enc = encoder(path, out, w, h, fps)
    fade = max(1, int(fps / 4))
    for n, fr in enumerate(frames(path, w, h, vf)):
        present, shown = set(), []
        for t in tracks:
            if not t["start"] <= n <= t["end"]:
                continue
            xs = np.array(t["frames"])
            b = [float(np.interp(n, xs, t["boxes"][:, k])) for k in range(4)]
            if t["name"]:
                present.add(t["name"])
            elif blur_unknown:
                x1, y1, x2, y2 = max(int(b[0]), 0), max(int(b[1]), 0), min(int(b[2]), w), min(int(b[3]), h)
                if x2 > x1 and y2 > y1:
                    k = max(15, ((x2 - x1) // 3) | 1)
                    fr[y1:y2, x1:x2] = cv2.GaussianBlur(fr[y1:y2, x1:x2], (k, k), 0)
                continue            # a blurred bystander gets no box or label
            ov.box(fr, b, KNOWN if t["name"] else UNKNOWN)
            shown.append((t, b))
        # labels: named faces first, then Unknown from largest to smallest; an Unknown label that would cover another label is left out
        taken = []
        for t, b in sorted(shown, key=lambda p: (p[0]["name"] is None, -area(p[1]))):
            s = ov.label(t["name"], t["score"])
            y = int(b[1]) - s.shape[0] - int(8 * ov.u)
            if y < 0:
                y = int(b[3]) + int(8 * ov.u)
            x = int(min(max(b[0], 0), w - s.shape[1]))
            r = (x, y, x + s.shape[1], y + s.shape[0])
            if t["name"] is None and any(r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3] for o in taken):
                continue
            taken.append(r)
            blend(fr, s, x, y, min(1.0, (n - t["start"] + 1) / fade))
        ov.draw_panel(fr, present)
        enc.stdin.write(fr.tobytes())
        if progress and total:
            progress("render", min(1.0, n / total))
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the video")


# ---------------------------------------------------------------- whole job
def load_threshold():
    return json.load(open(os.path.join(HERE, "..", "config.json"), encoding="utf-8")).get("identify_threshold", 0.40)


def make_demo(people_dir, video, out, thr=None, blur_unknown=False, show_panel=True, det_size=640, progress=None, log=print):
    thr = load_threshold() if thr is None else thr
    pipe = FacePipeline(Config(quality_filter=False, flip_tta=True))
    log("enrolling:")
    people = enroll(pipe, people_dir, log)
    if not people:
        raise SystemExit("nobody could be enrolled: check the photos")
    w, h, fps, total, vf = probe(video)
    log(f"video: {w}x{h} at {fps:.2f} fps, about {total} frames")
    tracks, cuts, n = analyse(pipe, video, w, h, fps, total, vf, people, det_size, progress=progress)
    decide(tracks, list(people), thr)
    tracks = finish_tracks(tracks, cuts, n, fps)
    render(video, out, w, h, fps, n, vf, tracks, people, blur_unknown, show_panel, progress)
    report = [{"start_s": round(t["start"] / fps, 2), "end_s": round(t["end"] / fps, 2), "name": t["name"] or "Unknown",
               "closest": t["closest"], "score": round(t["score"], 3), "runner_up": round(t["runner_up"], 3),
               "samples": len(t["scores"])} for t in tracks]
    log(f"threshold {thr}; {len(tracks)} face tracks, {len(cuts)} cuts")
    for r in report:
        log(f"  {r['start_s']:7.2f}s - {r['end_s']:7.2f}s  {r['name']:24s} closest {str(r['closest']):24s} score {r['score']:.3f}  next {r['runner_up']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", required=True); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--consent", action="store_true", help="confirm everyone in --people agreed to be named in this video")
    ap.add_argument("--threshold", type=float, default=None, help="default: identify_threshold in config.json")
    ap.add_argument("--blur-unknown", action="store_true", help="blur faces that are not enrolled (bystanders)")
    ap.add_argument("--no-panel", action="store_true"); ap.add_argument("--det-size", type=int, default=640, help="1024 finds smaller faces, slower")
    ap.add_argument("--report", default=None, help="also save the per-face results as JSON")
    a = ap.parse_args()
    if not a.consent:
        sys.exit("refusing: pass --consent to confirm everyone in --people agreed to be named in this video")

    def progress(phase, f):
        print(f"\r{phase}: {f * 100:5.1f}%", end="", flush=True)
        if f >= 1:
            print()
    rep = make_demo(a.people, a.video, a.out, a.threshold, a.blur_unknown, not a.no_panel, a.det_size, progress,
                    lambda s: print("\n" + s if s.startswith("threshold") else s))
    if a.report:
        json.dump(rep, open(a.report, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
