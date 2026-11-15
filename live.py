"""Live recognition demo in the browser (works with headless OpenCV, no display needed).

  python live.py [--source 0|video.mp4|rtsp://...] [--port 8000]
Open http://127.0.0.1:8000 : live annotated stream, enroll button (name + consent tick), people list with delete.
Enrolling takes the largest face in the current frame; click 3-5 times with slight head turns.
Recognition is per-track: a new/unknown face is matched immediately, a known one re-checked every few seconds.
"""
import argparse, json, os, shutil, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FACE_THREADS", "2")   # leave CPU free for the camera and video stream, so the picture stays smooth
from pipeline import FacePipeline, detect_all, embed_face, imread  # noqa: E402
from gallery import Gallery, load_settings, draw  # noqa: E402

INBOX = os.path.join(HERE, "gallery_inbox")   # photos sent through demo/collect.py

PAGE = """<!doctype html><meta charset=utf-8><title>Face Memory</title>
<style>body{font:15px system-ui;margin:16px;background:#111;color:#eee}img{max-width:100%;border-radius:8px}
input,button{font:inherit;padding:6px 10px;margin:4px 0}.row{display:flex;gap:24px;flex-wrap:wrap}
li{margin:4px 0}#msg{color:#8f8;min-height:1.4em}</style>
<h2>Face Memory (live)</h2><div class=row><div><img src=/stream></div><div>
<h3>Enroll</h3><input id=pname placeholder="Full name"><br>
<label><input type=checkbox id=consent> This person consented to being enrolled</label><br>
<button onclick=enroll()>Enroll face in view</button><div id=msg></div>
<h3>Photos sent by teammates</h3><button onclick=inbox()>Enroll sent photos</button><div id=imsg style="white-space:pre-wrap;color:#8f8"></div>
<h3>Remembered</h3><ul id=ppl></ul></div></div>
<script>
async function refresh(){let r=await (await fetch('/people')).json();
ppl.replaceChildren(...r.map(p=>{let li=document.createElement('li');li.textContent=p.name+' ('+p.samples+') ';
let b=document.createElement('button');b.textContent='forget';b.onclick=()=>del(p.name);li.append(b);return li}))}
async function enroll(){let r=await (await fetch('/enroll',{method:'POST',body:JSON.stringify({name:pname.value,consent:consent.checked})})).json();
msg.textContent=r.message;refresh()}
async function inbox(){imsg.textContent='Working...';let r=await (await fetch('/enroll_inbox',{method:'POST'})).json();imsg.textContent=r.message;refresh()}
async function del(n){await fetch('/forget',{method:'POST',body:JSON.stringify({name:n})});refresh()}
refresh()</script>"""


class State:
    def __init__(self, pipe, gal, thr):
        self.pipe, self.gal, self.thr = pipe, gal, thr
        self.lock = threading.Lock()
        self.pipe_lock = threading.Lock()   # one user of the models and the gallery at a time
        self.jpeg, self.frame, self.faces, self.seq = None, None, [], 0
        self.tracks, self.next_id = {}, 0
        self.cam_fps, self.recog_ms = 0.0, 0.0


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def step(st, frame, recheck=3.0):
    """Detect every frame; embed only for new tracks, unknown tracks, or known ones not checked for `recheck` s."""
    now, faces = time.time(), detect_all(st.pipe, frame, 320)[1]
    old, new, used = st.tracks, {}, set()
    out = []
    for f in sorted(faces, key=lambda f: -f["det"]):
        best = max(((iou(f["bbox"], t["bbox"]), k) for k, t in old.items() if k not in used), default=(0, None))
        t = old[best[1]] if best[0] > 0.3 else {"id": st.next_id, "name": None, "score": -1.0, "checked": 0.0}
        if best[0] > 0.3:
            used.add(best[1])
        else:
            st.next_id += 1
        t = dict(t, bbox=f["bbox"], kps=f["kps"])
        if now - t["checked"] > (recheck if t["name"] else 0.5):
            emb, _, _ = embed_face(st.pipe, frame, f["kps"])
            t["checked"] = now
            if emb is not None:
                who, s, _ = st.gal.match(emb)
                t["name"], t["score"] = (who if s >= st.thr else None), s
        new[t["id"]] = t
        out.append({"bbox": t["bbox"], "name": t["name"], "score": t["score"]})
    st.tracks = new
    return out


def capture_loop(st, source, fps_cap):
    """Camera thread: always shows the NEWEST frame with the latest known boxes. It never waits for the face models,
    which is what stops old frames queueing up and the picture falling behind."""
    is_file = not str(source).isdigit() and "://" not in str(source)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    gap = 1 / (cap.get(cv2.CAP_PROP_FPS) or fps_cap) if is_file else 1 / fps_cap
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue   # loop video files
            time.sleep(0.2); continue
        with st.lock:
            st.frame = frame
            faces = st.faces
        ok, buf = cv2.imencode(".jpg", draw(frame.copy(), faces), [cv2.IMWRITE_JPEG_QUALITY, 75])
        st.jpeg = buf.tobytes()
        st.seq += 1
        now = time.time()
        st.cam_fps = 0.9 * st.cam_fps + 0.1 / max(now - last, 1e-3)
        wait = gap - (now - last)
        last = now
        if wait > 0:
            time.sleep(wait)


def recognise_loop(st):
    """Face thread: works on whichever frame is newest when it is free, skipping the ones in between."""
    done = -1
    while True:
        with st.lock:
            frame, seq = st.frame, st.seq
        if frame is None or seq == done:
            time.sleep(0.01); continue
        done, t0 = seq, time.time()
        with st.pipe_lock:
            res = step(st, frame)
        with st.lock:
            st.faces = res
        st.recog_ms = 0.8 * st.recog_ms + 0.2 * (time.time() - t0) * 1000
        time.sleep(0.02)


def same_person_name(st, name):
    """'ALI KHAN' and 'Ali  Khan' are one person: reuse the spelling already in the gallery."""
    key = " ".join(name.split()).casefold()
    return next((n for n in st.gal.people() if " ".join(str(n).split()).casefold() == key), " ".join(name.split()))


def enroll_inbox(st):
    """Enrol photos sent through demo/collect.py. The originals are deleted once a person is enrolled (their face crops stay in the gallery)."""
    if not os.path.isdir(INBOX):
        return "No photos have been sent yet."
    lines = []
    for sub in sorted(os.listdir(INBOX)):
        d = os.path.join(INBOX, sub)
        if not os.path.isfile(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        name, ok, why = same_person_name(st, meta["name"]), 0, []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jpg"):
                continue
            img = imread(os.path.join(d, f))
            with st.pipe_lock:
                img, fs = detect_all(st.pipe, img, 640, pad_retry=True) if img is not None else (None, [])
                big = sorted(fs, key=lambda x: -(x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
                area = lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])
                if not big:
                    why.append("no face found"); continue
                if len(big) > 1 and area(big[1]) > 0.45 * area(big[0]):
                    why.append("more than one face"); continue
                emb, crop, _ = embed_face(st.pipe, img, big[0]["kps"])
                if emb is None:
                    why.append("too blurry or dark"); continue
                added, msg = st.gal.add(name, emb, crop, meta["consent"], st.thr)
            ok += added
            if not added:
                why.append(msg.split(" (score")[0])   # the same reason once, not once per photo
        note = f" (skipped: {'; '.join(sorted(set(why)))})" if why else ""
        lines.append(f"{name}: {ok} photo(s) enrolled{note}")
        if ok:
            shutil.rmtree(d)
    return "\n".join(lines) or "No photos have been sent yet."


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif self.path == "/people":
                self._json([{"name": n, "samples": c} for n, c in st.gal.people().items()])
            elif self.path == "/stats":
                self._json({"camera_fps": round(st.cam_fps, 1), "recognition_ms": round(st.recog_ms)})
            elif self.path == "/stream":
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                sent = -1
                try:
                    while True:
                        jpeg, seq = st.jpeg, st.seq
                        if jpeg and seq != sent:      # send each new frame once, as soon as it exists
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
                            sent = seq
                        time.sleep(0.01)
                except ConnectionError:     # the browser tab was closed
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            port = self.server.server_port
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.send_error(403)    # another website open in the browser must not control this page
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            name = (body.get("name") or "").strip()
            if self.path == "/enroll_inbox":
                return self._json({"message": enroll_inbox(st)})
            if self.path == "/forget":
                with st.pipe_lock:
                    n = st.gal.remove(name)
                return self._json({"message": f"forgot {n} samples of {name}"})
            if not name:
                return self._json({"message": "type a name first"})
            if not body.get("consent"):
                return self._json({"message": "tick the consent box first"})
            with st.lock:
                frame = None if st.frame is None else st.frame.copy()
            with st.pipe_lock:
                fs = detect_all(st.pipe, frame, 640)[1] if frame is not None else []
                if not fs:
                    return self._json({"message": "no face in view"})
                f = max(fs, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                emb, crop, _ = embed_face(st.pipe, frame, f["kps"])
                if emb is None:
                    return self._json({"message": "face too blurry/dark, try again"})
                name = same_person_name(st, name)
                ok, why = st.gal.add(name, emb, crop, "confirmed via live UI checkbox", st.thr)
            self._json({"message": f"enrolled sample {st.gal.people()[name]} for {name}" if ok else why})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gallery", default=os.path.join(HERE, "gallery")); ap.add_argument("--fps", type=float, default=15)
    a = ap.parse_args()
    cfg, rec, thr = load_settings()
    st = State(FacePipeline(cfg, rec_onnx=rec), Gallery(a.gallery), thr)
    threading.Thread(target=capture_loop, args=(st, a.source, a.fps), daemon=True).start()
    threading.Thread(target=recognise_loop, args=(st,), daemon=True).start()
    print(f"open http://127.0.0.1:{a.port}  (threshold {thr})")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st)).serve_forever()


if __name__ == "__main__":
    main()
