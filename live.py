"""Live recognition demo in the browser (works with headless OpenCV, no display needed).

  python live.py [--source 0|video.mp4|rtsp://...] [--port 8000]
Open http://127.0.0.1:8000 : live annotated stream, enroll button (name + consent tick), people list with delete.
Enrolling takes the largest face in the current frame; click 3-5 times with slight head turns.
Recognition is per-track: a new/unknown face is matched immediately, a known one re-checked every few seconds.
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import FacePipeline, detect_all, embed_face  # noqa: E402
from gallery import Gallery, load_settings, draw  # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8><title>Face Memory</title>
<style>body{font:15px system-ui;margin:16px;background:#111;color:#eee}img{max-width:100%;border-radius:8px}
input,button{font:inherit;padding:6px 10px;margin:4px 0}.row{display:flex;gap:24px;flex-wrap:wrap}
li{margin:4px 0}#msg{color:#8f8;min-height:1.4em}</style>
<h2>Face Memory (live)</h2><div class=row><div><img src=/stream></div><div>
<h3>Enroll</h3><input id=name placeholder="Full name"><br>
<label><input type=checkbox id=consent> This person consented to being enrolled</label><br>
<button onclick=enroll()>Enroll face in view</button><div id=msg></div>
<h3>Remembered</h3><ul id=ppl></ul></div></div>
<script>
async function refresh(){let r=await (await fetch('/people')).json();
ppl.innerHTML=r.map(p=>`<li>${p.name} (${p.samples}) <button onclick="del('${p.name.replace(/'/g,"\\\\'")}')">forget</button></li>`).join('')}
async function enroll(){let r=await (await fetch('/enroll',{method:'POST',body:JSON.stringify({name:name.value,consent:consent.checked})})).json();
msg.textContent=r.message;refresh()}
async function del(n){await fetch('/forget',{method:'POST',body:JSON.stringify({name:n})});refresh()}
refresh()</script>"""


class State:
    def __init__(self, pipe, gal, thr):
        self.pipe, self.gal, self.thr = pipe, gal, thr
        self.lock = threading.Lock()
        self.jpeg, self.frame, self.faces = None, None, []
        self.tracks, self.next_id = {}, 0


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
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str) and not source.isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue   # loop video files
            time.sleep(0.2); continue
        with st.lock:
            st.frame = frame.copy()
        res = step(st, frame)
        with st.lock:
            st.faces = res
        ok, buf = cv2.imencode(".jpg", draw(frame, res), [cv2.IMWRITE_JPEG_QUALITY, 80])
        st.jpeg = buf.tobytes()
        time.sleep(max(0, 1 / fps_cap - 0.001))


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
            elif self.path == "/stream":
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if st.jpeg:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(st.jpeg) + st.jpeg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            name = (body.get("name") or "").strip()
            if self.path == "/forget":
                return self._json({"message": f"forgot {st.gal.remove(name)} samples of {name}"})
            if not name:
                return self._json({"message": "type a name first"})
            if not body.get("consent"):
                return self._json({"message": "tick the consent box first"})
            with st.lock:
                frame = None if st.frame is None else st.frame.copy()
            fs = detect_all(st.pipe, frame, 640)[1] if frame is not None else []
            if not fs:
                return self._json({"message": "no face in view"})
            f = max(fs, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            emb, crop, _ = embed_face(st.pipe, frame, f["kps"])
            if emb is None:
                return self._json({"message": "face too blurry/dark, try again"})
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
    print(f"open http://127.0.0.1:{a.port}  (threshold {thr})")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st)).serve_forever()


if __name__ == "__main__":
    main()
