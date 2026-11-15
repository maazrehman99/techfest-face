"""Photo drop page for teammates who cannot enrol in front of the webcam.

  python demo/collect.py        prints the secret local address; put a tunnel in front of it to share it

This is the ONLY thing meant to be reachable from the internet, so it does almost nothing:
every page needs the secret token in the address, it accepts images only, and it has no way to list or view what was sent.
Photos are re-saved as plain JPEG (this removes GPS and other metadata) into gallery_inbox/<random id>/, with meta.json
holding the name and the person's own consent.

Two secret addresses. Holding the public one gives no way to enrol anyone or to list who is enrolled:
  public  check (<token>) and enroll request (<token>/register), one page with two tabs. An older upload address redirects here.
  check   anyone's own camera or an uploaded picture: Known + name, or Unknown (reachable through the tunnel). Pictures are never saved and
          the gallery is never listed; an Unknown answer does not reveal the closest name.
  admin   see who is waiting and enrol them             (reachable through the tunnel from any device; only the owner should hold this address)
The face models load on the first check or enrolment, about 500 MB, so the upload page alone stays light.
"""
import io, json, os, re, secrets, shutil, sys, threading, time
from types import SimpleNamespace
from flask import Flask, Response, abort, jsonify, redirect, request
from PIL import Image, ImageOps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.environ.setdefault("FACE_THREADS", "2")
INBOX = os.path.join(ROOT, "gallery_inbox")
PORT = 7861
MAX_PHOTOS, MAX_SUBMISSIONS, MAX_SIDE = 6, 20, 1600
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


def load_token(fname=".token"):
    os.makedirs(INBOX, exist_ok=True)
    p = os.path.join(INBOX, fname)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(secrets.token_urlsafe(18))
    return open(p, encoding="utf-8").read().strip()


TOKEN, CHECK, ADMIN = load_token(), load_token(".check_token"), load_token(".admin_token")
busy = threading.Lock()        # one user of the models at a time; extra check requests are told to retry
load_lock, enroll_lock = threading.Lock(), threading.Lock()
M, last_seen, sent_by = {}, {}, {}
RECEIPTS = os.path.join(INBOX, ".receipts.json")     # submission id -> name, so a person can see their own status only


def models():
    """Detector + recognizer + gallery, loaded once. The gallery is re-read if the file changed."""
    with load_lock:
        if not M:
            from pipeline import Config, FacePipeline
            from gallery import load_settings
            M["pipe"] = FacePipeline(Config(quality_filter=False, flip_tta=True))
            M["thr"] = load_settings()[2]
        from gallery import Gallery
        npz = os.path.join(ROOT, "gallery", "gallery.npz")
        stamp = os.path.getmtime(npz) if os.path.exists(npz) else 0
        if M.get("stamp") != stamp:
            M["gal"], M["stamp"] = Gallery(os.path.join(ROOT, "gallery")), stamp
    return M


@app.post("/<token>/frame")
def frame(token):
    if not secrets.compare_digest(token, CHECK):
        abort(404)
    photo = request.args.get("photo") == "1"     # an uploaded picture: look harder, allow more faces
    if (request.content_length or 0) > 6 * 1024 * 1024:
        abort(413)
    who, now = request.headers.get("Cf-Connecting-IP") or request.remote_addr, time.time()
    if now - last_seen.get(who, 0) < 0.4:
        return jsonify(retry=True), 429
    if len(last_seen) > 500:
        last_seen.clear()
    last_seen[who] = now
    if not busy.acquire(blocking=False):
        return jsonify(retry=True), 429
    try:
        import cv2, numpy as np
        from pipeline import detect_all, embed_face
        m = models()
        try:
            with Image.open(io.BytesIO(request.get_data())) as im:
                bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return jsonify(error="not an image"), 400
        h, w = bgr.shape[:2]
        side = lambda f: min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])
        if photo:      # two detector sizes: 640 catches big close-up faces, 1280 catches small faces in group photos
            bb, kp = m["pipe"].app.det_model.detect(bgr, input_size=[(640, 640), (1280, 1280)], max_num=0, metric="default")
            found = [{"bbox": b[:4].tolist(), "kps": k, "det": float(b[4])} for b, k in zip(bb, kp)] if len(bb) else []
        else:
            found = detect_all(m["pipe"], bgr, 640)[1]
        fs = sorted([f for f in found if f["det"] >= 0.5 and side(f) >= 20], key=lambda f: -side(f))
        limit = 15 if photo else 6                      # largest faces first; each one adds about half a second
        extra, fs = max(0, len(fs) - limit), fs[:limit]
        out = []
        for f in fs:
            e, _, _ = embed_face(m["pipe"], bgr, f["kps"])
            name, score, _ = m["gal"].match(e)
            known = name is not None and score >= m["thr"]
            x1, y1, x2, y2 = f["bbox"]
            # a very small face that does not match is "too far", not a real Unknown: there was too little detail to say
            out.append({"box": [x1 / w, y1 / h, x2 / w, y2 / h], "name": str(name) if known else None,
                        "score": round(float(score), 2) if known else None, "small": bool(side(f) < 40)})
        return jsonify(faces=out, extra=extra)      # the frame is not kept: it only ever existed in this function
    finally:
        busy.release()


def waiting_dir(rid):
    """The folder of one waiting submission, or 404. The id is checked first, so a request can never reach outside the inbox."""
    d = os.path.join(INBOX, rid)
    if not re.fullmatch(r"[0-9a-f]{12}", rid) or not os.path.isfile(os.path.join(d, "meta.json")):
        abort(404)
    return d


@app.get("/<token>/status")
def status(token):
    if not secrets.compare_digest(token, ADMIN):
        abort(404)
    waiting = []
    for d in os.listdir(INBOX):
        p = os.path.join(INBOX, d, "meta.json")
        if os.path.isfile(p):
            waiting.append({"id": d, "name": json.load(open(p, encoding="utf-8"))["name"],
                            "photos": sorted(f for f in os.listdir(os.path.join(INBOX, d)) if f.endswith(".jpg"))})
    npz = os.path.join(ROOT, "gallery", "gallery.npz")
    people = {}
    if os.path.exists(npz):
        import numpy as np
        for n in np.load(npz)["names"]:
            people[str(n)] = people.get(str(n), 0) + 1
    return jsonify(waiting=waiting, people=people)


@app.get("/<token>/photo/<rid>/<fname>")
def preview(token, rid, fname):
    """A small preview, so the person holding the private address can see who they are approving or rejecting."""
    if not secrets.compare_digest(token, ADMIN) or not re.fullmatch(r"\d+\.jpg", fname):
        abort(404)
    path = os.path.join(waiting_dir(rid), fname)
    if not os.path.isfile(path):
        abort(404)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((240, 240))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return Response(buf.getvalue(), mimetype="image/jpeg")


@app.post("/<token>/reject/<rid>")
def reject(token, rid):
    """Deletes one waiting submission for good, and lets its sender see that it was not added."""
    if not secrets.compare_digest(token, ADMIN):
        abort(404)
    d = waiting_dir(rid)
    if not enroll_lock.acquire(blocking=False):      # never delete a folder while enrolment is reading it
        return jsonify(message="Enrolling right now, please try again in a moment."), 409
    try:
        name = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))["name"]
        shutil.rmtree(d)
        with load_lock:
            receipts = read_receipts()
            receipts[rid + ":rejected"] = True
            with open(RECEIPTS, "w", encoding="utf-8") as out:
                json.dump(receipts, out, ensure_ascii=False)
    finally:
        enroll_lock.release()
    return jsonify(message=f"{name} was rejected. Their photos were deleted.")


@app.post("/<token>/enroll")
def enroll(token):
    if not secrets.compare_digest(token, ADMIN):
        abort(404)
    if not any(os.path.isfile(os.path.join(INBOX, d, "meta.json")) for d in os.listdir(INBOX)):
        return jsonify(message="No photos have been sent yet.")     # no need to load the models for this
    if not enroll_lock.acquire(blocking=False):     # the page may be open on two devices
        return jsonify(message="Already enrolling, please wait a moment."), 409
    try:
        from live import enroll_inbox
        m = models()
        return jsonify(message=enroll_inbox(SimpleNamespace(pipe=m["pipe"], gal=m["gal"], thr=m["thr"], pipe_lock=busy)))
    finally:
        enroll_lock.release()


@app.after_request
def no_store(r):
    r.headers["Cache-Control"] = "no-store"
    r.headers["X-Robots-Tag"] = "noindex"
    r.headers["Referrer-Policy"] = "no-referrer"
    return r


def check(token):
    if not (secrets.compare_digest(token, CHECK) or secrets.compare_digest(token, TOKEN)):
        abort(404)


def read_receipts():
    try:
        return json.load(open(RECEIPTS, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.get("/<token>/register")
def register_page(token):
    if not secrets.compare_digest(token, CHECK):
        abort(404)
    return PAGE


@app.get("/<token>/receipt/<rid>")
def receipt(token, rid):
    """Status of one submission, for whoever holds its unguessable id. This cannot be used to ask about any other name."""
    if not secrets.compare_digest(token, CHECK) or not re.fullmatch(r"[0-9a-f]{12}", rid):
        abort(404)
    name = read_receipts().get(rid)
    if name is None:
        abort(404)
    if read_receipts().get(rid + ":rejected"):
        return jsonify(state="not_added", name=name)
    if os.path.isfile(os.path.join(INBOX, rid, "meta.json")):
        return jsonify(state="waiting", name=name)
    key = lambda n: " ".join(n.split()).casefold()
    try:
        people = json.load(open(os.path.join(ROOT, "gallery", "meta.json"), encoding="utf-8"))
    except (OSError, ValueError):
        people = {}
    match = next((n for n in people if key(n) == key(name)), None)
    return jsonify(state="enrolled", name=match) if match else jsonify(state="not_added", name=name)


@app.get("/<token>")
def page(token):
    if secrets.compare_digest(token, CHECK):
        return CHECK_PAGE
    if secrets.compare_digest(token, ADMIN):
        return ADMIN_PAGE
    if secrets.compare_digest(token, TOKEN):      # the old upload link keeps working
        return redirect(f"/{CHECK}/register")
    abort(404)


@app.post("/<token>/send")
def send(token):
    check(token)
    who, now = request.headers.get("Cf-Connecting-IP") or request.remote_addr, time.time()
    recent = [t for t in sent_by.get(who, []) if now - t < 600]
    if len(recent) >= 5:      # the address is now public, so one device cannot fill the waiting list
        return jsonify(error="Too many sends from this device. Please try again in a few minutes."), 429
    if len([d for d in os.listdir(INBOX) if os.path.isdir(os.path.join(INBOX, d))]) >= MAX_SUBMISSIONS:
        return jsonify(error="This link is full. Please tell the person who sent it to you."), 429
    name = re.sub(r"[^A-Za-z .'\-]", "", request.form.get("name", "")).strip()[:60]
    if len(name) < 2:
        return jsonify(error="Please type your name in English letters."), 400
    if request.form.get("consent") != "yes":
        return jsonify(error="Please tick the box to agree first."), 400
    files = request.files.getlist("photos")
    if not 1 <= len(files) <= MAX_PHOTOS:
        return jsonify(error=f"Please choose 1 to {MAX_PHOTOS} photos."), 400
    photos, bad = [], 0
    for f in files:
        try:
            with Image.open(io.BytesIO(f.read())) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=92)      # saved without EXIF, so no location data is kept
                photos.append(buf.getvalue())
        except Exception:    # anything Pillow cannot decode is simply not a usable photo
            bad += 1
    if not photos:
        return jsonify(error="Those files could not be opened as photos. Please use JPG or PNG."), 400
    rid = secrets.token_hex(6)
    d = os.path.join(INBOX, rid)
    os.makedirs(d)
    for k, b in enumerate(photos):
        with open(os.path.join(d, f"{k}.jpg"), "wb") as out:
            out.write(b)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as out:
        json.dump({"name": name, "consent": f"ticked by the person on the upload page, {time.strftime('%Y-%m-%d %H:%M')}"}, out, ensure_ascii=False)
    with load_lock:
        receipts = read_receipts()
        receipts[rid] = name
        with open(RECEIPTS, "w", encoding="utf-8") as out:
            json.dump(receipts, out, ensure_ascii=False)
    sent_by[who] = recent + [now]
    return jsonify(saved=len(photos), skipped=bad, id=rid, name=name)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enroll Your Face</title><!--HEAD--><style>
.scan{position:relative;width:min(70vw,290px);aspect-ratio:1;margin:26px auto 18px}
#ring{position:absolute;inset:-16px;width:calc(100% + 32px);height:calc(100% + 32px);transform:rotate(-90deg);overflow:visible}
.seg{fill:none;stroke:var(--line);stroke-width:2.2;stroke-linecap:round;transition:stroke .25s}.seg.done{stroke:var(--glow)}.seg.now{stroke:var(--ink);animation:pulse 1.2s ease-in-out infinite}.holding .seg.now{stroke:var(--glow);animation:none}
@keyframes pulse{50%{opacity:.35}}
.lens{position:absolute;inset:0;border-radius:50%;overflow:hidden;background:radial-gradient(circle at 50% 38%,#17404d,#0a212b 72%);display:grid;place-items:center}
.lens video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);display:none}.scan.live video{display:block}.scan.live .art,.scan.complete .art{display:none}
.art{width:46%;color:var(--dim)}.tickmark{display:none;width:40%;color:var(--glow)}.scan.complete .tickmark{display:block}.scan.complete .lens{box-shadow:inset 0 0 60px rgba(95,240,210,.22)}
.flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .25s}.flash.on{opacity:.75;transition:none}
#arrow{position:absolute;display:none;width:42px;height:42px;border-radius:50%;background:rgba(5,18,24,.72);color:var(--glow);place-items:center;left:50%;top:50%;margin:-21px}
#arrow svg{width:22px}#arrow[data-dir=l],#arrow[data-dir=r],#arrow[data-dir=u],#arrow[data-dir=d]{display:grid}
#arrow[data-dir=l]{transform:translateX(-112px)}#arrow[data-dir=r]{transform:translateX(112px) rotate(180deg)}#arrow[data-dir=u]{transform:translateY(-112px) rotate(90deg)}#arrow[data-dir=d]{transform:translateY(112px) rotate(-90deg)}
.stepn{color:var(--glow);font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;text-align:center;margin:0 0 2px}.stepn:empty{display:none}
.tip{font-family:var(--display);font-size:21px;font-weight:700;line-height:1.25;color:var(--ink);text-align:center;margin:0 0 6px;text-wrap:balance}
.hint{color:var(--dim);font-size:14px;text-align:center;margin:0 0 12px;min-height:21px}.small{font-size:13px;min-height:0;margin:12px 0 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}#open{margin-bottom:4px}
details{margin-top:18px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--dim);font-size:14px;padding:6px 0}
.drop{position:relative;display:block;border:1px dashed var(--line);border-radius:12px;padding:18px;text-align:center;color:var(--ink);cursor:pointer;font-weight:500;margin:8px 0 0}.drop input{position:absolute;opacity:0;inset:0;cursor:pointer}.drop:focus-within{outline:2px solid var(--glow);outline-offset:2px}
.count{color:var(--glow);font-size:14px;font-weight:600;margin:16px 0 8px}.count:empty,.camerr:empty{display:none}.camerr{color:var(--red);font-size:14px;margin:8px 0}
.thumbs{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:6px}.thumbs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;display:block}
.shot{position:relative;width:auto;min-height:0;padding:0;background:none;border:0;border-radius:10px}.shot i{position:absolute;top:-5px;right:-5px;width:20px;height:20px;border-radius:50%;background:var(--abyss);border:1px solid var(--line);color:var(--ink);font:12px/18px sans-serif;text-align:center;font-style:normal}
.tick{display:flex;gap:10px;font-weight:400;font-size:13px;line-height:1.45;color:var(--dim);margin:18px 0 14px}.tick input{width:18px;height:18px;margin-top:1px;accent-color:var(--glow);flex:none}
#msg{margin:12px 0 0;text-align:center}.err{color:var(--red)}.ok{color:var(--glow);font-weight:600}
.thanks{text-align:center;padding:22px 0 8px}.thanks svg{width:64px;color:var(--glow);margin-bottom:10px}.thanks h2{font-family:var(--display);font-size:24px;margin:0 0 6px}.thanks p{margin:0 0 18px}
@media (prefers-reduced-motion:reduce){.seg.now{animation:none}.flash{transition:none}}
</style></head><body><main><p class="eyebrow">Tech Fest &rsquo;26 &middot; Student demo</p><!--NAV--><h1>Enroll your face</h1><p class="lede">Takes about a minute. Turn your head when asked, and your photos are taken for you.</p>
<form id="f" class="card"><label for="n">Your name, as it should appear</label><input type="text" id="n" maxlength="60" autocomplete="name" placeholder="In English letters">
<div class="scan" id="scan"><svg id="ring" viewBox="0 0 100 100" aria-hidden="true"></svg><div class="lens"><video id="v" playsinline muted autoplay></video>
<svg class="art" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="24" cy="17" r="9"/><path d="M8 43c1.5-9.5 8-14 16-14s14.5 4.500 16 14"/></svg>
<svg class="tickmark" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 25l9 9 19-20"/></svg><div class="flash" id="flash"></div></div>
<span id="arrow" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg></span></div>
<p class="stepn" id="stepn"></p><p class="tip" id="tip" aria-live="polite"></p><p class="hint" id="hint"></p>
<button type="button" id="open">Start face scan</button><p class="camerr" id="camerr" role="alert"></p>
<div id="cam" hidden><div class="two"><button type="button" id="snap" class="ghost">Take photo now</button><button type="button" id="done" class="ghost">Close camera</button></div><p class="hint small">Your photos stay on your device until you press Send.</p></div>
<details><summary>Use photos you already have instead</summary><label class="drop"><input type="file" id="p" accept="image/*" multiple><span>Choose photos</span></label></details>
<p class="count" id="count"></p><div class="thumbs" id="th"></div>
<label class="tick"><input type="checkbox" id="c"><span>I agree that my photos can be used to recognise me, that my name and face can appear in a demo video posted on LinkedIn, and that anyone with the demo link who points a camera at me, or uses or uploads a photo of me, will see my name. I can ask for them to be removed at any time.</span></label>
<button id="b">Send photos</button><p id="msg" role="alert"></p></form></main><!--SHARED--><script>
const $=i=>document.getElementById(i);
const MAX=6,VER='0.10.14',CDN='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@'+VER,MODEL='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
// y = sideways turn, p = chin up or down, both measured from the person's own straight-ahead pose. Either direction is
// accepted first, then the other one is required, so nobody gets stuck by turning the wrong way.
const IDLE='Find good light. No sunglasses or mask, and only you in the picture.';
const STEPS=[{t:'Look straight at the camera',ok:(y,p)=>Math.abs(y)<.14},
 {t:'Slowly turn your head left',dir:'l',ok:y=>Math.abs(y)>=.17&&Math.abs(y)<.9,side:1},
 {t:'Now slowly turn your head right',dir:'r',ok:y=>y*turn<=-.17&&y*turn>-.9},
 {t:'Look ahead, then lift your chin up',dir:'u',ok:(y,p)=>Math.abs(y)<.14&&Math.abs(p)>=.07,tilt:1},
 {t:'Now lower your chin down',dir:'d',ok:(y,p)=>Math.abs(y)<.14&&p*tilt<=-.07},
 {t:'Look straight and smile',ok:(y,p)=>Math.abs(y)<.1&&Math.abs(p)<.04}];
let shots=[],stream=null,lm=null,lmFail=false,idx=0,base=null,turn=1,tilt=1,hold=0,last=null,loopId=0,lastT=-1,cool=0;
const hint=t=>{$('hint').textContent=t;$('scan').classList.toggle('holding',t==='Hold still...')};
$('ring').append(...STEPS.map((_,i)=>{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',50);c.setAttribute('cy',50);c.setAttribute('r',48);c.setAttribute('pathLength',60);c.setAttribute('stroke-dasharray','8.4 51.6');c.setAttribute('stroke-dashoffset',-(i*10+.8));c.setAttribute('class','seg');return c}));
function ring(){const finished=idx>=STEPS.length||shots.length>=MAX;[...$('ring').children].forEach((c,i)=>c.setAttribute('class','seg'+(i<idx?' done':i===idx&&stream?' now':'')));
 $('scan').classList.toggle('live',!!stream);$('scan').classList.toggle('complete',finished&&!stream);$('arrow').dataset.dir=stream&&STEPS[idx]?STEPS[idx].dir||'':'';
 $('stepn').textContent=stream&&STEPS[idx]?'Photo '+(idx+1)+' of '+STEPS.length:'';
 $('tip').textContent=stream?(shots.length>=MAX?'That is plenty. Close the camera and send.':STEPS[idx]?STEPS[idx].t:'One more if you like, then send.'):finished?'All your photos are ready':IDLE;
 $('open').textContent=finished?'Retake the photos':idx>0?'Continue face scan':'Start face scan';$('open').className=finished?'ghost':''}
async function loadLM(){const v=await import(CDN+'/vision_bundle.mjs'),fs=await v.FilesetResolver.forVisionTasks(CDN+'/wasm');
 const make=d=>v.FaceLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:MODEL,delegate:d},runningMode:'VIDEO',numFaces:2});
 try{return await make('GPU')}catch(e){return await make('CPU')}}
// One reading of the head: count = faces seen, size = gap between the eyes as a share of the picture. Runs on every detector result.
function onPose(raw,rawP,count,size,now){if(!stream||idx>=STEPS.length||shots.length>=MAX)return;
 if(!count){hold=0;last=null;return hint('Move your face into the circle')}if(count>1){hold=0;return hint('Only you in the picture, please')}if(size<.09){hold=0;return hint('Come a little closer')}
 if(!base&&idx>0)base={y:raw,p:rawP};
 const y=raw-(base?base.y:0),p=rawP-(base?base.p:rawP),st=STEPS[idx],moving=last&&(Math.abs(y-last.y)>.04||Math.abs(p-last.p)>.03);last={y,p};
 if(now<cool||!st.ok(y,p)||moving){hold=0;return hint(now<cool?'Got it':'')}
 if(!hold)hold=now;hint('Hold still...');if(now-hold<350)return;
 if(!base)base={y:raw,p:rawP};if(st.side)turn=Math.sign(y)||1;if(st.tilt)tilt=Math.sign(p)||1;
 hold=0;cool=now+700;capture(true)}
function capture(auto){const v=$('v');if(!v.videoWidth||shots.length>=MAX)return;const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);
 const f=$('flash');f.classList.add('on');requestAnimationFrame(()=>requestAnimationFrame(()=>f.classList.remove('on')));if(navigator.vibrate)navigator.vibrate(60);
 if(idx<STEPS.length)idx++;
 c.toBlob(b=>{if(!b)return;b.cam=true;shots.push(b);show();if(auto&&(idx>=STEPS.length||shots.length>=MAX)){closeCam();$('msg').className='ok';$('msg').textContent='All done. Check your name, tick the box and press Send.'}},'image/jpeg',.92)}
function watch(id){if(id!==loopId||!stream||!lm)return;const v=$('v');
 if(v.readyState>=2&&v.currentTime!==lastT){lastT=v.currentTime;try{const f=lm.detectForVideo(v,performance.now()).faceLandmarks||[];
  if(f[0]){const a=f[0],W=v.videoWidth,H=v.videoHeight,X=i=>a[i].x*W,Y=i=>a[i].y*H,ex=(X(33)+X(263))/2,ey=(Y(33)+Y(263))/2,gap=Math.hypot(X(263)-X(33),Y(263)-Y(33))||1;
   onPose((X(1)-ex)/gap,(Y(1)-ey)/((Y(13)-ey)||1),f.length,gap/Math.min(W,H),performance.now())}else onPose(0,0,0,0,performance.now())}catch(e){}}
 setTimeout(()=>watch(id),70)}
async function startAuto(){const id=++loopId;hold=0;last=null;lastT=-1;
 if(!lm&&!lmFail){hint('Getting ready, a few seconds the first time...');
  try{lm=await Promise.race([loadLM(),new Promise((_,r)=>setTimeout(()=>r(new Error('slow')),45000))])}catch(e){lmFail=true}}
 if(!stream||id!==loopId)return;
 if(lmFail){hint('Automatic photos are not available on this device. Please use the button below.');$('snap').className='';return}
 hint('');watch(id)}
function show(){$('msg').textContent='';$('th').replaceChildren(...shots.map((b,i)=>{const s=document.createElement('button');s.type='button';s.className='shot';s.setAttribute('aria-label','Remove photo '+(i+1));
  const m=new Image();m.src=URL.createObjectURL(b);m.alt='';const x=document.createElement('i');x.textContent='\u00d7';s.append(m,x);s.onclick=()=>{if(b.cam&&idx>0)idx--;shots.splice(i,1);show()};return s}));
 $('count').textContent=shots.length?shots.length+' of '+MAX+' photos ready'+(shots.length<3?', 3 or more is best':''):'';
 ring();$('snap').disabled=shots.length>=MAX}
function closeCam(){loopId++;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;$('cam').hidden=true;$('open').hidden=false;$('hint').textContent='';ring()}
$('open').onclick=async()=>{$('camerr').textContent='';if(idx>=STEPS.length||shots.length>=MAX){shots=shots.filter(b=>!b.cam);idx=0;base=null;show()}
 if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){$('camerr').textContent='This browser cannot open the camera here. If you opened the link inside WhatsApp, open it in Chrome or Safari instead, or choose photos below.';return}
 try{try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:1280},height:{ideal:1280}},audio:false})}
  catch(x){if(x.name==='NotAllowedError')throw x;stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false})}   // some cameras refuse the detailed request
  $('v').srcObject=stream;$('cam').hidden=false;$('open').hidden=true;show();startAuto()}
 catch(x){const why={NotAllowedError:'Camera permission was blocked. Allow the camera in your browser settings, then try again.',
   NotReadableError:'Another app is using your camera. Close it (for example Zoom, WhatsApp or another browser tab), then try again.',
   NotFoundError:'No camera was found on this device.',AbortError:'The camera did not start. Close other apps using it, then try again.'};
  $('camerr').textContent=(why[x.name]||'The camera could not be opened in this browser. Try Chrome or Safari.')+' You can also choose photos below. (Error: '+x.name+')'}};
$('done').onclick=closeCam;
$('snap').onclick=()=>capture(false);
ring();
$('p').onchange=()=>{for(const f of $('p').files)if(shots.length<MAX)shots.push(f);$('p').value='';show()};
$('f').onsubmit=async e=>{e.preventDefault();const fd=new FormData();fd.append('name',$('n').value);fd.append('consent',$('c').checked?'yes':'no');shots.forEach((b,i)=>fd.append('photos',b,'photo'+i+'.jpg'));
 if(!shots.length){$('msg').className='err';$('msg').textContent='Please take or choose at least one photo.';return}
 closeCam();$('b').disabled=true;$('msg').className='';$('msg').textContent='Sending...';
 try{const r=await fetch(BASE+'/send',{method:'POST',body:fd});if(r.status===413)throw new Error('Those photos are too large together. Please send fewer.');const j=await r.json();if(!r.ok)throw new Error(j.error);
  setReceipt(j.id,j.name);refreshMe();$('f').innerHTML='<div class="thanks"><svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="24" cy="24" r="21"/><path d="M14 25l7 7 13-15"/></svg><h2 class="ok">Thank you, your photos were received</h2><p>Waiting to be enrolled. You will be recognised once the team adds you.</p><a class="linkbtn" href="'+BASE+'">Go to Check</a></div>'}
 catch(x){$('msg').className='err';$('msg').textContent=x.message;$('b').disabled=false}};
</script></body></html>"""


CHECK_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Are You Known?</title><!--HEAD--><style>
.top{display:flex;gap:8px;margin:20px 0 12px}#next{padding:12px 14px;background:var(--surface);border-radius:12px;margin-bottom:10px;font-size:15px;text-wrap:pretty}.err{color:var(--red);font-size:14px}.err:empty{display:none}
.sonar{position:relative;aspect-ratio:16/10;border:1px solid var(--line);border-radius:18px;overflow:hidden;background:radial-gradient(circle at 50% 50%,#123744,#0a212b 70%);display:grid;place-items:center}
.sonar:has(+ #on:not([hidden])){display:none}.sonar i{position:absolute;left:50%;top:50%;aspect-ratio:1;border:1px solid rgba(95,240,210,.2);border-radius:50%;transform:translate(-50%,-50%)}.sonar i:nth-child(1){width:26%}.sonar i:nth-child(2){width:52%}.sonar i:nth-child(3){width:78%}
.sonar b{position:relative;background:rgba(5,18,24,.72);padding:8px 14px;border-radius:99px;color:var(--ink);font-size:14px;font-weight:500}
.stage{position:relative;line-height:0;border-radius:18px;overflow:hidden;background:#000;margin-bottom:12px;border:1px solid var(--line)}video{width:100%;height:auto}video.mirror{transform:scaleX(-1)}#pic{width:100%;display:block}canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.stage:has(video:not([hidden]))::after{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:var(--glow);box-shadow:0 0 18px 3px rgba(95,240,210,.55);opacity:.7;animation:sweep 3.2s linear infinite;pointer-events:none}
@keyframes sweep{to{top:100%}}
#verdict{padding:16px;border-radius:14px;text-align:center;font-family:var(--display);font-weight:700;font-size:21px;line-height:1.25;background:var(--surface);border:1px solid var(--line);margin-bottom:10px;overflow-wrap:anywhere;text-wrap:balance}
#verdict.known{background:var(--glow);color:#05241e;border-color:var(--glow)}#verdict.unknown{background:var(--coral);color:#2c1208;border-color:var(--coral)}
#list{margin:0 0 12px;padding:0;list-style:none;display:grid;gap:6px}#list li{display:flex;gap:10px;align-items:center;padding:10px 14px;background:var(--surface);border-radius:12px;overflow-wrap:anywhere}#list li::before{content:"";flex:none;width:9px;height:9px;border-radius:50%;background:currentColor}
#list li.known{color:var(--glow);font-weight:700}#list li.unknown{color:var(--coral)}#list li.far{color:var(--dim)}.row{display:flex;gap:8px}.note{font-size:13px;margin:18px 0 0;color:var(--dim)}
@media (prefers-reduced-motion:reduce){.stage::after{animation:none!important;display:none}}
</style></head><body><main><p class="eyebrow">Tech Fest &rsquo;26 &middot; Student demo</p><!--NAV--><h1>Are you in our database?</h1><p class="lede">Use your camera or upload a photo. One person or a whole group, it checks every face it can see.</p>
<div class="top"><button id="go">Start camera</button><button class="ghost" id="up">Upload a photo</button></div><input type="file" id="pf" accept="image/*" hidden><p class="err" id="err" role="alert"></p>
<div class="sonar" aria-hidden="true"><i></i><i></i><i></i><b>Waiting for a face</b></div>
<div id="on" hidden><div class="stage"><video id="v" playsinline muted autoplay></video><img id="pic" alt="Your uploaded photo" hidden><canvas id="o"></canvas></div><div id="verdict" role="status" aria-live="polite">Starting, the first check can take a minute</div><div id="next" hidden></div><ol id="list"></ol><div class="row"><button class="ghost" id="flip">Switch camera</button><button class="ghost" id="stop">Stop camera</button></div></div>
<p class="note">Camera pictures and uploaded photos go to the team&rsquo;s laptop only to be checked, and are never saved. This is a student demo, so it can make mistakes. Only use it on people who are happy with that.</p></main><!--SHARED--><script>
const $=i=>document.getElementById(i);let stream=null,facing='user',mirror=true;const c=document.createElement('canvas');
function verdict(cls,t){$('verdict').className=cls;$('verdict').textContent=t}
const who=f=>f.name||(f.small?(picUrl?'Too small to check':'Too far, move closer'):'Unknown');
let lastFaces=[],nextKey='';
function nextStep(){const n=$('next'),f=lastFaces.length===1?lastFaces[0]:null;
 const key=!f||(!f.name&&f.small)?'':f.name?'known|'+f.name:MY==='waiting'?'waiting':MY==='enrolled'?'unknown-enrolled':'unknown';if(key===nextKey)return;nextKey=key;n.hidden=!key;n.replaceChildren();if(!key)return;
 const link=(t,h)=>{const a=document.createElement('a');a.textContent=t;a.href=h;return a},add=BASE+'/register';
 if(f&&f.name)n.append('Already enrolled. You can check again anytime, or ',link('add more photos',add+'#name='+encodeURIComponent(f.name)),'.');
 else if(key==='waiting')n.append('Not recognised yet. Photos sent from this device are waiting to be enrolled.');
 else if(key==='unknown-enrolled')n.append('Not recognised this time. Try better light, or ',link('add more photos',add),'.');
 else n.append('Not enrolled yet. ',link('Enroll here',add),' and the team will add you.')}
window.onMe=()=>{nextKey='?';nextStep()};
const resetResult=()=>{lastFaces=[];nextStep();$('list').replaceChildren();draw([])};
function show(fs,extra){fs.sort((a,b)=>mirror?b.box[2]-a.box[2]:a.box[0]-b.box[0]);const many=fs.length>1;
 fs.forEach((f,i)=>{const w=f.name||(f.small?(picUrl?'Too small':'Too far'):'Unknown');f.tag=many?(i+1)+'\u00a0'+w:who(f)});draw(fs);
 const k=fs.filter(f=>f.name).length,far=fs.filter(f=>!f.name&&f.small).length,u=fs.length-k-far;
 if(!fs.length)verdict('','No face in view');else if(!many){const f=fs[0];f.name?verdict('known','Known: '+f.name):f.small?verdict('',who(f)):verdict('unknown','Unknown')}
 else verdict(k?'known':u?'unknown':'',fs.length+' people: '+k+' known, '+u+' unknown'+(far?', '+far+(picUrl?' too small':' too far'):'')+(extra?', '+extra+' more not checked':''));
 lastFaces=fs;nextStep();
 $('list').replaceChildren(...(many?fs.map((f,i)=>{const l=document.createElement('li');l.className=f.name?'known':f.small?'far':'unknown';l.textContent=(i+1)+'. '+who(f);return l}):[]))}
// Full names always: the text shrinks and wraps to fit the space each person has, so more people means smaller text, never a cut name.
function fitLabel(g,text,room,cap=15){for(const size of [15,14,13,12,11,10].filter(z=>z<=cap)){g.font='600 '+size+'px "Instrument Sans","Segoe UI",sans-serif';const lines=[];let now='';
  for(const word of text.split(/ +/)){const next=now?now+' '+word:word;if(now&&g.measureText(next).width>room){lines.push(now);now=word}else now=next}lines.push(now);
  if(size===10||(lines.length<=3&&Math.max(...lines.map(l=>g.measureText(l).width))<=room))return{size,lines,width:Math.max(...lines.map(l=>g.measureText(l).width))}}}
function draw(fs){const o=$('o'),r=o.getBoundingClientRect(),d=window.devicePixelRatio||1;o.width=r.width*d;o.height=r.height*d;const g=o.getContext('2d');g.scale(d,d);g.lineWidth=2.5;const placed=[],cap=fs.length<3?15:fs.length===3?13:fs.length===4?12:11;
 const roomFor=f=>Math.min(r.width-16,Math.max(70,(f.box[2]-f.box[0])*r.width+10,r.width/fs.length-16)),same=fs.length?Math.min(...fs.map(f=>fitLabel(g,f.tag,roomFor(f),cap).size)):cap;   // every label the same size
 for(const f of fs){const x=(mirror?1-f.box[2]:f.box[0])*r.width,y=f.box[1]*r.height,w=(f.box[2]-f.box[0])*r.width,h=(f.box[3]-f.box[1])*r.height,col=f.name?'#5ff0d2':f.small?'#9ebfc2':'#ff9b71';
  g.strokeStyle=col;g.strokeRect(x,y,w,h);const L=fitLabel(g,f.tag,roomFor(f),same),lh=L.size+3,tw=L.width+12,th=L.lines.length*lh+7;
  let lx=Math.max(0,Math.min(x,r.width-tw)),ly=y>=th?y-th:y+h;ly=Math.max(0,Math.min(ly,r.height-th));
  for(let tries=0;tries<fs.length;tries++){const hit=placed.find(b=>lx<b.x+b.w&&lx+tw>b.x&&ly<b.y+b.h&&ly+th>b.y);if(!hit)break;ly=hit.y-th>=0?hit.y-th:hit.y+hit.h;ly=Math.min(ly,r.height-th)}
  placed.push({x:lx,y:ly,w:tw,h:th});g.fillStyle=col;g.beginPath();g.roundRect?g.roundRect(lx,ly,tw,th,6):g.rect(lx,ly,tw,th);g.fill();g.fillStyle='#05241e';L.lines.forEach((t,i)=>g.fillText(t,lx+6,ly+3+lh*(i+1)-3))}}
async function tick(){if(!stream)return;const v=$('v');let wait=350;
 if(v.videoWidth){const s=Math.min(1,960/Math.max(v.videoWidth,v.videoHeight));c.width=v.videoWidth*s;c.height=v.videoHeight*s;c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  try{const b=await new Promise(r=>c.toBlob(r,'image/jpeg',.8));const r=await fetch(location.pathname.replace(/\/$/,'')+'/frame',{method:'POST',body:b});
   if(r.status===429)wait=1200;else if(!r.ok)throw 0;else{const j=await r.json();if(!stream)return;show(j.faces,j.extra)}}
  catch(e){verdict('','Connection problem, retrying');wait=2500}}
 setTimeout(tick,wait)}
async function open(){const was=stream;stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:facing,width:{ideal:1280},height:{ideal:720}},audio:false});mirror=facing==='user';$('v').className=mirror?'mirror':'';$('v').srcObject=stream;clearPhoto();$('v').hidden=false;$('flip').hidden=false;$('stop').textContent='Stop camera';$('on').hidden=false;$('go').hidden=true;if(!was){resetResult();verdict('','Checking...');tick()}}
let picUrl=null;function clearPhoto(){if(picUrl)URL.revokeObjectURL(picUrl);picUrl=null;$('pic').hidden=true;$('pic').removeAttribute('src')}
function stopCam(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null}
$('up').onclick=()=>$('pf').click();
$('pf').onchange=async()=>{const file=$('pf').files[0];$('pf').value='';if(!file)return;$('err').textContent='';stopCam();clearPhoto();picUrl=URL.createObjectURL(file);const pic=$('pic');pic.src=picUrl;
 try{await pic.decode()}catch(e){clearPhoto();$('on').hidden=true;$('go').hidden=false;$('err').textContent='That photo could not be opened. Please use a JPG or PNG.';return}
 mirror=false;$('v').hidden=true;pic.hidden=false;$('flip').hidden=true;$('stop').textContent='Close photo';$('on').hidden=false;$('go').hidden=false;resetResult();verdict('','Checking, this can take a little while');
 const s=Math.min(1,1600/Math.max(pic.naturalWidth,pic.naturalHeight));c.width=Math.round(pic.naturalWidth*s);c.height=Math.round(pic.naturalHeight*s);c.getContext('2d').drawImage(pic,0,0,c.width,c.height);const mine=picUrl;
 const b=await new Promise(r=>c.toBlob(r,'image/jpeg',.88));
 for(let i=0;i<40;i++){if(mine!==picUrl)return;try{const r=await fetch(location.pathname.replace(/\/$/,'')+'/frame?photo=1',{method:'POST',body:b});
   if(r.status===429){await new Promise(z=>setTimeout(z,1500));continue}if(r.status===413){verdict('','That photo is too large');return}if(!r.ok)throw 0;const j=await r.json();if(mine===picUrl)show(j.faces,j.extra);return}
  catch(e){verdict('','Connection problem. Please try again.');return}}
 verdict('','The demo is busy. Please try again in a minute.')};
$('flip').onclick=async()=>{const old=stream;old.getTracks().forEach(t=>t.stop());facing=facing==='user'?'environment':'user';
 try{await open()}catch(x){facing=facing==='user'?'environment':'user';try{await open();verdict('','This device has only one camera')}catch(y){stream=null;$('on').hidden=true;$('go').hidden=false;$('err').textContent='The camera could not be opened. Try Chrome or Safari.'}}};
$('go').onclick=async()=>{$('err').textContent='';if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){$('err').textContent='This browser cannot open the camera here. If you opened the link inside WhatsApp, open it in Chrome or Safari.';return}
 try{await open()}
 catch(x){$('err').textContent=(x.name==='NotAllowedError'?'Camera permission was blocked. Allow it in your browser settings, then try again.':x.name==='NotReadableError'?'Another app is using your camera. Close it, then try again.':'The camera could not be opened. Try Chrome or Safari.')}};
$('stop').onclick=()=>{stopCam();clearPhoto();lastFaces=[];nextStep();$('on').hidden=true;$('go').hidden=false};
</script></body></html>"""

ADMIN_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enroll Photos</title><!--HEAD--><style>
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);font-weight:600;margin:0 0 10px}ul{margin:0 0 4px;padding:0;list-style:none;display:grid;gap:6px}li{display:flex;justify-content:space-between;gap:12px;padding:10px 14px;background:var(--abyss);border-radius:10px;overflow-wrap:anywhere}li small{color:var(--dim);font-size:14px;flex:none}li.none{color:var(--dim)}
.card+.card{margin-top:12px}li.wait{display:block}.who{display:flex;justify-content:space-between;gap:12px;font-weight:600}.previews{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:10px 0}.previews img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;display:block;background:var(--surface)}.reject{min-height:44px;background:transparent;border-color:var(--line);color:var(--red);font-weight:600}.reject.sure{background:var(--red);border-color:var(--red);color:#2a0b0b}#b{margin-top:12px}#msg{white-space:pre-wrap;color:var(--glow);margin-top:10px;font-size:14px}#msg:empty{display:none}
</style></head><body><main><p class="eyebrow">Private team page</p><h1>Enroll sent photos</h1><p class="lede">Reject anyone who should not be added, then enroll the rest. Keep this link to yourself: it shows people&rsquo;s photos.</p><section class="card"><h2>Waiting</h2><ul id="w"></ul><button id="b">Enroll everyone waiting</button><div id="msg" role="status"></div></section><section class="card"><h2>In the database</h2><ul id="p"></ul></section></main><script>
const $=i=>document.getElementById(i),base=location.pathname.replace(/\/$/,'');
function li(t,n){const l=document.createElement('li');l.textContent=t;if(n===undefined)l.className='none';else{const m=document.createElement('small');m.textContent=n+(n===1?' photo':' photos');l.append(m)}return l}
function person(x){const l=document.createElement('li');l.className='wait';const top=document.createElement('div');top.className='who';const nm=document.createElement('span');nm.textContent=x.name;const m=document.createElement('small');m.textContent=x.photos.length+(x.photos.length===1?' photo':' photos');top.append(nm,m);
 const pv=document.createElement('div');pv.className='previews';x.photos.forEach(f=>{const i=new Image();i.src=base+'/photo/'+x.id+'/'+f;i.alt='Photo sent by '+x.name;i.loading='lazy';pv.append(i)});
 const r=document.createElement('button');r.className='reject';r.textContent='Reject';let t=0;r.onclick=async()=>{if(!r.classList.contains('sure')){r.classList.add('sure');r.textContent='Tap again to delete these photos';t=setTimeout(()=>{r.classList.remove('sure');r.textContent='Reject'},4000);return}
  clearTimeout(t);r.disabled=true;r.textContent='Rejecting...';try{const j=await(await fetch(base+'/reject/'+x.id,{method:'POST'})).json();$('msg').textContent=j.message}catch(e){$('msg').textContent='The connection dropped. Please try again.'}sig='';load()};
 l.append(top,pv,r);return l}
let sig='';async function load(){let j,raw;try{raw=await(await fetch(base+'/status')).text();j=JSON.parse(raw)}catch(e){return}if(raw===sig)return;sig=raw;
 $('w').replaceChildren(...(j.waiting.length?j.waiting.map(person):[li('Nothing waiting')]));$('b').hidden=!j.waiting.length;
 const e=Object.entries(j.people);$('p').replaceChildren(...(e.length?e.map(([n,c])=>li(n,c)):[li('Nobody yet')]))}
$('b').onclick=async()=>{$('b').disabled=true;$('msg').textContent='Working, the first time can take a minute...';try{const r=await(await fetch(base+'/enroll',{method:'POST'})).json();$('msg').textContent=r.message}catch(e){$('msg').textContent='The connection dropped. Reload this page: anyone already added will show under In the database.'}$('b').disabled=false;load()};load();setInterval(load,5000);
</script></body></html>"""


HEAD = r"""<meta name="color-scheme" content="dark"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"><style>
:root{--abyss:#07171f;--surface:#102c37;--line:#2a5461;--ink:#e9f7f5;--dim:#a3c3c6;--glow:#5ff0d2;--coral:#ff9b71;--red:#ff9090;--display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;min-height:100vh;color:var(--ink);font:16px/1.5 "Instrument Sans","Segoe UI",system-ui,sans-serif;background:radial-gradient(90vmax 60vmax at 50% -12%,#14505c 0,transparent 68%),var(--abyss)}
main{max-width:560px;margin:auto;padding:32px 16px 56px}.eyebrow{color:var(--glow);font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px}
h1{font-family:var(--display);font-size:clamp(30px,8vw,40px);font-weight:800;line-height:1.08;letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance}.lede{color:var(--dim);font-size:17px;margin:0 0 20px}p{margin:0 0 14px}
.card{background:rgba(16,44,55,.72);border:1px solid var(--line);border-radius:20px;padding:20px}label{display:block;margin:0 0 6px;font-weight:600}
input[type=text]{width:100%;min-height:48px;background:var(--abyss);border:1px solid var(--line);color:var(--ink);border-radius:12px;padding:11px 13px;font:inherit}input::placeholder{color:#7fa3a8}
button{width:100%;min-height:48px;background:var(--glow);border:1px solid var(--glow);color:#05241e;font:700 16px "Instrument Sans","Segoe UI",system-ui,sans-serif;padding:11px 14px;border-radius:12px;cursor:pointer;transition:transform .12s,background .15s}button:active{transform:scale(.98)}button:disabled{opacity:.5;cursor:default}
.tabs{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;background:rgba(16,44,55,.72);border:1px solid var(--line);border-radius:14px;margin:0 0 22px}.tabs a{min-height:44px;display:grid;place-items:center;border-radius:10px;color:var(--dim);font-weight:600;text-decoration:none}.tabs a[aria-current]{background:var(--glow);color:#05241e}
#me{padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:12px;margin:0 0 20px;font-size:15px;overflow-wrap:anywhere}#me.waiting,#me.not_added{border-color:var(--coral)}#me.enrolled{border-color:var(--glow)}main a:not(.linkbtn):not(.tabs a){color:var(--glow);font-weight:600}
.linkbtn{display:inline-grid;place-items:center;min-height:48px;padding:0 22px;border-radius:12px;background:var(--glow);color:#05241e;font-weight:700;text-decoration:none}
.ghost{background:transparent;border-color:var(--line);color:var(--ink);font-weight:500}.ghost:hover{background:rgba(95,240,210,.08)}:focus-visible{outline:2px solid var(--glow);outline-offset:2px}
@media (prefers-reduced-motion:reduce){button{transition:none}}
</style>"""
NAV = """<nav class="tabs" aria-label="Sections"><a id="tabC">Check</a><a id="tabR">Enroll</a></nav><div id="me" role="status" hidden></div>"""
SHARED = r"""<script>
const IS_REG=/\/register\/?$/.test(location.pathname),BASE=location.pathname.replace(/\/register\/?$/,'').replace(/\/$/,'');let MY='none',MYNAME='';
document.getElementById('tabC').href=BASE;document.getElementById('tabR').href=BASE+'/register';document.getElementById(IS_REG?'tabR':'tabC').setAttribute('aria-current','page');
function setReceipt(id,name){try{localStorage.setItem('face_receipt',JSON.stringify({id,name}))}catch(e){}}
function getReceipt(){try{return JSON.parse(localStorage.getItem('face_receipt'))}catch(e){return null}}
function paintMe(){const m=document.getElementById('me');m.replaceChildren();m.hidden=MY==='none';m.className=MY;if(MY==='none')return;const b=document.createElement('b');b.textContent=MYNAME;
 const link=(t,h)=>{const a=document.createElement('a');a.textContent=t;a.href=h;return a};
 if(MY==='waiting')m.append(b,' is waiting to be enrolled. You will be recognised once the team adds you.');
 else if(MY==='enrolled')m.append(b,' is enrolled. ',link('Check now',BASE),' or ',link('add more photos',BASE+'/register'),'.');
 else m.append(b,' was not added. Please ',link('enroll again',BASE+'/register'),'.')}
function fillName(v){const n=document.getElementById('n');if(n&&!n.value)n.value=v}
async function refreshMe(){const r=getReceipt();if(!r||!r.id)return;try{const x=await fetch(BASE+'/receipt/'+r.id);if(x.status===404){try{localStorage.removeItem('face_receipt')}catch(e){}MY='none'}else if(x.ok){const j=await x.json();MY=j.state;MYNAME=j.name;if(IS_REG&&MY==='enrolled')fillName(MYNAME)}else return}catch(e){return}paintMe();if(window.onMe)window.onMe()}
if(IS_REG&&location.hash.startsWith('#name=')){try{fillName(decodeURIComponent(location.hash.slice(6)))}catch(e){}}
refreshMe();setInterval(()=>{if(MY==='waiting'&&!document.hidden)refreshMe()},15000);
</script>"""
PAGE, CHECK_PAGE, ADMIN_PAGE = (x.replace("<!--HEAD-->", HEAD).replace("<!--NAV-->", NAV).replace("<!--SHARED-->", SHARED) for x in (PAGE, CHECK_PAGE, ADMIN_PAGE))


if __name__ == "__main__":
    print(f"public page: http://127.0.0.1:{PORT}/{CHECK}", flush=True)
    print(f"enrol page:  http://127.0.0.1:{PORT}/{ADMIN}", flush=True)
    app.run("127.0.0.1", PORT, threaded=True)
