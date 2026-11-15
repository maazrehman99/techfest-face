"""Local web interface for the demo-video maker.

  python demo/app.py        then open http://127.0.0.1:7860

Add people with a few photos each, pick a video, press the button, watch and download the result.
Runs only on this computer (127.0.0.1). Uploads and results stay in demo/work/ until "Delete everything" is pressed.
"""
import os, re, shutil, sys, threading, time, traceback, uuid
from flask import Flask, abort, jsonify, request, send_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from facevideo import IMG_EXT, load_threshold, make_demo  # noqa: E402

WORK = os.path.join(HERE, "work")
PORT = 7860
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
app = Flask(__name__)
jobs, lock = {}, threading.Lock()


def clean_name(s):
    """A person's name becomes a folder name and a label: drop what Windows forbids, keep Urdu and other letters."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s or "").strip().strip(".")
    return s[:60]


def busy():
    return any(j["state"] == "running" for j in jobs.values())


def worker(job, people, video, out, thr, blur, panel):
    def progress(phase, f):
        job["phase"], job["fraction"] = phase, f

    def log(s):
        job["log"].append(s)
    try:
        job["report"] = make_demo(people, video, out, thr, blur, panel, progress=progress, log=log)
        job["state"] = "done"
    except (Exception, SystemExit) as e:   # make_demo uses SystemExit for "no usable photos / unreadable video"
        job["error"] = str(e) or repr(e)
        job["state"] = "failed"
        traceback.print_exc()
    job["seconds"] = round(time.time() - job["t0"])


@app.before_request
def same_site_only():
    """Another website open in the browser must not be able to post to this local server."""
    origin = request.headers.get("Origin")
    if request.method == "POST" and origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
        abort(403)


@app.get("/")
def index():
    return PAGE.replace("__THR__", f"{load_threshold():.2f}")


@app.post("/api/start")
def start():
    with lock:
        if busy():
            return jsonify(error="A video is already being made. Wait for it to finish."), 409
        if request.form.get("consent") != "yes":
            return jsonify(error="Please confirm that everyone agreed to be named."), 400
        video = request.files.get("video")
        ext = os.path.splitext(video.filename)[1].lower() if video else ""
        if ext not in VIDEO_EXT:
            return jsonify(error="Please choose a video file (mp4, mov, mkv, avi or webm)."), 400
        try:
            thr = float(request.form.get("threshold", load_threshold()))
        except ValueError:
            return jsonify(error="Bad threshold."), 400

        jid = uuid.uuid4().hex[:10]
        root = os.path.join(WORK, jid)
        people = os.path.join(root, "people")
        os.makedirs(people)
        count, seen = 0, set()
        for i, raw in enumerate(request.form.getlist("names")):
            name = clean_name(raw)
            files = [f for f in request.files.getlist(f"photos_{i}") if os.path.splitext(f.filename)[1].lower() in IMG_EXT]
            if not name or not files:
                continue
            if name.lower() in seen:
                shutil.rmtree(root)
                return jsonify(error=f"The name '{name}' is used twice."), 400
            seen.add(name.lower())
            d = os.path.join(people, name)
            os.makedirs(d)
            for k, f in enumerate(files):
                f.save(os.path.join(d, f"{k}{os.path.splitext(f.filename)[1].lower()}"))
            count += 1
        if not count:
            shutil.rmtree(root)
            return jsonify(error="Add at least one person with a name and a photo."), 400
        vid, out = os.path.join(root, "input" + ext), os.path.join(root, "demo.mp4")
        video.save(vid)
        job = {"state": "running", "phase": "starting", "fraction": 0.0, "log": [], "report": None, "error": None,
               "t0": time.time(), "out": out}
        jobs[jid] = job
        threading.Thread(target=worker, daemon=True,
                         args=(job, people, vid, out, thr, request.form.get("blur") == "yes", request.form.get("panel") == "yes")).start()
    return jsonify(id=jid)


@app.get("/api/job/<jid>")
def status(jid):
    j = jobs.get(jid) or abort(404)
    return jsonify({k: j.get(k) for k in ("state", "phase", "fraction", "log", "report", "error", "seconds")})


@app.get("/api/job/<jid>/video")
def video(jid):
    j = jobs.get(jid) or abort(404)
    if j["state"] != "done":
        abort(404)
    return send_file(j["out"], mimetype="video/mp4", conditional=True, as_attachment="download" in request.args,
                     download_name="face-recognition-demo.mp4")


@app.post("/api/clear")
def clear():
    with lock:
        if busy():
            return jsonify(error="A video is still being made."), 409
        jobs.clear()
        shutil.rmtree(WORK, ignore_errors=True)
    return jsonify(ok=True)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Recognition Demo</title>
<style>
:root{--bg:#0f1412;--card:#181f1c;--line:#2a3430;--text:#eef2f0;--dim:#9aa8a2;--accent:#2ee6a6;--amber:#ffaa46;--red:#ff7a7a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:auto;padding:32px 16px 64px}h1{font-size:26px;margin:0}header p{color:var(--dim);margin:4px 0 24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}.card h2{font-size:15px;margin:0 0 12px}
.step{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--accent);color:#07130e;font-size:12px;font-weight:700;margin-right:8px}
.hint{color:var(--dim);font-size:13px;margin:0 0 12px}
.person{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 0;border-top:1px solid var(--line)}
input[type=text]{flex:1;min-width:130px;background:var(--bg);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px 10px;font:inherit}
.pick{background:var(--bg);border:1px dashed var(--line);border-radius:8px;padding:8px 10px;cursor:pointer;color:var(--dim);font-size:13px;white-space:nowrap}
.pick:hover{border-color:var(--accent)}.pick input{display:none}
.thumbs{display:flex;gap:4px;flex-wrap:wrap;width:100%}.thumbs img{width:44px;height:44px;object-fit:cover;border-radius:6px}
button{font:inherit;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--text);padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}.x{padding:4px 10px;color:var(--dim)}
.go{width:100%;background:var(--accent);border:0;color:#07130e;font-weight:700;padding:13px;font-size:16px;margin-top:16px}.go:disabled{opacity:.45;cursor:not-allowed}
label.row{display:flex;gap:10px;align-items:flex-start;margin:10px 0}label.row input{margin-top:4px;accent-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}.thr{display:flex;justify-content:space-between;color:var(--dim);font-size:13px}
.drop{display:block;border:1px dashed var(--line);border-radius:10px;padding:26px;text-align:center;color:var(--dim);cursor:pointer;overflow-wrap:anywhere}.drop:hover{border-color:var(--accent)}.drop input{display:none}
.bar{height:10px;background:var(--bg);border-radius:6px;overflow:hidden;margin:10px 0}.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s}
video{width:100%;border-radius:10px;background:#000;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-top:1px solid var(--line);overflow-wrap:anywhere}th{color:var(--dim);font-weight:500}
.named{color:var(--accent);font-weight:600}.unk{color:var(--amber)}.err{color:var(--red)}.wide{grid-column:1/-1}.tw{overflow-x:auto}
pre{background:var(--bg);border-radius:8px;padding:10px;font-size:12px;max-height:150px;overflow:auto;white-space:pre-wrap;margin:8px 0 0;color:var(--dim)}
.acts{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}a.btn{background:var(--accent);color:#07130e;font-weight:700;text-decoration:none;padding:8px 14px;border-radius:8px}
footer{margin-top:20px;color:var(--dim);font-size:13px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}
</style></head><body><main>
<header><h1>Face Recognition Demo</h1><p>Teach it a few faces, give it a video, and get the video back with names on the faces.</p></header>
<form id="f" class="grid">
<section class="card"><h2><span class="step">1</span>People to recognise</h2>
<p class="hint">3 to 5 clear photos per person work best, taken on different days. Only one main face in each photo.</p>
<div id="people"></div><button type="button" id="add">+ Add person</button></section>
<section class="card"><h2><span class="step">2</span>Your video</h2>
<label class="drop" id="vlabel"><input type="file" id="video" accept="video/*"><span id="vname">Click to choose a video</span></label>
<h2 style="margin-top:18px"><span class="step">3</span>Options</h2>
<div class="thr"><span>Match strictness</span><b id="thrv">__THR__</b></div>
<input type="range" id="thr" min="0.25" max="0.65" step="0.01" value="__THR__">
<div class="thr"><span>More names, more mistakes</span><span>Fewer names, safer</span></div>
<label class="row"><input type="checkbox" id="panel" checked><span>Show the people panel in the corner</span></label>
<label class="row"><input type="checkbox" id="blur"><span>Blur people who aren't enrolled (good for bystanders)</span></label>
<label class="row"><input type="checkbox" id="consent"><span>Everyone I added agreed to be named in this video</span></label>
<button class="go" id="go" type="submit">Make my video</button><p class="err" id="err" role="alert"></p></section>
</form>
<section class="card" id="run" hidden style="margin-top:16px"><h2 id="runt">Working</h2><div class="bar"><i id="bar"></i></div>
<p class="hint" id="runp"></p><pre id="log"></pre></section>
<section class="card" id="res" hidden style="margin-top:16px"><h2>Your video</h2><video id="out" controls></video>
<div class="acts"><a class="btn" id="dl">Download video</a></div><div class="tw"><table id="tbl"></table></div></section>
<footer><span>Runs only on this computer. Nothing is uploaded to the internet.</span><button type="button" id="clear">Delete everything</button></footer>
</main><script>
const $=s=>document.getElementById(s);let n=0;
function addPerson(){const i=n++,d=document.createElement('div');d.className='person';d.dataset.i=i;
 const name=document.createElement('input');name.type='text';name.placeholder='Full name, in English letters';name.setAttribute('aria-label','Person name');name.maxLength=60;
 const lab=document.createElement('label');lab.className='pick';const file=document.createElement('input');file.type='file';file.accept='image/*';file.multiple=true;
 const txt=document.createElement('span');txt.textContent='Add photos';lab.append(file,txt);
 const x=document.createElement('button');x.type='button';x.className='x';x.textContent='Remove';x.onclick=()=>d.remove();
 const th=document.createElement('div');th.className='thumbs';
 file.onchange=()=>{th.innerHTML='';txt.textContent=file.files.length+' photo(s)';for(const f of file.files){const m=new Image();m.src=URL.createObjectURL(f);m.alt='';th.append(m)}};
 d.append(name,lab,x,th);$('people').append(d)}
$('add').onclick=addPerson;addPerson();
$('thr').oninput=e=>$('thrv').textContent=(+e.target.value).toFixed(2);
$('video').onchange=e=>$('vname').textContent=e.target.files[0]?.name||'Click to choose a video';
const fmt=s=>{s=Math.round(s);return s<60?s+' s':Math.floor(s/60)+' min '+(s%60)+' s'};
$('f').onsubmit=async e=>{e.preventDefault();$('err').textContent='';const fd=new FormData();
 for(const[idx,d]of[...document.querySelectorAll('.person')].entries()){fd.append('names',d.querySelector('input[type=text]').value);
  for(const f of d.querySelector('input[type=file]').files)fd.append('photos_'+idx,f)}
 if($('video').files[0])fd.append('video',$('video').files[0]);
 fd.append('threshold',$('thr').value);fd.append('consent',$('consent').checked?'yes':'no');fd.append('blur',$('blur').checked?'yes':'no');fd.append('panel',$('panel').checked?'yes':'no');
 $('go').disabled=true;$('res').hidden=true;
 try{const r=await fetch('/api/start',{method:'POST',body:fd}),j=await r.json();if(!r.ok)throw new Error(j.error);poll(j.id,Date.now())}
 catch(x){$('err').textContent=x.message;$('go').disabled=false}};
async function poll(id,t0){$('run').hidden=false;
 const j=await(await fetch('/api/job/'+id)).json();
 const pct=j.phase==='analyse'?j.fraction*70:j.phase==='render'?70+j.fraction*30:2;
 $('bar').style.width=(j.state==='done'?100:pct)+'%';$('log').textContent=j.log.join('\n');$('log').scrollTop=1e6;
 const sec=(Date.now()-t0)/1000;$('runt').textContent=j.phase==='analyse'?'Finding and recognising faces':j.phase==='render'?'Drawing names and saving the video':'Learning the faces';
 $('runp').textContent=fmt(sec)+' so far'+(pct>8?', about '+fmt(sec*(100-pct)/pct)+' left':'')+'. Keep this page open.';
 if(j.state==='running')return setTimeout(()=>poll(id,t0),1000);
 $('go').disabled=false;
 if(j.state==='failed'){$('runt').textContent='It did not work';$('runp').textContent=j.error;$('runp').className='err';return}
 $('run').hidden=true;$('res').hidden=false;$('out').src='/api/job/'+id+'/video';$('dl').href='/api/job/'+id+'/video?download=1';
 const t=$('tbl');t.innerHTML='<tr><th>When</th><th>Shown as</th><th>Closest match</th><th>Score</th></tr>';
 for(const r of j.report){const tr=t.insertRow();tr.insertCell().textContent=r.start_s.toFixed(1)+' to '+r.end_s.toFixed(1)+' s';
  const c=tr.insertCell();c.textContent=r.name;c.className=r.name==='Unknown'?'unk':'named';tr.insertCell().textContent=r.closest||'none';tr.insertCell().textContent=r.score.toFixed(2)}}
let armed=false;$('clear').onclick=async e=>{if(!armed){armed=true;e.target.textContent='Click again to delete all uploads and videos';setTimeout(()=>{armed=false;e.target.textContent='Delete everything'},4000);return}
 const r=await fetch('/api/clear',{method:'POST'}),j=await r.json();armed=false;e.target.textContent=r.ok?'Deleted':j.error;$('res').hidden=true;$('out').removeAttribute('src')};
</script></body></html>"""


if __name__ == "__main__":
    print(f"open http://127.0.0.1:{PORT}")
    app.run("127.0.0.1", PORT, threaded=True)
