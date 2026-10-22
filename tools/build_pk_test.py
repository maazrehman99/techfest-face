"""Build the Pakistani TEST set candidates (protocol: report/PK_TEST_PROTOCOL.md). Uses NO recognition scores.
Source: free-licence Wikipedia article images already listed in data/pk_wiki/records_partial.jsonl.
Per person (not one of the pk_wiki people): keep images whose file name contains the person's name (identity anchor),
download the 640px thumbnail, keep those with ONE dominant face (face DETECTOR only), skip near-duplicates (tiny-image
hash), take 2. Writes data/pk_test_cand/<Name>/{1,2}.jpg, sources.csv, and contact sheets to eyeball."""
import csv, json, os, re, sys, collections, time
import cv2, numpy as np, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import Config, FacePipeline, detect_all
UA = {"User-Agent": "techfest-face-research/0.1 (university competition, non-commercial)"}
OUT = "data/pk_test_cand"; os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
rs = [json.loads(l) for l in open("data/pk_wiki/records_partial.jsonl", encoding="utf-8")]
used = {r["identity"] for r in csv.DictReader(open("data/pk_wiki/manifest.csv", encoding="utf-8"))}
by = collections.defaultdict(list)
for r in rs: by[r["qid"]].append(r)
pipe = FacePipeline(Config())
CACHE = os.path.join(OUT, "_dl"); os.makedirs(CACHE, exist_ok=True)
def get(u):
    cp = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", u.split("?")[0])[-150:] + ".jpg")
    if os.path.exists(cp): return open(cp, "rb").read()
    for k in range(6):
        try:
            r = requests.get(u, headers=UA, timeout=40)
            if r.status_code == 429:   # server asked us to slow down: wait as long as it says
                w = min(int(r.headers.get("retry-after", 60)), 900) + 5; print("429, waiting", w, "s", flush=True); time.sleep(w); continue
            r.raise_for_status(); time.sleep(1.5); open(cp, "wb").write(r.content); return r.content
        except requests.RequestException: time.sleep(5 * (k + 1))
def dominant(img):
    _, fs = detect_all(pipe, img, pad_retry=True)
    if not fs: return False
    a = sorted(((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in fs), reverse=True)
    w = int(fs[0]["bbox"][2] - fs[0]["bbox"][0])
    return (len(a) == 1 or a[1] < 0.45 * a[0]) and max(f["bbox"][2] - f["bbox"][0] for f in fs) >= 70
def tiny(img): return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16)).astype(float)
rows, sheet = [], []
for q, v in by.items():
    if q in used: continue
    tok = [norm(t) for t in v[0]["name"].split() if len(t) > 2]
    anc = [r for r in v if any(t in norm(r["file"]) for t in tok)]
    if len(anc) < 2: continue
    good = []
    for r in anc:
        lp = os.path.join("data/pk_wiki", r.get("path", "none")); b = open(lp, "rb").read() if os.path.exists(lp) else get(r["thumb"])
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
        if img is None or not dominant(img): continue
        t = tiny(img)
        if any(np.abs(t - g[2]).mean() < 6 for g in good): continue   # near-duplicate of a kept photo
        good.append((r, img, t))
        if len(good) == 2: break
    if len(good) < 2: continue
    d = os.path.join(OUT, re.sub(r"[^A-Za-z0-9]+", "_", v[0]["name"]).strip("_")); os.makedirs(d, exist_ok=True)
    for k, (r, img, _) in enumerate(good, 1):
        cv2.imwrite(f"{d}/{k}.jpg", img)
        rows.append([f"{os.path.basename(d)}/{k}.jpg", v[0]["name"], v[0]["gender"], r["page"], r["license"], r["author"].strip().replace("\n", " ")])
    sheet.append((os.path.basename(d), [cv2.resize(g[1], (200, 240)) for g in good]))
    print(len(sheet), v[0]["name"], flush=True)
with open(f"{OUT}/sources.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file", "person", "gender", "page_url", "licence", "author"]); w.writerows(rows)
for s in range(0, len(sheet), 12):
    cells = []
    for name, ims in sheet[s:s + 12]:
        c = np.full((270, 410, 3), 255, np.uint8); c[30:270, 0:200] = ims[0]; c[30:270, 205:405] = ims[1]
        cv2.putText(c, name[:34], (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1); cells.append(c)
    while len(cells) % 3: cells.append(np.full((270, 410, 3), 255, np.uint8))
    cv2.imwrite(f"{OUT}/sheet_{s // 12 + 1}.jpg", np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]))
print("people", len(sheet))
