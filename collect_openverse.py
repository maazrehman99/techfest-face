"""Collect a small TEST-ONLY set of Pakistani public figures (with coverings) from Creative Commons sources via Openverse.

  python collect_openverse.py            # writes data/pk_test/<Name>/<n>.jpg and data/pk_test/records.jsonl

Why not Google Images: it is unlicensed and against Google's terms. Openverse indexes Flickr + Wikimedia and returns the
licence, creator and source link of every image, so each photo can be credited in the report (CC BY / BY-SA / CC0 / PDM).
Label trust: an image is kept only if the person's name appears in its title or tags (the "anchor" rule from
process_wikimedia.py); the photos are then checked by eye on contact sheets. Test only: never train on these photos.
"""
import json, os, re, time, argparse
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pk_test")
API = "https://api.openverse.org/v1/images/"
UA = {"User-Agent": "techfest-face-research/0.1 (student competition, test-only set; contact via GitHub maazrehman99)"}

# name -> expected covering/trait (a hint only: the real tag is set after looking at each photo)
PEOPLE = {
    "Malala Yousafzai": "dupatta", "Benazir Bhutto": "dupatta", "Maryam Nawaz": "dupatta", "Shireen Mazari": "dupatta",
    "Sherry Rehman": "dupatta", "Hina Rabbani Khar": "dupatta", "Asma Jahangir": "dupatta", "Mahira Khan": "none",
    "Sajal Aly": "none", "Yumna Zaidi": "none", "Ayeza Khan": "none", "Bushra Ansari": "none",
    "Babar Azam": "cap", "Shaheen Afridi": "cap", "Mohammad Rizwan": "cap", "Shadab Khan": "cap", "Wasim Akram": "cap",
    "Shahid Afridi": "beard", "Fazal-ur-Rehman": "beard", "Tahir-ul-Qadri": "beard", "Siraj-ul-Haq": "beard",
    "Asif Ali Zardari": "topi", "Junaid Jamshed": "beard", "Tariq Jameel": "beard", "Abdul Sattar Edhi": "beard",
    "Imran Khan": "none", "Nawaz Sharif": "none", "Bilawal Bhutto Zardari": "none", "Fawad Khan": "none",
}


def get(url, **kw):
    for i in range(3):
        r = requests.get(url, headers=UA, timeout=30, **kw)
        if r.status_code == 429:
            print("  rate limited, waiting 30 s"); time.sleep(30); continue
        return r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-person", type=int, default=10)
    ap.add_argument("--min-side", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec_path = os.path.join(OUT, "records.jsonl")
    seen = set()
    if os.path.exists(rec_path):
        seen = {json.loads(l)["url"] for l in open(rec_path, encoding="utf-8")}
    rec = open(rec_path, "a", encoding="utf-8")
    for name, hint in PEOPLE.items():
        surname = re.sub(r"[^a-z ]", " ", name.lower()).split()[-1]
        first = re.sub(r"[^a-z ]", " ", name.lower()).split()[0]
        r = get(API, params={"q": name, "license": "by,by-sa,cc0,pdm", "page_size": 40})
        if r.status_code != 200:
            print(name, "search failed", r.status_code); continue
        folder = os.path.join(OUT, name.replace(" ", "_")); os.makedirs(folder, exist_ok=True)
        n = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        for it in r.json().get("results", []):
            if n >= a.per_person:
                break
            text = ((it.get("title") or "") + " " + " ".join(t.get("name", "") for t in it.get("tags") or [])).lower()
            if not (first in text and surname in text):          # label trust: full name must appear
                continue
            if it["url"] in seen or min(it.get("width") or 0, it.get("height") or 0) < a.min_side:
                continue
            try:
                img = get(it["url"])
                if img.status_code != 200 or len(img.content) < 5000 or "image" not in img.headers.get("content-type", ""):
                    continue
            except Exception as e:
                print("  download failed", str(e)[:60]); continue
            n += 1; seen.add(it["url"])
            fn = f"{n:02d}.jpg"
            open(os.path.join(folder, fn), "wb").write(img.content)
            rec.write(json.dumps({"file": f"{name.replace(' ', '_')}/{fn}", "person": name, "hint": hint, "url": it["url"],
                                  "page": it.get("foreign_landing_url"), "title": it.get("title"), "license": it.get("license"),
                                  "license_version": it.get("license_version"), "creator": it.get("creator"),
                                  "source": it.get("source")}, ensure_ascii=False) + "\n"); rec.flush()
        print(f"{name:26s} kept {n}")
        time.sleep(1)


if __name__ == "__main__":
    main()
