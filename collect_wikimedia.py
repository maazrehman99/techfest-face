"""Collect CC/public-domain photos of Pakistani public figures (Wikimedia) with licence records.

commons.wikimedia.org is unreachable from our network, so: Wikidata (people) -> en.wikipedia article images ->
imageinfo (licence, author, thumbnail URL via the shared Commons repo) -> upload.wikimedia.org download.

  python collect_wikimedia.py --max-people 600 --out data/pk_wiki
Output: <out>/raw/<QID>/<n>.jpg, <out>/records.jsonl (one row per image: person, gender, licence, author, source page).
Only public figures with a Wikipedia article; only freely licensed images. Resumable.
"""
import argparse, json, os, re, time, requests
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "techfest-face-research/0.1 (university competition, non-commercial)"}
WD = "https://query.wikidata.org/sparql"
EN = "https://en.wikipedia.org/w/api.php"
BAD = re.compile(r"logo|signature|flag|map|coat|seal|stamp|poster|cover|screenshot|icon|symbol|emblem|banner|\bstadium|building|city|road|mosque", re.I)
FREE = re.compile(r"^(CC[ -]|Public domain|PD|Attribution|GFDL)", re.I)


def get(url, **kw):
    for k in range(4):
        try:
            r = requests.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=40, **kw)
            if r.status_code == 429:
                time.sleep(5 * (k + 1)); continue
            r.raise_for_status(); return r
        except requests.RequestException:
            time.sleep(2 * (k + 1))
    return None


def people(n):
    q = """SELECT ?p ?pLabel ?genderLabel ?born ?title (COUNT(?x) AS ?links) WHERE {
      ?p wdt:P31 wd:Q5; wdt:P27 wd:Q843; wikibase:sitelinks ?links; wdt:P569 ?born.
      FILTER(YEAR(?born) >= 1947)
      FILTER NOT EXISTS { ?p wdt:P27 wd:Q902 }          # also a Bangladeshi citizen (East Pakistan era)
      OPTIONAL { ?p wdt:P19 ?bp . ?bp wdt:P17 ?bc }
      FILTER(!BOUND(?bc) || ?bc = wd:Q843)               # born in Pakistan (or birthplace unknown)
      OPTIONAL { ?p wdt:P21 ?g . ?g rdfs:label ?genderLabel FILTER(lang(?genderLabel)='en') }
      ?a schema:about ?p; schema:isPartOf <https://en.wikipedia.org/>; schema:name ?title.
      SERVICE wikibase:label { bd:serviceParam wikibase:language 'en'. }
    } GROUP BY ?p ?pLabel ?genderLabel ?born ?title ?links ORDER BY DESC(?links) LIMIT %d""" % n
    r = get(WD, params={"query": q, "format": "json"}, headers={"Accept": "application/sparql-results+json"})
    out = []
    for b in r.json()["results"]["bindings"]:
        out.append({"qid": b["p"]["value"].rsplit("/", 1)[1], "name": b["pLabel"]["value"], "title": b["title"]["value"],
                    "gender": b.get("genderLabel", {}).get("value", "unknown"), "born": b["born"]["value"][:4]})
    return out


WIKIS = ["en", "ur", "pnb", "sd", "hi", "ar", "fa", "de", "fr", "es", "ru", "it", "pt", "zh", "ja", "id", "tr", "bn", "nl", "sv"]


def sitelinks(batch):
    """qid -> {wiki code: article title} via the Wikidata API."""
    r = get("https://www.wikidata.org/w/api.php", params={"action": "wbgetentities", "ids": "|".join(p["qid"] for p in batch),
                                                           "props": "sitelinks", "format": "json"})
    j = r.json() if r is not None else {}
    return {q: {w: e["sitelinks"][w + "wiki"]["title"] for w in WIKIS if w + "wiki" in e.get("sitelinks", {})}
            for q, e in j.get("entities", {}).items()}


def article_images(wiki, titles):
    """-> {article title: [canonical 'File:...' titles]} for one wiki (titles are localized there, so canonicalise)."""
    res, params = {}, {"action": "query", "prop": "images", "imlimit": "max", "titles": "|".join(titles),
                       "format": "json", "redirects": 1}
    while True:
        r = get(f"https://{wiki}.wikipedia.org/w/api.php", params=params)
        if r is None:
            break
        j = r.json()
        for pg in j.get("query", {}).get("pages", {}).values():
            res.setdefault(pg["title"], []).extend("File:" + i["title"].split(":", 1)[1] for i in pg.get("images", []) if ":" in i["title"])
        if "continue" not in j:
            break
        params.update(j["continue"])
    return res


def imageinfo(files):
    out = {}
    for i in range(0, len(files), 40):
        r = get(EN, params={"action": "query", "prop": "imageinfo", "titles": "|".join(files[i:i + 40]), "format": "json",
                            "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 640})
        if r is None:   # transient failure after retries: skip this chunk instead of crashing the whole run
            continue
        for pg in r.json().get("query", {}).get("pages", {}).values():
            ii = (pg.get("imageinfo") or [None])[0]
            if ii and ii.get("mime") == "image/jpeg" and ii.get("thumburl"):
                em = ii.get("extmetadata", {})
                out[pg["title"]] = {"thumb": ii["thumburl"], "page": ii["descriptionurl"], "w": ii["width"], "h": ii["height"],
                                    "license": em.get("LicenseShortName", {}).get("value", ""),
                                    "author": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", ""))[:200]}
        time.sleep(0.3)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-people", type=int, default=600); ap.add_argument("--out", default="data/pk_wiki")
    ap.add_argument("--max-per-person", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = people(a.max_people); print(len(P), "people from Wikidata", flush=True)
    recs, seen, start = [], set(), 0
    part, prog = os.path.join(a.out, "records_partial.jsonl"), os.path.join(a.out, "progress.txt")
    if os.path.exists(part) and os.path.exists(prog):   # resume after a crash
        recs = [json.loads(l) for l in open(part, encoding="utf-8")]
        seen = {(r["qid"], r["file"]) for r in recs}
        start = int(open(prog).read().strip())
        print(f"resuming at person {start} with {len(recs)} saved images", flush=True)
    for i in range(start, len(P), 25):
        batch = P[i:i + 25]
        n0 = len(recs)
        links = sitelinks(batch)
        byq = {p["qid"]: p for p in batch}
        files = {q: set() for q in byq}   # qid -> candidate files from every language edition
        def one(w):
            t2q = {links[q][w]: q for q in byq if q in links and w in links[q]}
            return t2q, (article_images(w, list(t2q)) if t2q else {})
        with ThreadPoolExecutor(10) as ex:
            for t2q, found in ex.map(one, WIKIS):
                for t, fs in found.items():
                    q = t2q.get(t)
                    if q:
                        files[q] |= {f for f in fs if f.lower().endswith((".jpg", ".jpeg")) and not BAD.search(f)}
        info = imageinfo(sorted(set().union(*files.values())))
        for q, fs in files.items():
            p = byq[q]
            for f in sorted(fs):
                m = info.get(f)
                if m and FREE.match(m["license"]) and min(m["w"], m["h"]) >= 200 and (q, f) not in seen:
                    seen.add((q, f)); recs.append({**p, "file": f, **m})
        with open(part, "a", encoding="utf-8") as f:
            for r in recs[n0:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        open(prog, "w").write(str(i + len(batch)))
        print(f"{i + len(batch)}/{len(P)} people, {len(recs)} free images", flush=True)
    per, keep = {}, []
    for r in recs:
        per[r["qid"]] = per.get(r["qid"], 0) + 1
        if per[r["qid"]] <= a.max_per_person:
            keep.append(r)
    keep = [r for r in keep if per[r["qid"]] >= 3]  # an identity needs several photos
    print(len(keep), "images for", len({r["qid"] for r in keep}), "people with >=3 free images", flush=True)

    def dl(ir):
        k, r = ir
        d = os.path.join(a.out, "raw", r["qid"]); os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, f"{k}.jpg"); r["path"] = os.path.relpath(fp, a.out)
        if not os.path.exists(fp):
            x = get(r["thumb"])
            if x is None:
                r["path"] = None; return
            open(fp, "wb").write(x.content); time.sleep(0.15)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(dl, enumerate(keep)))
    with open(os.path.join(a.out, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in keep:
            if r.get("path"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done", sum(1 for r in keep if r.get("path")), "images downloaded", flush=True)


if __name__ == "__main__":
    main()
