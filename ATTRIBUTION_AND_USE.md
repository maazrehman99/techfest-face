# Data attribution, licences and permitted use

## Pakistani public-figure photos (`data/pk_wiki`)
- Source: Wikimedia Commons images used in Wikipedia articles of Pakistani public figures (people found via Wikidata).
- Only images with a free licence were kept (CC BY, CC BY-SA, other CC, public domain). **Per-image licence, author and source page are in `data/pk_wiki/manifest.csv` / `records.jsonl`.** Credit the author named there when showing an image. CC BY-SA images require adaptations (our aligned crops) to carry the same licence.
- Only adults (born 2005 or earlier). People on `exclude_people.txt` (sensitive cases) are left out.
- These people were not asked about face-recognition research; the licence allows reuse of the photo, it is not their consent. Therefore:

## Permitted use (read before touching the data)
1. Research for the Nemo's Deep Dive competition only. No commercial use, no product, no public release of the dataset or any gallery.
2. Keep the repository **private**; share only with team members. Do not make it public without re-checking every licence.
3. Never use the data or the model to identify private individuals or to build a watch-list.
4. Keep copies only on team machines; **delete the data, the models trained on it and any galleries after the competition** (target: by 2026-11-30).
5. Any person enrolled in a face gallery (`gallery.py`) must have given real, written consent; they can ask for erasure (`gallery.py remove`).
6. The ArcFace pretrained weights (InsightFace `buffalo_l`) are non-commercial research only.

## halfFace (benchmark)
CC BY 4.0, "halfFace: A face covering mask dataset of South Asian people", Mendeley Data pk44mkx9vm. Evaluation only; not redistributed in this repo (only its file list).

## IMFDB (added 2026-10-22)
Indian Movie Face Database, IIIT Hyderabad (S. Setty et al., "Indian Movie Face Database: A Benchmark for Face Recognition Under Wide Variations", NCVPRIPG 2013). Downloaded from the official page https://cvit.iiit.ac.in/projects/IMFDB/ . **The site states no licence or terms.** These are film frames of real actors: used for research training/evaluation only, kept local, not redistributed (only the person split list is in git), cited in the report, to be deleted after the competition. Data is Indian, a South Asian proxy.

## Possible extra test photos (not collected yet)
Hand-picked photos of Pakistani public figures for a test-only set (`data/pk_test/`, git-ignored): keep the source link of every photo; Creative Commons sources (Openverse, Flickr, Wikimedia) preferred; no bulk download from Google Images (unlicensed, against its terms). Never trained on.

## Limits to state in the report
Public figures skew male, urban and elite; results do not represent all Pakistanis. halfFace is South Asian, not Pakistani-specific.
