# Data sources, licences and use

This repository contains no face images. It holds only split lists (which person is in which split), so the results can be checked against the same people. Each dataset has to be downloaded from its own source.

## halfFace (evaluation and fine-tuning)

"halfFace: A face covering mask dataset of South Asian people", Mendeley Data pk44mkx9vm, CC BY 4.0. 938 photos of 166 people, masked and unmasked. Split by person 60/15/25 (`data/halfface/manifest_ft.csv`). After downloading, run `python make_halfface_manifest.py`.

## IMFDB (fine-tuning and evaluation)

Indian Movie Face Database, IIIT Hyderabad: S. Setty et al., "Indian Movie Face Database: A Benchmark for Face Recognition Under Wide Variations", NCVPRIPG 2013. Official page: https://cvit.iiit.ac.in/projects/IMFDB/ . The site states no licence, so the data is used for research only, cited, and not redistributed. `build_imfdb.py` keeps 40 photos per actor (98 actors) and splits by actor (`data/imfdb/manifest_eval.csv`).

## Pakistani public figures (test only)

Freely licensed images (CC BY, CC BY-SA, public domain) from Wikipedia articles about Pakistani public figures, found through Wikidata (`collect_wikimedia.py`) and cleaned so each photo shows the named person (`process_wikimedia.py`). Only adults; people listed in `exclude_people.txt` are left out.

A free licence allows reuse of a photo, but it is not the person's consent to face recognition research. For that reason these photos are not published in this repository and were not used to train the GPU model. They served only as an unseen test.

## Pretrained models

InsightFace `buffalo_l` (SCRFD-10G detector, ArcFace ResNet-50 trained on WebFace600K). The weights are licensed for non-commercial research only and are not included here.

## Rules we followed

1. Splits are by person, never by photo. No test person was used for training or for choosing a threshold.
2. Face memory (`gallery.py`) enrols a person only with the `--consent` flag, stores a consent record, and `gallery.py remove` erases them.
3. The system is not to be used to identify private individuals or to build a watch-list.

## Known limits of the data

Public figures skew male, urban and well known, so they do not represent all Pakistanis. halfFace is South Asian and IMFDB is Indian; neither is Pakistani-specific. About 90% of halfFace people are men.
