# TechFest Face Recognition Challenge (Pakistani context)

Nemo's Deep Dive, Round 2 "Every Fish Has a Face" (NED CSIT Tech Fest '26).
**Report due 2026-10-22. Hidden benchmark 2026-10-23.**

Pipeline: detect -> align -> quality filter -> embed (InsightFace ArcFace w600k_r50) -> match + threshold.
Must run offline on CPU, never crash on a no-face image.

## Setup
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
Models (`models/`) and datasets (`data/`) are NOT in git (651MB / 2GB). Get buffalo_l from InsightFace
(it auto-downloads to ~/.insightface/models) and copy to `models/`. halfFace: Mendeley pk44mkx9vm (CC BY 4.0).
Then `python make_halfface_manifest.py`.

## Files
- `pipeline.py` pipeline with toggleable stages; `evalkit.py` / `evaluate.py` metrics (acc/P/R/F1/ROC/FAR/FRR/EER), ablation, per-group
- `run.py` organizer entry point (`verify`, `embed`, `pairs`); threshold in `config.json`
- `build_ft_data.py` halfFace person split + aligned crops + combined fine-tune manifest (decision A+B); `finetune.py` GPU fine-tune
- `report/REPORT_DRAFT.md` report draft; `tools/dl_parts.sh` resumable parallel downloader for slow networks
- `build_imfdb.py` adds IMFDB to the training list; `tools/eval_crops.py`, `tools/boot_ci.py` compare original vs fine-tuned (GPU, bootstrap); `tools/eval_pk_test.py` small Pakistani test; `tools/gcp_finetune.sh` cloud GPU job; `tools/ntfy*.sh` phone alerts; `collect_openverse.py` (parked)
- **All numbers in plain words: `report/RESULTS_FOR_REPORT.md`; raw results `results/ft_gpu_run/`**
- **Where we are and how to continue: `NEXT_STEPS.md`**
- `augment.py` synthetic masks/hijab/cap/glasses/lighting/blur/CCTV
- `DECISIONS.md` every decision + reasoning (judges care about this: keep it updated)
- `REQUESTS_TO_SEND.md` questions for organizers; BUPT dataset request
- `memory/` Claude Code project notes (context for whoever picks this up)

## Run the baseline eval (CPU, ~1 img/s per 12 cores)
```
FACE_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python evaluate.py \
  --manifest data/halfface/manifest.csv --out results/halfface --ablation
```

## Team tasks (pick one, tell the group, commit on a branch)
| # | Task | Output |
|---|------|--------|
| 1 | Send organizer questions + BUPT request (`REQUESTS_TO_SEND.md`) | replies saved in `DECISIONS.md` |
| 2 | Run halfFace baseline eval, then LFW sanity | `results/halfface/`, numbers for report |
| 3 | Training data: IMFDB / BUPT-Indian / whatever organizers allow; consent notes | `data/train/` manifest + `DECISIONS.md` entry |
| 4 | Colab fine-tune notebook (ArcFace head on our data, export ONNX) | `finetune.ipynb`, new `.onnx` |
| 5 | Pick threshold from EER/FAR target, write `config.json`, test `run.py` on a clean venv + no-face images | `config.json`, test log |
| 6 | Report + slides (honest: data is South Asian proxy, not Pakistani-specific) | `report/` |

Order that matters: 2 -> 5 can start now; 3 -> 4 is the critical path (no GPU locally, use Colab/Kaggle).

## Face memory (enroll once, recognise forever, no retraining)
The embedding model is frozen; people are "remembered" as embeddings in `gallery/`. Every enrolment needs `--consent`.
```
python gallery.py enroll --name "Ali Khan" --images photos/ali --consent
python gallery.py identify --images group.jpg --save out/
python gallery.py scan --input cctv.mp4 --out appearances.csv --every 1
python gallery.py remove --name "Ali Khan"        # erase a person
python live.py --source 0                          # browser UI at http://127.0.0.1:8000 (enroll + live labels)
python gallery.py reindex                          # after swapping in the fine-tuned recognizer
```
`identify_threshold` in config.json (default 0.40) must be tuned on the Pakistani eval set. Quick check on halfFace
(3 unmasked enrolment photos/person, 15 people): 33/37 masked probes correct, 4 Unknown, 0 wrong.
