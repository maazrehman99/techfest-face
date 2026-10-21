# NEXT_STEPS: status and how to continue (updated 2026-10-21 ~16:30, owner left the lab PC)

Read `CLAUDE.md` (brief) and `HISTORY.md` (what happened) first. This file is the hand-over: where we stopped and the
exact commands to continue. **Report due 2026-10-22, hidden benchmark 2026-10-23.**

## Decision taken: A + B (2026-10-21)
The Wikimedia re-run (1500 people) left most people with only ONE usable photo (~90 usable images), too few to train on.
- **B:** halfFace is split BY PERSON into train / val / held-out test (60/15/25%, gender-stratified, seed 0) by
  `build_ft_data.py`. Training data = halfFace train people + Wikimedia train crops. The baseline MUST be re-measured on
  the same held-out halfFace people (the old 126-person numbers are not comparable).
- **A:** Wikimedia = small real-Pakistani test set. The newer 1500-person run lives on a teammate's PC: get the whole
  `data/pk_wiki` folder (incl. `raw/`, `records.jsonl`), then loosen `process_wikimedia.py`
  (`--min-images 2 --dominance 1.5 --admit 0.35`) and re-check the contact sheets.

## State of the lab PC (Windows 11, RTX 5070 Ti 16 GB, driver 596, 20 threads, 32 GB)
Project at `C:\Users\User\Desktop\techfest\techfest-face`.
- `.venv` (Python 3.12.10) has everything: requirements.txt + torch 2.8.0+cu128 + torchvision 0.23.0+cu128 + onnx2torch.
  **GPU verified** (torch sees the 5070 Ti, sm_120). The torch wheels are kept in `C:\Users\User\Desktop\techfest\wheels\`.
- `models/` downloaded (buffalo_l + `w600k_r50.onnx`).
- **halfFace zip: 593 of 659 MB downloaded, in pieces** `data/halfface/halfface.zip.part0..7` (not finished, not unzipped).
- Tested on this PC: `run.py verify/pairs` (same / different person, blank image, missing file -> no_face, no crash),
  Urdu folder/person names (fixed, see HISTORY), `gallery.py` enroll/identify with an Urdu name,
  and a 2-epoch GPU smoke run of `finetune.py` (trains, exports ONNX, export loads in the pipeline).
- Not done yet: anything that needs the halfFace images.

## Network note (important on the lab PC)
Internet is very slow and unstable (50 KB/s to 1 MB/s, Ethernet and Wi-Fi alternate being the faster one).
`tools/dl_parts.sh` (Git Bash) downloads in parallel pieces over both connections, resumes, and switches connection
when a piece stalls. Kill it by killing every process whose command line contains `dl_parts.sh`, then `curl.exe`
(stopping only the outer shell leaves the loops running).

## Continue here (in order)
```
cd C:\Users\User\Desktop\techfest\techfest-face
# 1. finish halfFace (Git Bash; the IPs are this PC's Wi-Fi and Ethernet addresses, check with ipconfig)
bash tools/dl_parts.sh 192.168.137.174,172.16.9.71 "https://data.mendeley.com/public-api/zip/pk44mkx9vm/download/2" data/halfface/halfface.zip 691391715 8
#    -> prints "DONE ... 691391715 bytes"; then unzip so the paths in data/halfface/manifest.csv exist:
#    data/halfface/extracted/halfFace_A face covering mask dataset of South Asian people/Dataset1_Full Facial Image/pXXX/...
#    (check the zip's inner folder layout; the zip may contain a nested zip)
# 2. person split + aligned crops + combined manifest
.venv\Scripts\python.exe build_ft_data.py
# 3. "before": baseline on the held-out halfFace people
.venv\Scripts\python.exe evaluate.py --manifest data/halfface/manifest_ft.csv --root data/halfface --out results/halfface_ft_baseline --ablation
# 4. fine-tune on the GPU (minutes)
.venv\Scripts\python.exe finetune.py --manifest data/ft_manifest.csv --root data --out runs/ft1 --allow-halfface --workers 0
# 5. "after": same held-out people, fine-tuned recognizer
.venv\Scripts\python.exe evaluate.py --manifest data/halfface/manifest_ft.csv --root data/halfface --out results/halfface_ft_finetuned --ablation --rec runs/ft1/finetuned.onnx
# 6. keep the better model: if fine-tuned wins, set "rec_onnx": "runs/ft1/finetuned.onnx" in config.json
#    (and re-pick "threshold" from results/halfface_ft_finetuned/results.json thr_val_FAR1e-2), then gallery.py reindex
```
Then fill the [TBD] parts of `report/REPORT_DRAFT.md` (sections 1, 4, 5, 6.2, 6.6). Honesty rules: if no epoch beats
the baseline, say so and ship the baseline.

## Still open
- Send the organizer questions in `REQUESTS_TO_SEND.md` (hidden-benchmark input format decides how `run.py` is packaged).
- Option A (Pakistani test set) needs the teammate's newer `data/pk_wiki` folder.
- `identify_threshold` (0.40 placeholder) and unknown-person rejection; real webcam test of `live.py`.
- `runs/` and `models/` are git-ignored: the fine-tuned ONNX (~170 MB) must be shared outside git (USB / drive).
- After the competition: delete data, models and galleries (target 2026-11-30).
