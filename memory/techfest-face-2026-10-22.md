---
name: techfest-face-2026-10-22
description: CURRENT state of the Techfest face project (2026-10-22): what we decided, what we measured, what is left. Supersedes the 2026-10-19 snapshot in techfest-face.md
metadata:
  type: project
---

Competition: NED CSIT Tech Fest '26, "Every Fish Has a Face" (Round 2). Report due 2026-10-22, hidden benchmark 2026-10-23. NED said on 2026-10-21 that the consent-process item in the instructions was a mistake and is not required; everything else is unchanged.

**Model (our deliberate choice, D1/D2, approved 2026-10-18):** InsightFace `buffalo_l` = SCRFD-10G detector + ArcFace ResNet-50 (`w600k_r50`) recogniser, cosine similarity. AdaFace was never tested. Weights are non-commercial research only; cite them.

**Decision D13 (2026-10-22): the ORIGINAL model ships** (`config.json` `rec_onnx` = null). The fine-tune (GPU, 2,919 photos / 159 people) gave a small mixed result and is reported honestly.

Key numbers (full list in `report/RESULTS_FOR_REPORT.md`, raw in `results/ft_gpu_run/`):
- Original model, new held-out halfFace people (40): accuracy 97.1%, EER 2.9%, masked 94.3%, unmasked 98.2%. IMFDB held-out actors (24): accuracy 87.5%, EER 10.5%.
- Alignment ablation: without alignment EER 25.4% (essential); quality filter changes nothing on halfFace; flip averaging helps slightly (2.3%).
- Fine-tuned: halfFace EER 3.9% (worse), IMFDB 10.0% (slightly better), masked halfFace better; bootstrap says the cropped-face gain (-1.1 points) is marginal and absent on clean halfFace.
- Wikimedia Pakistani check (13 people, 48 unseen photos): 99.2% (original) vs 99.4% (fine-tuned); too easy to separate the models.

**Why:** the judges score pipeline quality, evaluation rigour and honesty, not a new backbone; the data (no public Pakistani identity set) is the real limit.
**How to apply:** do not retrain with this data; spend the remaining time on the offline `run.py` dry run, sending `REQUESTS_TO_SEND.md`, threshold re-pick, report, and an optional hand-picked Pakistani test set with coverings (`tools/eval_pk_test.py`). Rules in `CLAUDE.md` section 8 still apply (never train on test people, keep faces local, no bulk Google Images scraping, drive any cloud VM by hand and delete it only after results are verified).

**Update (evening 2026-10-22): the real benchmark format is the organizers' `student_pack` (identification, not verify).** `submission/make_submission.py <group> <data_dir>` embeds every image in `train/` and `test/` and writes `<group>.npz`; score = top-1 cosine match of each test image to the same person's train image (top-5 breaks ties). Practice pack (7 Pakistani public figures, one photo each): original pipeline 4/7 top-1; now 6/7 top-1, 7/7 top-5. Two changes: (1) embedding = average of ArcFace R50 (`w600k_r50`) and R100 (`models/glintr100.onnx`, from InsightFace `antelopev2`; git-ignored, copy it by hand; non-commercial licence like the rest) giving 1024-d vectors; on IMFDB held-out actors (24, one gallery photo each, `tools/id_bench.py`) top-1 is 75.5% R50, 77.6% R100, 78.1% average; (2) for test images with several faces the script picks the face closest to a train face (faces under 15% of the largest are ignored). The practice gain is mostly from (2) and rests on only 7 people. `SUB_R100=0` / `SUB_MULTI=0` switch them off. Speed: about 4 s per face for R100 on this laptop CPU, so a big pack is slow; `tools/vm_eval.sh` (setup / run / stop, never deletes) runs it on the GCP VM `maaz-dev-vm` (project `r2r-coros-dev`, asia-south1-b, 1x L4), CPU by default because onnxruntime-gpu crashed on CUDA 12.9 earlier. Whether cloud compute is allowed on lab day is unconfirmed. Lab day: `python submission/make_submission.py group10 <folder with train/ and test/>`, upload `group10.npz`.
