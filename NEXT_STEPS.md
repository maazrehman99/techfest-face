# NEXT_STEPS: status and recommendations (written 2026-10-21 ~02:30, owner closing their PC)

Read `CLAUDE.md` (brief) and `HISTORY.md` (what happened) first. This file is the hand-over: where we stopped, what is pending, what we recommend.

## State of the repo (pushed to `main`)
Working and tested: pipeline, evaluation harness, baseline numbers, augmentation (gender-aware), face memory (`gallery.py`, `live.py`), Wikimedia collector + cleaner, `finetune.py` (CPU-tested on the real tiny manifest: trains, validates, falls back to the baseline, exports verified ONNX).
**Never run:** `finetune.py` on a GPU or on Windows; `live.py` with a real webcam; `gallery.py scan` on a real video; the GPU-VM helper script (it lives outside this repo).

## The key finding: the Pakistani data we could collect is tiny
Wikimedia public-figure photos: 1500 people checked -> 688 free images -> 56 people with >=3 -> **13 people / 48 images after label cleaning** (train 10 people, val 2, test 1; the test split cannot yield EER). Cleaning works (inspect `results/pk_wiki/contact_*.jpg`; one probable mislabel, a few near-duplicates). So the Wikimedia set is a small real-Pakistani sample, NOT a training set.

## DECISION PENDING (owner has not chosen yet): how to get training volume
- **A.** Use the Wikimedia crops only as a small real-Pakistani TEST set (loosen `process_wikimedia.py` rules to ~20-30 people: `--min-images 2`, `--dominance 1.5`, `--admit 0.35`; expect more label noise, re-check contact sheets).
- **B.** Split halfFace (166 South Asian people, many real beards, masked/unmasked) BY PERSON into train/val/test and fine-tune with augmentation. This reverses the earlier "halfFace is evaluation-only" rule, so the baseline MUST be re-measured on the held-out halfFace people (`evaluate.py`) for a fair before/after.
- **C.** Add real consented photos of the three team members (dupatta/turban/beard shots). Even 20-30 people x 10 photos beats the whole Wikimedia set.
**Recommendation: A + B, and C if there is time.** Do not start the GPU fine-tune until this is decided.

## Recommendations on "make a big dataset Pakistani-like" (researched 2026-10-21)
- Large public sets (WebFace260M/42M, MS1M, VGGFace2, CASIA) are web-scraped celebrity photos without consent, hundreds of GB, need access requests, and the pretrained ArcFace was already trained on WebFace600K (a cleaned part of the same data), so little to gain. Current status/licences not verified.
- **Do not recolour skin tone and call it Pakistani data.** It changes colour, not facial structure; it is misleading and leans on a stereotype (Pakistanis span many complexions). Colour/lighting/occlusion changes are fine as *robustness augmentation*, described honestly as appearance variation.
- Closest relevant public data: **BUPT-BalancedFace** (has an Indian group) and **RFW** (Indian test); both need a request to the authors (access terms not verified), so they will not arrive before the 10-22/10-23 deadlines. Request them anyway (see `REQUESTS_TO_SEND.md`) for follow-up and for the report.
- **Beards:** current overlays are crude flat shapes. Improve them cheaply by blending the lower-face region of real bearded men (halfFace has many) onto clean-shaven faces using the 112x112 aligned crops (CPU-only). Generative editing (diffusion/GAN) looks better but needs a GPU and can change the person's identity (corrupts labels).

## Suggested order for the next session
1. Owner picks A / B / C (recommended A+B).
2. Implement the choice: halfFace person-split manifest (+ re-run baseline on the held-out people), loosened Wikimedia pass for the test set, optional real-texture beard augmentation.
3. Run `finetune.py` on a GPU (teammate's Windows RTX 5080 per `HANDOFF_FINETUNE.md`, or a rented Linux GPU VM, or free Kaggle/Colab). The training itself takes minutes; setup takes longer than training.
4. Evaluate fine-tuned vs baseline on halfFace held-out people AND the Pakistani Wikimedia set (per-group, ablation). Keep whichever is better; if the fine-tune does not help, say so in the report.
5. Measure unknown-person rejection and tune `identify_threshold` (currently a placeholder 0.40); `gallery.py reindex` after changing the model.
6. Test `live.py` with a real webcam; enroll the team with written consent for the demo.
7. Report (due 2026-10-22) and presentation. Required honesty points: halfFace is a South Asian proxy, Wikimedia set is small public figures (skews male/urban/elite), augmentation is synthetic, ArcFace weights are non-commercial, data licences/attribution (`ATTRIBUTION_AND_USE.md`), results of the fine-tune even if negative.
8. Owner still has to send the organizer questions in `REQUESTS_TO_SEND.md` (is public data allowed? hidden-benchmark input format?). Invite teammates to the private repo as collaborators.
9. After the competition delete the data, models and galleries (target 2026-11-30).
