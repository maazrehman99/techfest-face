# NEXT_STEPS: status and how to continue (updated 2026-10-22 ~03:00)

Read `CLAUDE.md` (brief), `HISTORY.md` (what happened) and **`report/RESULTS_FOR_REPORT.md` (every number, plain words)**.
**Report due 2026-10-22, hidden benchmark 2026-10-23.**

## State
- Model decision taken (D13): **ship the original ArcFace R50** (`config.json` `rec_onnx` = null). The fine-tune (`runs/ft1/finetuned.onnx`, not in git) gave a small, mixed result and is reported honestly.
- GPU VM deleted; nothing is running in the cloud. Results are in `results/ft_gpu_run/`.
- Data lists: `data/ft_manifest.csv`, `data/halfface/manifest_ft.csv`, `data/imfdb/manifest_eval.csv` (who is in which split). Images stay local (halfFace and IMFDB are git-ignored).

## Do next, in this order
1. **Send the organizer questions** in `REQUESTS_TO_SEND.md` (input format of the hidden benchmark decides how `run.py` is packaged; is public data allowed?). The user sends these.
2. **Offline dry run of `run.py`** on a clean setup: `verify`, `pairs`, `embed`, blank image, missing file, non-ASCII path; no internet; CPU only.
3. **Re-pick the threshold** on halfFace-like validation data (the mixed halfFace+IMFDB validation gave about 5% false accepts on clean halfFace instead of 1%). Current `config.json` threshold 0.276 came from the first halfFace validation; the new halfFace validation gives 0.290 at FAR 1%. Decide and note it.
4. **Report**: `report/REPORT_DRAFT.md` is filled from the results; check wording, add the figures (ROC pngs are in `results/ft_gpu_run/*/roc.png`), add the Pakistani photos section if a test set exists.
5. **Harder Pakistani test (optional, test only)**: the user hand-picks 3 to 5 photos per person (with dupatta, hijab, cap, beard, mask) into `data/pk_test/<Name>/` plus source links, then `python tools/eval_pk_test.py --dir data/pk_test --ft runs/ft1/finetuned.onnx --thr 0.290 --thr-ft 0.350`. Or an Openverse API key for `collect_openverse.py` (it hit a bot check without one).
6. Measure unknown-person rejection, tune `identify_threshold` (0.40 placeholder), `gallery.py reindex`; test `live.py` with a real webcam; presentation and live demo.
7. After the competition: delete data, models and galleries (target 2026-11-30).

## Useful commands
- Check/run the cloud job (needs a login and a project id): `bash tools/gcp_finetune.sh check PROJECT` (then `run`/`resume`/`delete`). Rule: drive it by hand; delete only after results are downloaded and verified.
- Compare original vs fine-tuned on crops: `tools/eval_crops.py`; significance: `tools/boot_ci.py`; full pipeline: `evaluate.py --manifest ... --rec ...`.
- Phone alerts: `tools/ntfy.sh "title" "message" [priority]` (topic in `~/.config/techfest_ntfy_topic`).
