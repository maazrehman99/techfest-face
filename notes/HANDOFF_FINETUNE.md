# Handoff: fine-tune the face model on your RTX 5080 PC

Written 2026-10-21. Read this top to bottom; you do not need the chat history.

## 1. What this project is (2 minutes)
Competition: NED CSIT "Nemo's Deep Dive", Round 2 "Every Fish Has a Face". We must build a face recognition system for **Pakistani faces** (dupatta, scarf/hijab, turban, cap, beard, masks, CCTV/low light) and **fine-tune a pretrained face-embedding model**. Report due **2026-10-22**, organizers' hidden benchmark **2026-10-23**.

The system has two modes on one model:
1. **Verify** (`run.py verify`): two images in, "same person or not" + score out. This is what the organizers test.
2. **Face memory** (`gallery.py`, `live.py`): enroll a person once (with consent), then show their name or "Unknown" on images/video/live camera. No retraining per person.

Model: InsightFace **ArcFace ResNet-50** (`models/w600k_r50.onnx`, weights are non-commercial research only: fine for the competition, must be cited). Detector: SCRFD (`models/buffalo_l/det_10g.onnx`). Both are reused, not written by us.

## 2. Your job
Run `finetune.py` on the 5080, get `finetuned.onnx` + `log.txt` + `history.csv`, and send them back. That is all. The script decides by itself whether the fine-tune helped (see 6).

## 3. Where we are (what is already done)
- Baseline (frozen ArcFace) measured on halfFace (public South Asian masked/unmasked set): **EER 2.32%, accuracy 97.6%, TAR@FAR1% 95.8%**. Little headroom there, so the real proof must come from the Pakistani held-out split below.
- Training data = CC-licensed / public-domain photos of **Pakistani public figures from Wikimedia** (no private individuals, licence + author recorded per image in `records.jsonl`/`manifest.csv`). Split **by person** into train/val/test (70/15/15) so nobody appears in two sets. Size is small (hundreds of images), so the recipe is deliberately conservative.
- Augmentation (`augment.py`) adds dupatta, hijab, turban, cap, beard (male faces only), glasses, masks, lighting, blur, low-res, CCTV effects to **training** crops only.
- Data status when this was written: collection was still running and `process_wikimedia.py` had not yet been run. **Do not start until you have the dataset zip** (`techfest_pk_wiki.zip`, produced after processing; you will get it from the team). Its `manifest.csv` has columns `crop, identity, split, gender, license, ...`.

## 4. Hard rules (these protect the honesty of the results)
1. **Never train on halfFace or on the `test` split.** `finetune.py` trains on `train`, picks the best epoch on `val`, and never touches `test`.
2. Do not edit the manifest's `split` column.
3. Do not add augmented images to val/test.
4. Keep every log. If the fine-tune does not help, the report says so; that is a valid result.
5. Faces of real people: keep the dataset on your PC only, do not post it anywhere, delete it after the competition.

## 5. Setup on the 5080 PC (Windows 11 first, Linux notes in brackets)
The project was built and tested on Linux only. I reviewed the code and these steps for Windows problems (file encodings, paths, multiprocessing, PowerShell syntax) but **nothing has been run on Windows or on the 5080**, so expect small hiccups and report them.

**Before you start (the 6 Windows traps):**
1. **Python 3.11 or 3.12**, not 3.13/3.14 (some packages have no wheels there yet). Check with `python --version`; install from python.org and tick "Add to PATH".
2. **Plain ASCII folder path**, short and not inside OneDrive/Desktop, e.g. `C:\techfest`. OpenCV cannot open images from folders with non-English characters, and Windows has a 260-character path limit.
3. **NVIDIA driver 570 or newer** (`nvidia-smi` in PowerShell shows it; update with the NVIDIA app or GeForce Experience). You do NOT need to install the CUDA Toolkit; the PyTorch wheel brings its own CUDA 12.8 runtime.
4. PowerShell blocks the venv activation script by default: run `Set-ExecutionPolicy -Scope Process Bypass` first (affects only that window).
5. Use **PowerShell**, not cmd. The commands below are PowerShell. [Linux: the same without the PowerShell lines; `source .venv/bin/activate`.]
6. If you use WSL2 instead, treat it as Linux; it works too, but plain Windows is simpler.

**Get the code and data:** `git clone https://github.com/maazrehman99/techfest-face.git C:\techfest` (you must be invited as a collaborator; the repo is private). The processed Pakistani crops are already in `data\pk_wiki\`.

**Get the pretrained model** (it is too big for git, 167 MB):
```
cd C:\techfest
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -OutFile models\buffalo_l.zip
Expand-Archive models\buffalo_l.zip -DestinationPath models\buffalo_l -Force
Copy-Item models\buffalo_l\w600k_r50.onnx models\w600k_r50.onnx
```
(`finetune.py` reads `models\w600k_r50.onnx`; the zip is 288 MB and has the other InsightFace models too.) Do not `pip install insightface`: it needs a C++ compiler on Windows and the fine-tune does not use it.

**Python environment:**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install onnx2torch onnx onnxruntime opencv-python-headless scikit-learn pandas tqdm numpy matplotlib
```

**Check the GPU before the long run:**
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(2000,2000,device='cuda'); print((x@x).sum().item())"
```
It must print `True NVIDIA GeForce RTX 5080` and a number. A "no kernel image / sm_120" error means an older PyTorch build was installed: `pip uninstall -y torch torchvision` and repeat the cu128 install line.

## 6. Run it
```
python finetune.py --manifest data/pk_wiki/manifest.csv --root data/pk_wiki --out runs/ft1
```
Defaults: 15 epochs, batch 64, lr 1e-4, last 30% of layers trainable, BatchNorm frozen, early stopping after 4 epochs without improvement. Expected: well under an hour; memory use about 4-6 GB of the 16 GB. If you run out of memory add `--bs 32`. On Windows the data loader starts worker processes by re-importing the script, which can be slow or hang: if the first epoch does not start within a couple of minutes, stop it and add `--workers 0` (slower but safe). Run the command from the project folder with the venv activated; use backslashes or quotes as needed, e.g. `python finetune.py --manifest data\\pk_wiki\\manifest.csv --root data\\pk_wiki --out runs\\ft1`.

What the log shows: `epoch 0 (baseline) val EER x%`, then one line per epoch. Lower validation EER is better. At the end it prints either
- `best val EER A% vs baseline B%` (the fine-tune helped), or
- `NO epoch beat the baseline ... exporting the unchanged baseline` (it did not help; report that honestly).

Optional second run, only if the first one improved: `--trainable 0.5 --lr 5e-5 --out runs/ft2`. Do not run more than 2-3 configurations; every extra config chosen by val is a bit of tuning on val.

## 7. Send back
The whole `runs/ft1/` folder: `finetuned.onnx` (about 170 MB), `log.txt`, `history.csv`. Also paste the last 15 lines of the log in chat.

## 8. If something fails
Send the last 20 lines of the error. Likely causes in order: (a) GPU/driver/PyTorch build (section 5), (b) `onnx2torch` cannot convert the model (send the error; the team has a fallback), (c) paths: run from the project folder and check `data/pk_wiki/manifest.csv` exists, (d) out-of-memory: `--bs 32`.

## 9. What the team does with your result
Set `rec_onnx` in `config.json` to the new file, re-run `evaluate.py` on halfFace and on the held-out Pakistani `test` split, compare with the baseline numbers, rebuild the face-memory gallery (`gallery.py reindex`), and write the report.

*Status caveat: `finetune.py` was written and smoke-tested on CPU only. It has never run on the 5080, so expect to debug small things on your side and tell us.*
