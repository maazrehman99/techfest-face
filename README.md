# Face recognition for Pakistani faces

Offline face verification and identification that runs on a normal CPU. Built for Round 2 of Nemo's Deep Dive ("Every Fish Has a Face") at NED CSIT TechFest '26, where the team finished runner-up.

The task was to adapt a pretrained face recognition model to Pakistani faces, including people wearing a dupatta, hijab, cap, turban or mask, and to evaluate it honestly. No public identity-labelled Pakistani face dataset exists, so the work uses South Asian datasets as a proxy and a small set of Pakistani public figures as a final check.

Full write-up: [docs/Face_Recognition_Report.pdf](docs/Face_Recognition_Report.pdf)

## Results

Held-out people only. Thresholds were picked on a separate validation split, never on the test set.

| Test set | Accuracy | EER | AUC |
|---|---|---|---|
| halfFace, 40 held-out South Asian people | 97.1% | 2.9% | 0.991 |
| halfFace, same people, without face alignment | 74.6% | 25.4% | 0.814 |
| IMFDB, 24 held-out Indian film actors (hard pose and light) | 87.5% | 10.5% | 0.955 |
| 13 Pakistani public figures, 48 unseen Wikimedia photos | 99.2% | not measured | 1.000 |

Three findings:

- Face alignment matters most. Without it the error rate rises from 2.9% to 25.4%.
- Fine-tuning ArcFace on about 3,000 South Asian photos (Google Cloud L4 GPU) did not clearly help. It improved hard movie frames and masked faces a little, made clean photos worse (EER 2.9% to 3.9%), and a bootstrap over people showed the gain was marginal. The original model ships, and the fine-tune is reported as a mixed result.
- Masks, cloth face covers, beards and CCTV-quality images lower the match score the most. A dupatta, turban or cap barely changes it, because the face itself stays visible.

Every number, per-group breakdowns (mask, pose, gender, blur, face size) and the fine-tuning curve are in [docs/RESULTS.md](docs/RESULTS.md).

## How it works

```
image -> detect face (SCRFD-10G, 5 landmarks) -> quality check (blur, light, size)
      -> align to 112x112 -> ArcFace ResNet-50 embedding (512 numbers)
      -> cosine similarity -> threshold
```

The same model serves two modes:

- Verify: two photos in, "same person" or "different person" out, with a score. This was the benchmark mode.
- Face memory: enrol a person once (with their consent), then name them or say "Unknown" in photos, video or a live webcam feed, with no retraining.

Design choices and the reasoning behind each one are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

Tested with Python 3.12.

```
python -m venv .venv
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

The model weights are not in the repo. Install InsightFace's `buffalo_l` pack (it downloads to `~/.insightface/models/buffalo_l` on first use) and copy it to `models/buffalo_l/`.

## Usage

Verify two photos:

```
python run.py verify --a person1.jpg --b person2.jpg
```

Score a list of pairs (CSV with two image path columns):

```
python run.py pairs --csv pairs.csv --out scores.csv
```

Face memory:

```
python gallery.py enroll   --name "Ali Khan" --images photos/ali --consent
python gallery.py identify --images group.jpg --save out/
python gallery.py scan     --input video.mp4 --out appearances.csv
python gallery.py remove   --name "Ali Khan"
python live.py --source 0      # webcam demo at http://127.0.0.1:8000
```

An image with no detectable face returns `no_face` and a rejection. It never crashes the run. The decision thresholds live in `config.json`.

Evaluate on a labelled dataset (ROC, EER, TAR at fixed FAR, per-group results, stage ablation):

```
python evaluate.py --manifest data/halfface/manifest.csv --out results/halfface --ablation
```

## Repository layout

| Path | Contents |
|---|---|
| `pipeline.py` | Detection, quality filter, alignment and embedding, each stage switchable |
| `run.py` | Command-line entry point for verification |
| `gallery.py`, `live.py` | Face memory and the live webcam demo |
| `evalkit.py`, `evaluate.py` | Metrics, ablation and per-group evaluation |
| `augment.py` | Synthetic coverings (mask, hijab, dupatta, cap, turban, beard) and image damage (blur, low light, CCTV) |
| `build_ft_data.py`, `build_imfdb.py`, `finetune.py` | Person-disjoint splits and GPU fine-tuning, exported to ONNX |
| `collect_wikimedia.py`, `process_wikimedia.py` | Collection and cleaning of freely licensed photos of Pakistani public figures |
| `submission/` | Identification script used on the competition's lab day |
| `tools/` | Cloud GPU job, bootstrap tests, identification benchmark |
| `data/` | Split lists only (which person is in which split). No face images. |
| `results/` | Raw evaluation output (JSON, CSV, ROC plots) |
| `docs/` | Report, full results, design decisions, data sources |

## Data and ethics

- halfFace (CC BY 4.0) and IMFDB (IIIT Hyderabad) are used as South Asian proxies. Neither is Pakistani-specific.
- The Pakistani check uses freely licensed Wikimedia photos of adult public figures. A free licence is not the person's consent, so these photos are not published here and were not used to train the GPU model.
- People are split by identity: nobody appears in both training and test data.
- Face memory refuses to enrol anyone without the `--consent` flag, and `gallery.py remove` erases a person.
- The pretrained InsightFace weights are licensed for non-commercial research only.

Sources and licences: [docs/DATA.md](docs/DATA.md).

## Limitations

- Only about 3,000 training photos of 159 people, too few to improve a model trained on millions of faces.
- About 90% of halfFace people are men, so results for women are uncertain.
- The Pakistani test is small and easy (clear portraits). A harder Pakistani test with real coverings is still needed.
- Unknown-person rejection in face memory has not been measured yet. The identification threshold (0.40) is a conservative default.

## Team

Built together by:

- Maaz Rehman ([@maazrehman99](https://github.com/maazrehman99))
- Shaikh Muhammad Zain ([@ZainDev04](https://github.com/ZainDev04))
- Syed Mohammad Huzaifa ([@huzaifaguru](https://github.com/huzaifaguru))
