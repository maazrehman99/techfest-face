# Every Fish Has a Face: Face Recognition for Pakistani Faces

**NED CSIT Tech Fest '26, Nemo's Deep Dive, Round 2** | Team: [team name] | Members: [names] | Date: 22 October 2026

> DRAFT. Everything marked **[TBD]** is filled in after the fine-tune and the held-out re-evaluation run.
> Sources for every number: `results/halfface/`, `results/aug/`, `DECISIONS.md`, `HISTORY.md`.

---

## 1. Summary
We built an end-to-end, offline, CPU-runnable face verification and identification system aimed at Pakistani faces,
including head and face coverings (dupatta, hijab/scarf, turban, cap), beards, surgical/cloth masks and CCTV-quality
images. The pipeline is: detect (SCRFD) -> quality check -> 5-point alignment -> ArcFace ResNet-50 embedding ->
cosine similarity -> threshold chosen on a validation split.

On the South Asian halfFace benchmark the pretrained baseline reaches **97.6% accuracy, EER 2.32%, AUC 0.993,
TAR 95.8% at FAR 1%** (3,312 test pairs; threshold fixed on separate validation people). Masked faces are the main
weakness (96.6% accuracy vs 99.85% unmasked). We then fine-tuned the recognizer on South Asian + Pakistani public-figure
data with covering augmentation: **[TBD: fine-tuned result vs baseline on the held-out people]**.

Two modes share one model:
1. **Verify**: two images in -> same / different person + similarity score (`run.py`). This is the benchmark entry point.
2. **Face memory**: enroll a consenting person once, then recognise them in photos, video or a live camera feed and show
   name + score or "Unknown", with no retraining (`gallery.py`, `live.py`).

## 2. Problem and constraints
- Recognise Pakistani faces reliably, including cultural coverings, beards, masks and low-quality capture.
- No public identity-labelled Pakistani face dataset exists (we searched; PFDB only labels Pakistani vs not).
- Must run offline on CPU and never crash on an image without a face.
- Team of three with one week; GPU access limited (one consumer NVIDIA GPU).

## 3. Pipeline (design and reasoning)
| Stage | What we use | Why |
|---|---|---|
| Detection | SCRFD-10G (InsightFace `buffalo_l`), 5 landmarks | Accurate, CPU-friendly, gives landmarks for alignment. We added a pad-and-retry fallback because SCRFD misses extreme close-ups, and upscaling + padding for tiny crops. |
| Quality check | Laplacian blur, brightness, face size, detector score | Cheap and interpretable; each term maps to a stated condition (blur, lighting, resolution). Its effect is measured, not assumed (section 6). |
| Alignment | Similarity transform to the 112x112 ArcFace template | Ablation shows it is essential (EER 2.3% -> 27.6% without it). |
| Embedding | ArcFace ResNet-50 (`w600k_r50`, WebFace600K), 512-d | Strongest model that runs acceptably on CPU; clean fine-tuning path (ONNX -> PyTorch). |
| Matching | Cosine similarity, threshold from validation | ArcFace is trained with an angular margin, so cosine is its native metric. Threshold 0.276 = FAR 1% on validation people, never tuned on test. |

Failure handling: an image with no detectable face returns `status=no_face`, score -1 and decision "different";
the run never crashes.

Reused components (cited, not written by us): InsightFace SCRFD and ArcFace `buffalo_l` weights
(non-commercial research licence). Our own work: evaluation harness, quality filter, augmentation, data collection and
cleaning tools, fine-tuning script, face-memory system.

## 4. Data
| Set | Size | Role |
|---|---|---|
| halfFace (Mendeley pk44mkx9vm, CC BY 4.0) | 166 South Asian people, 938 images, masked/unmasked, 3 poses | Benchmark; for fine-tuning split **by person** into train / val / held-out test (60/15/25%) |
| Pakistani public figures (Wikimedia, CC/public-domain) | [TBD final count; first pass 13 people / 48 images] | Real Pakistani faces: training + small Pakistani test |
| Synthetic coverings (`augment.py`) | applied on the fly | Training-time augmentation only |

**Wikimedia collection.** Pakistani people from Wikidata, their article images from 20 Wikipedia language editions,
licence and author recorded per image. Filters: adults only, born in Pakistan after 1947, Bangladeshi citizens
excluded (Wikidata wrongly lists East-Pakistan-era Bangladeshis as Pakistani), sensitive cases excluded.
**Label cleaning:** article images often show other people (group photos, colleagues, graves, medals). We keep only
images where one face clearly dominates, trust images whose file name contains the person's name ("anchors"), and admit
other images only if they match the anchors (cosine >= 0.40). Of ~1,500 people checked, most had only one usable photo;
the cleaned set is small. We report this honestly instead of padding it with unreliable data.

**Why not AI-generated faces:** generators do not keep one identity consistent across images, encode stereotypes, and
create a synthetic-vs-real gap. We do not use them for training. We also do not recolour skin to "make" data Pakistani:
that changes colour, not identity, and relies on a stereotype.

**Why split halfFace by person (decision of 21 Oct):** the Pakistani set alone is too small to fine-tune on. Splitting
halfFace by person gives training volume while keeping a clean held-out test: no person appears in two splits (checked
in code), and the baseline is re-measured on exactly the same held-out people.

## 5. Fine-tuning
- Start from pretrained `w600k_r50` (ONNX converted to PyTorch).
- Only the last 30% of layers trainable; BatchNorm frozen (small batches would corrupt its statistics).
- ArcFace margin head (s=32, m=0.30) initialised from each identity's mean pretrained embedding.
- Augmentation on training crops only, gender-aware: masks, cloth cover, hijab, dupatta, cap, turban, glasses,
  beard (male faces only), low/harsh light, blur, low resolution, JPEG, CCTV simulation; plus horizontal flip.
- AdamW, lr 1e-4 (head 1e-3), cosine schedule, mixed precision, up to 15 epochs, early stopping on **validation EER**.
- Safety rule: if no epoch beats the untouched baseline on validation, the baseline is kept and we say so.
- Hardware: NVIDIA RTX 5070 Ti. Training time **[TBD]**.

Training curve: **[TBD: history.csv plot, loss and val EER per epoch]**.

## 6. Evaluation
Protocol: identities never shared between splits; thresholds chosen on validation; all pairs of the same person as
genuine pairs and an equal number of random different-person impostor pairs; metrics at the fixed threshold.

### 6.1 Baseline on halfFace (126 test people, 3,312 pairs)
| Metric | Value |
|---|---|
| Accuracy | 97.6% |
| Precision / Recall / F1 | 0.979 / 0.973 / 0.976 |
| FAR / FRR (val EER threshold 0.261) | 2.05% / 2.72% |
| EER | 2.32% |
| AUC | 0.993 |
| TAR @ FAR 1% / 0.1% | 95.8% / 88.2% |
| No-face rate | 1.8% |

### 6.2 Baseline vs fine-tuned on the held-out halfFace people [TBD]
| Model | EER | Accuracy | TAR@FAR1% | Masked acc. | Unmasked acc. |
|---|---|---|---|---|---|
| Baseline (re-measured on the same held-out people) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Fine-tuned | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Pakistani public-figure test: **[TBD]**.

### 6.3 Ablation (baseline, halfFace test)
| Configuration | EER | Accuracy |
|---|---|---|
| Full pipeline | 2.32% | 97.6% |
| Without quality filter | 2.44% | 97.6% |
| Without alignment (plain box crop) | 27.6% | 72.4% |
| With flip test-time augmentation | 2.11% | 97.9% |
| Minimal (no alignment, no filter) | 27.2% | 72.9% |

Alignment is by far the most important stage. The quality filter has almost no effect on halfFace (cooperative,
well-lit photos) and is kept for real-world CCTV/phone input. Flip averaging gives a small gain at twice the
embedding cost.

### 6.4 Per-group results (baseline)
| Group | Accuracy | FAR | FRR |
|---|---|---|---|
| Unmasked | 99.85% | 0.29% | 0.0% |
| Masked | 96.6% | 4.63% | 2.16% |
| Frontal / left / right pose | 98.0% / 98.6% / 95.3% | | |
| Male / female | 97.5% / 98.8% | | |

Female results rest on only 258 pairs (the dataset has few women), so they are uncertain.

### 6.5 Robustness to coverings and image quality (synthetic, 60 people)
Mean same-person similarity (clean = 0.728; lower means harder):
cloth cover 0.417, surgical mask 0.458, CCTV 0.561, beard 0.572, low-res 0.606, hijab 0.611, sunglasses 0.617,
blur 0.625, glasses 0.664, JPEG 0.665, dupatta 0.668, low light 0.668, turban 0.680, cap 0.683, harsh light 0.709.
Masks, cloth covers, CCTV degradation and beards are the hardest conditions; this guided the augmentation mix.
Head coverings that leave the face visible (dupatta, turban, cap) barely affect the model.

### 6.6 Face memory
Small test: 15 halfFace people, 3 unmasked enrollment photos each; 37 masked probe images: 33 correct, 4 "Unknown",
**0 wrong names**. Unknown-person rejection and the identification threshold: **[TBD]**.

## 7. Ethics, consent and licences
- Wikimedia images: free licences only, author and licence recorded per image; adults only; sensitive cases excluded.
  A free licence is not the person's consent to face-recognition research, so the data is used only for this
  competition, kept private and deleted afterwards (target 30 Nov 2026).
- Face memory requires `--consent` for every enrollment, stores a consent note, and supports erasure (`gallery.py remove`).
  Only team members who gave written consent are enrolled for the demo.
- Pretrained weights are for non-commercial research; this is not a product.
- The system must not be used to identify private individuals or build watch-lists.

## 8. Limitations (stated honestly)
- halfFace is a South Asian proxy, not a Pakistani-specific dataset; it is also cooperative and well lit.
- The Pakistani set is small and made of public figures, who skew male, urban and elite; results do not represent all
  Pakistanis.
- Covering augmentation is synthetic (flat overlays), not real photos of the same person with and without coverings.
- Few women in the data, so gender results are uncertain.
- Live webcam and CCTV video use are demonstrated but not formally benchmarked.

## 9. How to run
```
python run.py verify --a img1.jpg --b img2.jpg          # -> {"score": ..., "decision": 0/1, ...}
python run.py pairs --csv pairs.csv --out scores.csv    # batch of pairs
python run.py embed --images DIR --out embeddings.npz
python gallery.py enroll --name "Name" --images photos/ --consent
python live.py --source 0                                # browser demo at http://127.0.0.1:8000
```
Offline, CPU, Python 3.12. Threshold and model file in `config.json`.

## 10. References
- Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR 2019.
- Guo et al., Sample and Computation Redistribution for Efficient Face Detection (SCRFD), ICLR 2022.
- InsightFace project, `buffalo_l` model pack.
- halfFace: A face covering mask dataset of South Asian people, Mendeley Data, pk44mkx9vm (CC BY 4.0).
- Wikimedia Commons / Wikidata; per-image attribution in `data/pk_wiki/manifest.csv`.
