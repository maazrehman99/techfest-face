# Results

Every number below was copied by script from the result files in `results/`, not typed by hand. Short version first, then every number, then the limits.

## 0. Key points

1. **We use ArcFace (ResNet-50, InsightFace `buffalo_l`) as the recogniser, with SCRFD-10G as the face detector. It was our choice, made on purpose** (decisions D1 and D2 in `DECISIONS.md`). Reason: it is the strongest model that runs on a normal CPU, it is a standard, well-trusted design, and the competition judges the pipeline, the fine-tuning attempt and the evaluation, not a new backbone. AdaFace was the optional challenger; we never tested it for lack of time. The weights are non-commercial research only, so we cite them and never present this as a product.
2. **The pipeline is: detect the face, find 5 landmarks, check quality, straighten (align) to 112 by 112, turn into 512 numbers (embedding), compare with cosine similarity, apply a threshold.** One model serves two modes: *verify* (two photos, same or different person) and *face memory* (name or Unknown).
3. **Alignment is essential.** Without it the error (EER) jumps from 2.9% to 25.4% on the held-out halfFace people.
4. **The original model is already strong:** 97.1% accuracy and 2.9% EER on 40 held-out South Asian people (halfFace), 98.2% accuracy on unmasked faces and 94.3% on masked faces.
5. **Fine-tuning on our small data did NOT clearly help.** It lowered the validation error a little at the first epoch (13.24% to 12.04%) and then got worse (it memorised the training people). On the full pipeline it was slightly worse on clean halfFace photos, slightly better on the hard IMFDB movie frames, better on masked faces, and women on IMFDB lost accuracy at the shared threshold (83.0% to 77.8%) although their threshold-free EER did not change (14.5% vs 14.4%), so that drop is mostly the threshold moving. So we keep the original model and report the fine-tune as an honest experiment with mixed results.
6. **On 13 Pakistani public figures (48 Wikimedia photos the fine-tuned model never saw) both models are almost perfect** (99.2% vs 99.4% accuracy, AUC 0.99996). This shows the system works on Pakistani faces, but the test is easy (clear portraits, no coverings), so it cannot separate the two models.
7. **Data is the main limit.** No public Pakistani identity dataset exists. We used South Asian proxies (halfFace, IMFDB) plus a small Wikimedia set of Pakistani public figures. Only about 3,000 training photos of 159 people, and few women.
8. **Honest labels:** halfFace is South Asian (not Pakistani-specific); IMFDB is Indian film actors; Wikimedia photos are public figures from article pages. IMFDB's website states no licence, so we credit it and keep it local.
9. **Evaluation rigour:** people never appear in two splits, thresholds are chosen on validation only, no photo of a held-out person was used for training, differences are checked with a bootstrap over people, and negative results are reported.
10. **The organizers (NED) announced on 2026-10-21 that the consent-process item in the instructions was a mistake and is not required.** We still credit every data source and keep face data local.

## 1. Data we used

| Set | Role | People | Photos | Notes |
|---|---|---|---|---|
| halfFace (Mendeley, CC BY 4.0) | train / val / test by person (60/15/25) | 166 (151 men, 15 women) | 938 (470 masked, 468 unmasked) | South Asian proxy, real masks, controlled photos |
| IMFDB (IIIT Hyderabad) | train / val / test by actor | 98 actors, 40 photos each | 3,920 | Indian film frames, about 32% women, hard pose and light. Site states no licence. |
| Wikimedia Pakistani public figures (CC) | Pakistani check, never trained on the GPU run | 13 | 48 | 41 men and 7 women photos, article portraits |
| LFW | sanity check only | | | |

Split sizes used in the GPU run (people / photos):

| Split | halfFace | IMFDB | Total |
|---|---|---|---|
| Train | 100 / 559 | 59 / 2,360 | **159 / 2,919** |
| Validation | 26 / 141 | 15 / 600 | **41 / 741** |
| Test (held out) | 40 / 226 | 24 / 960 | **64 / 1,186** |

Note: the Wikimedia photos were **left out of the GPU training** (kept local, privacy rule), so the Pakistani check is truly unseen by both models. The full-pipeline test used the original photos of the validation and test people only (halfFace 372 photos of 66 people, IMFDB 1,560 photos of 39 actors).

## 2. The original model, full pipeline, held-out people

Threshold = the value that gives equal errors on the validation people. "Pairs" = same-person and different-person pairs.

| Model | Accuracy | Precision | Recall | F1 | False accepts (FAR) | False rejects (FRR) | EER | AUC | TAR at FAR 1% | Pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| Original, halfFace held-out (40 people) | 97.1% | 97.3% | 96.9% | 97.1% | 2.7% | 3.1% | 2.9% | 0.9906 | 94.2% | 1,030 |
| Original, IMFDB held-out (24 actors) | 87.5% | 84.2% | 92.3% | 88.1% | 17.3% | 7.7% | 10.5% | 0.9545 | 71.1% | 28,092 |

- Photos where no face was found: halfFace 2.4%, IMFDB 0.0%. Photos rejected by the quality filter: halfFace 0.0%, IMFDB 14.8%.
- Thresholds picked on validation (cosine): equal-error 0.241, FAR 1% 0.290 (halfFace); equal-error 0.115, FAR 1% 0.208 (IMFDB). The two data sets need different thresholds, so one number cannot be perfect for both. The shipped `config.json` threshold (0.276) was picked earlier on halfFace validation at FAR 1%.

### Ablation: what each pipeline stage is worth (original model, halfFace held-out)

| Pipeline variant (original model, halfFace held-out) | Accuracy | EER | AUC |
|---|---|---|---|
| Full pipeline (detect, align, quality filter) | 97.1% | 2.9% | 0.9906 |
| Without the quality filter | 97.1% | 2.9% | 0.9906 |
| Without alignment (plain face box) | 74.6% | 25.4% | 0.8141 |
| With mirror-image averaging | 97.7% | 2.3% | 0.9912 |
| Minimal (no alignment, no filter) | 74.6% | 25.4% | 0.8141 |

Plain reading: alignment is the single most important step. The quality filter changed nothing on these cooperative photos. Mirror-image averaging helps a little.

### By group (halfFace held-out, original vs fine-tuned)

| Group (halfFace held-out) | Pairs | Original acc | Fine-tuned acc | Original EER | Fine-tuned EER |
|---|---|---|---|---|---|
| mask=unmasked | 220 | 98.2% | 97.7% | 0.0% | 0.0% |
| mask=masked | 194 | 94.3% | 96.4% | 5.7% | 4.6% |
| pose=front | 80 | 98.8% | 98.8% | 2.5% | 2.5% |
| pose=left | 64 | 98.4% | 95.3% | 1.6% | 3.1% |
| pose=right | 64 | 98.4% | 96.9% | 1.6% | 3.1% |
| gender=male | 968 | 96.4% | 95.7% | 3.6% | 4.0% |
| gender=female | 62 | 100.0% | 100.0% | 0.0% | 0.0% |
| blur_bin=blurry | 206 | 97.1% | 94.2% | 3.9% | 2.9% |
| blur_bin=mid | 138 | 97.8% | 97.8% | 1.5% | 2.2% |
| blur_bin=sharp | 232 | 97.8% | 97.8% | 1.7% | 2.2% |
| bright_bin=normal | 866 | 96.7% | 95.8% | 3.0% | 4.2% |
| size_bin=large>120px | 1,028 | 96.5% | 96.0% | 3.1% | 3.7% |

### By group (IMFDB held-out)

| Group (IMFDB held-out) | Pairs | Original acc | Fine-tuned acc | Original EER | Fine-tuned EER |
|---|---|---|---|---|---|
| gender=male | 10,000 | 84.3% | 84.0% | 11.3% | 11.9% |
| gender=female | 9,932 | 83.0% | 77.8% | 14.5% | 14.4% |
| blur_bin=blurry | 2,182 | 83.7% | 83.3% | 15.0% | 15.2% |
| blur_bin=mid | 3,944 | 87.2% | 88.4% | 11.1% | 10.4% |
| blur_bin=sharp | 5,232 | 90.4% | 91.6% | 5.6% | 4.9% |
| bright_bin=dark | 4,160 | 87.9% | 88.8% | 10.2% | 9.6% |
| bright_bin=normal | 10,000 | 87.6% | 87.9% | 10.0% | 10.0% |
| size_bin=mid | 4,288 | 87.1% | 87.8% | 10.3% | 9.4% |
| size_bin=small<60px | 2,480 | 83.5% | 84.0% | 15.8% | 16.0% |
| size_bin=large>120px | 5,606 | 91.1% | 90.9% | 3.4% | 3.1% |

## 3. Earlier results (first held-out split, before the person split)

From the first baseline run on 126 halfFace people, 3,312 pairs: EER 2.32%, AUC 0.993, accuracy 97.6%, precision 0.979, recall 0.973, F1 0.976, TAR at FAR 1% 95.8%, TAR at FAR 0.1% 88.2%, no-face rate 1.8%. Unmasked accuracy 99.85%, masked 96.6%. Not comparable with the new tables because the held-out people are different.

How much each covering lowers the same-person similarity (60 people, clean photos average 0.728; lower = harder): cloth cover 0.417, surgical mask 0.458, CCTV-style degradation 0.561, beard 0.572, hijab 0.611, blur 0.625, dupatta 0.668, turban 0.680, cap 0.683. Our synthetic dupatta, turban and cap overlays are crude flat shapes, so the model barely reacts to them; masks, cloth covers, beards and CCTV degradation are the real challenges.

Face-memory test (halfFace, 15 people, 3 enrolment photos each): 33 of 37 masked photos named correctly, 4 said Unknown, 0 named the wrong person. Unknown-person rejection and the identification threshold (0.40 is a placeholder) are not measured yet.

## 4. The fine-tuning experiment

Setup (plain words): start from the original ArcFace weights; keep about 70% of the layers frozen and retrain the last 30% (30,683,136 of 43,572,288 numbers); learning rate 0.0001; batches of 64; random coverings (dupatta, hijab, turban, cap, mask, beard for men only), blur, low light and CCTV-style damage added to training photos only; stop when the validation error has not improved for 4 epochs; keep the best epoch, and fall back to the original if none is better. Trained on one NVIDIA L4 GPU on a Google Cloud VM in Mumbai (about 5 seconds per epoch).

| Epoch | Training loss | Validation EER |
|---|---|---|
| 0 (original model) | n/a | 13.24% |
| 1 | 5.13 | 12.04% |
| 2 | 3.76 | 13.11% |
| 3 | 3.11 | 12.85% |
| 4 | 2.76 | 13.23% |
| 5 | 2.42 | 13.80% |

Best epoch was 1. Later epochs got worse, which means the model memorised its 159 training people.

### Original vs fine-tuned on the cropped faces of the held-out people (GPU)

| Test group (crops) | People | Photos | Original EER | Fine-tuned EER | Original TAR@FAR1% | Fine-tuned TAR@FAR1% |
|---|---|---|---|---|---|---|
| all_test | 64 | 1186 | 12.1% | 10.9% | 68.4% | 68.8% |
| halfface | 40 | 226 | 1.7% | 1.5% | 96.8% | 98.0% |
| halfface_masked | 40 | 112 | 1.4% | 1.9% | 98.1% | 98.1% |
| halfface_unmasked | 40 | 114 | 0.0% | 0.0% | 100.0% | 100.0% |
| imfdb | 24 | 960 | 13.4% | 12.5% | 65.1% | 64.3% |

### Is the difference real? (bootstrap over people, 200 resamples)

| Group | Fine-tuned minus original EER (points) | 95% range | Fine-tuned better in |
|---|---|---|---|
| all_test (64 people) | -1.10 | -2.22 to -0.04 | 97.5% of 200 resamples |
| halfface (40 people) | +0.17 | -0.85 to +1.18 | 36.5% of 200 resamples |
| imfdb (24 people) | -0.98 | -1.91 to +0.14 | 94.0% of 200 resamples |

### Original vs fine-tuned, full pipeline

| Model | Accuracy | Precision | Recall | F1 | False accepts (FAR) | False rejects (FRR) | EER | AUC | TAR at FAR 1% | Pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| Original, halfFace | 97.1% | 97.3% | 96.9% | 97.1% | 2.7% | 3.1% | 2.9% | 0.9906 | 94.2% | 1,030 |
| Fine-tuned, halfFace | 96.1% | 95.8% | 96.5% | 96.1% | 4.3% | 3.5% | 3.9% | 0.9876 | 93.6% | 1,030 |
| Original, IMFDB | 87.5% | 84.2% | 92.3% | 88.1% | 17.3% | 7.7% | 10.5% | 0.9545 | 71.1% | 28,092 |
| Fine-tuned, IMFDB | 88.3% | 84.6% | 93.5% | 88.9% | 17.0% | 6.5% | 10.0% | 0.9626 | 70.7% | 28,092 |

Conclusion: a small gain on hard photos (IMFDB EER 10.5% to 10.0%, AUC 0.9545 to 0.9626) and on masked faces, but a clear loss on clean halfFace photos (EER 2.9% to 3.9%, AUC 0.9906 to 0.9876). Women on IMFDB lost accuracy at the shared threshold (83.0% to 77.8%) while their EER stayed the same (14.5% vs 14.4%): the fine-tuned scores sit higher, so one threshold fits worse. A shipped model needs one threshold, so this still counts against it. **Decision: keep the original model** (`config.json` `rec_onnx` stays null). The fine-tuned file is `runs/ft1/finetuned.onnx` (174 MB, not in git; sha256 eed639baed2de1f9c221f341f6478ddd1f81ea0363c0584baa3817f2894b9318).

## 5. Pakistani public figures (Wikimedia, never trained on in the GPU run)

13 people, 48 photos, 1,081 pairs (70 same-person pairs). Thresholds taken from the halfFace validation people at FAR 1% (original 0.290, fine-tuned 0.350), not from this set.

| | Original | Fine-tuned |
|---|---|---|
| Accuracy | 99.2% | 99.4% |
| Same person correctly accepted | 100.0% | 100.0% |
| Different person wrongly accepted | 0.9% | 0.7% |
| AUC | 0.99996 | 0.99996 |
| 95% range on accuracy | 96.1% to 100.0% | 96.7% to 100.0% |
| Photos with no face found | 2.1% | 2.1% |

Both are near perfect and the difference (about 2 pairs) is noise. A harder Pakistani test with real coverings (dupatta, hijab, cap, beard, mask) is still missing; the script `tools/eval_pk_test.py` is ready for hand-picked photos in `data/pk_test/`.

## 6. Limits

- Only about 3,000 training photos of 159 people; far too little to improve a model trained on millions of faces.
- About 90% of halfFace people are men (15 women in 166). Women results are weak and uncertain.
- Test sets are small (64 held-out people), so numbers have wide error ranges; we used a bootstrap to show it.
- halfFace and IMFDB are South Asian and Indian proxies, not Pakistani data. No public Pakistani identity dataset exists.
- Wikimedia people are public figures in article photos (celebrity-style); the label cleaning is lenient on purpose.
- The threshold chosen on mixed validation data gave about 5% false accepts on clean halfFace (target 1%); thresholds must be picked on data like the benchmark.
- The face detector ran on CPU in the evaluation because the GPU build of ONNX Runtime would not start on the cloud VM (CUDA 12.9 mismatch); the numbers are identical, only slower.
- Real webcam test of the live demo and unknown-person rejection are still to do.

## 7. What we did, in order

1. 2026-10-18/19: chose ArcFace R50 (D1, D2), built the pipeline, evaluation kit, augmentation and the data collector; baseline on halfFace.
2. 2026-10-20/21: collected Wikimedia public figures (only 13 usable people), decided A+B: split halfFace by person for training and use Wikimedia as the Pakistani check.
3. 2026-10-21/22: added IMFDB (100 actors, official download), built the combined training list, set up a Google Cloud GPU VM in Mumbai, fixed three setup problems (Python environment, a missing system library, GPU ONNX), fine-tuned, evaluated before and after, deleted the VM after verifying the results.
4. Result: keep the original ArcFace model; report the fine-tuning as a careful negative-to-mixed result.
