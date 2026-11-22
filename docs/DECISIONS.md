# Design decisions (with reasoning)

Written during the competition (18 to 22 October 2026). Every choice is justified against the problem statement, the 22 Oct report deadline, and the fact that the team has **no local GPU**.

## D1. Reuse InsightFace (detector + recognizer) instead of writing our own
- Statement allows "any suitable architecture"; judged on pipeline quality, fine-tuning on Pakistani data, and evaluation rigor, not on re-inventing backbones.
- InsightFace `buffalo_l` ships SCRFD-10G detector (5-point landmarks), ArcFace ResNet-50 trained on WebFace600K, ONNX, CPU-runnable.
- Cost of writing our own: days. Benefit: none for ranking. Our original work goes into the evaluation harness, quality filter, data/consent tooling, fine-tuning and fairness analysis.
- License: pretrained weights are non-commercial research only. Acceptable for a competition; must be cited in the report.

## D2. Baseline = ArcFace R50 (w600k_r50); AdaFace is the challenger
- ArcFace R50 is the strongest model that runs acceptably on CPU for evaluation and has a clean PyTorch fine-tune path (convert ONNX weights or use an equivalent IResNet50 checkpoint).
- AdaFace's quality-adaptive margin targets blur/low-res, which matches the CCTV/mobile requirement, but needs its own checkpoint and PyTorch inference. Compare both on our validation set before spending GPU hours; fine-tune the winner only.

## D3. Cosine similarity on L2-normalised embeddings; threshold chosen at a target FAR on a validation split
- ArcFace is trained with angular margin loss, so cosine is the metric the embedding space was optimised for.
- Threshold picked on a validation set (identities disjoint from the benchmark) at EER and at FAR=1e-2/1e-3, never on the test set, so reported numbers are not tuned on the benchmark.

## D4. Quality filter = blur (Laplacian variance), brightness, face size, detector score
- Cheap, interpretable, CPU-only, and each term maps to a statement item (blur, poor lighting, low resolution).
- Used for (a) reporting results per quality bin, (b) optionally rejecting unusable images. Its effect is measured in the ablation rather than assumed.

## D5. Evaluation harness is custom code
- Existing toolkits (InsightFace, face.evoLVe) cover LFW/CFP/AgeDB/IJB bins only: no FAR/FRR/EER tables, no per-group slices, no ablation runner.
- Harness: identity-disjoint pair generation, ROC/AUC, EER, TAR@FAR, accuracy/precision/recall/F1 at the chosen threshold, per-group breakdowns, and a stage-toggle ablation.

## D6. Compute: Colab (VS Code extension, notebooks) for fine-tuning; local CPU for everything else
- No local GPU. Colab free T4 is enough for fine-tuning an R50 on a small dataset. Kaggle (~30 GPU h/week) is the backup.
- Local CPU handles detection, embedding extraction, evaluation, and report generation, which keeps scarce GPU quota for training only.

## D7. Data (LOCKED 2026-10-20, supersedes the earlier team-collection plan)
Final goal: a model well trained for Pakistani faces, including dupatta/scarf/hijab, turban/cap and beards.
- Team is three friends, so large self-collection is not feasible. No public Pakistani identity-labelled dataset exists.
- **Real identity data:** public South Asian sets (halfFace, IMFDB, BUPT Indian by request, RFW Indian) PLUS CC-licensed photos of Pakistani public figures (cricketers, actors, politicians, journalists) from Wikimedia Commons: identity-labelled, several photos per person, real beards/turbans/dupattas. Licenses recorded per image; no private individuals scraped.
- **Coverings:** `augment.py` overlays (dupatta, scarf/hijab, turban, cap, beard, glasses, mask, lighting, blur, CCTV) applied to real identities, so identity labels stay correct. The real public-figure photos are the check that the synthetic coverings transfer.
- **LLM/AI-generated faces are NOT used for identity training:** generators do not keep one identity consistent across photos, they encode stereotypes, and synthetic-vs-real is a domain gap. They may only be used for illustration in the presentation, labelled as synthetic.
- **The three team members** enrol with written consent as the live-demo gallery and as a tiny real test; far too few people for training or for headline metrics.
- Split by identity into train/val/test; halfFace stays eval-only. The report states plainly that data is South Asian + Pakistani public figures, not a large consented Pakistani collection.

## D8. Fine-tuning recipe (conservative because the dataset is small)
- Low LR, freeze early blocks, ArcFace margin loss with small margin, mask/blur/low-res/lighting augmentation, early stopping on validation EER.
- Always compare against the frozen baseline: a fine-tune that raises train accuracy but not validation EER is rejected, and we say so in the report.

## D9. Two modes, one model (LOCKED 2026-10-20)
- Mode 1 verify (hidden benchmark): two images -> same/different + cosine score (`run.py verify|pairs`).
- Mode 2 face memory: enroll once with consent, no retraining; on images/video/live show **name + match score, else Unknown** (`gallery.py`, `live.py`). Extra "details" fields are out of scope.
- Both use the same fine-tuned recognizer (config.json rec_onnx); after swapping the model run `gallery.py reindex`.
## D10. Compute (LOCKED)
- Colab notebook written by us, run by the user on a T4 (Kaggle backup); export ONNX, wire into config.json.

## D11. Compute update (2026-10-22): Google Cloud GPU VM replaces Colab (supersedes D10)
- One NVIDIA L4 VM (`g2-standard-8`) in Mumbai (`asia-south1-b`), a company cloud project (name deliberately not recorded here), driven step by step from the laptop (`tools/gcp_finetune.sh`, later by hand). VM deleted only after results were downloaded and the model checksum matched. Hard stop 3 h as a cost guard.
- Rule from the user: no script may delete the VM on its own after a failure (it did once and cost a 14-minute re-upload). Face data uploaded: only aligned 112x112 crops of halfFace/IMFDB and the original validation/test photos for the evaluation; no Wikimedia, no personal photos.

## D12. Data (2026-10-21/22): A+B plus IMFDB
- halfFace split by person 60/15/25 (build_ft_data.py); Wikimedia only as a small Pakistani check; IMFDB added (build_imfdb.py: 98 actors x 40 photos, split by actor 59/15/24). Reason: about 600 training photos were too few.

## D13. Which model ships (2026-10-22): the ORIGINAL ArcFace R50
- Rule (D8): keep the fine-tune only if it is better. Result: small gain on hard IMFDB frames and masked faces, loss on clean halfFace (EER 2.9% to 3.9%), no gain on the Pakistani check, bootstrap shows the gain is marginal. `config.json` `rec_onnx` stays null. The fine-tune is reported as an honest mixed result. ArcFace R50 as the base model stays our deliberate choice (D1, D2); AdaFace was never tested.

## D14. Extra Pakistani test photos (open)
- A hand-picked, test-only set of Pakistani photos with real coverings is allowed; bulk scraping of Google Images is not (unlicensed). Openverse (CC) answered with a bot check, so this is still open. The evaluation script is ready: `tools/eval_pk_test.py`.
