---
name: techfest-face
description: Nemo's Deep Dive Round 2 "TechFest Face Recognition Challenge: Pakistani Context"; project at ~/techfest-face; plan, decisions, deadlines
metadata:
  type: project
---

Task: end-to-end face recognition pipeline (detect, align, quality filter, embed, match+threshold), fine-tune on self-collected consented Pakistani faces, eval on non-overlapping benchmark (acc/P/R/F1/ROC/FAR/FRR/EER, ablation, per-group). Report due 2026-10-22; hidden benchmark 2026-10-23; presentation counts too.

Decisions (2026-10-18, user approved): collect/create own data with consent (no public Pakistani identity-labelled set exists; PFDB only Pakistani-vs-not); compute = Colab via official VS Code extension (notebooks only) with Kaggle ~30h/wk backup; user has NO GPU (CPU only, 12 cores, 31GB). Model: compare InsightFace ArcFace (buffalo_l/w600k_r50, non-commercial license) vs AdaFace, fine-tune the winner. Wrap existing projects (insightface, AdaFace), only write the eval harness + data/consent tooling ourselves.

**Why:** reasoning must be documented per decision (user asked). **How to apply:** keep ~/techfest-face/DECISIONS.md updated; build eval harness + baseline first (CPU), fine-tune on Colab.

Status: research done, building started (nothing committed yet).

## Update 2026-10-18 (evening)
- Deliverables: report (due 10-22) + WORKING FILES the organizers run themselves on their hidden benchmark (format/IO not specified; ask organizers). Package must be offline, CPU, pinned deps, run.py with verify/embed/pairs modes, config.json threshold, never crash on no-face.
- User decided NOT to collect own volunteer data. Plan: public South Asian data (halfFace CC BY 4.0, 166 ppl masked/unmasked, gender in filename, from Mendeley pk44mkx9vm; IMFDB; BUPT Indian by request; RFW Indian as benchmark). Report must state openly that data is South Asian/Indian proxy, not Pakistani-specific. Risk: statement says teams collect own data + present consent process.
- Built in ~/techfest-face: pipeline.py (analyze/decide, toggle stages), evalkit.py, evaluate.py (parallel, ablation, per-group), run.py (CLI entry), make_lfw_manifest.py, ingest.py, COLLECTION_PROTOCOL.md, DECISIONS.md. CPU runs ~1 img/s (12 cores); LFW sanity ablation running (results/lfw).
- Gotchas: pkill -f kills own shell (use [e]valuate pattern); sklearn LFW pixels are 0-1 floats.
- Next: halfFace download -> manifest -> eval; Colab fine-tune notebook; README/requirements; report.

## State at shutdown 2026-10-18 night
- Built: augment.py (synthetic masks/hijab/dupatta/cap/turban/glasses/beard/lighting/blur/low-res/CCTV; preview results/aug/preview.png), halfFace manifest (data/halfface/manifest.csv, 938 imgs/166 ppl, 15 women; eval-only), detector pad-and-retry fallback for extreme close-ups, run.py, REQUESTS_TO_SEND.md (organizer questions + BUPT access request).
- LFW sanity eval was 61% done when PC shut down (infos.pkl only saved at end, so rerun from scratch); halfFace baseline eval was queued behind it. Resume: `cd ~/techfest-face; . .venv/bin/activate; FACE_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python evaluate.py --manifest data/halfface/manifest.csv --out results/halfface --ablation` (run halfFace first; LFW is only a sanity check).
- User to send TODAY: organizer questions + BUPT request. Wikimedia was unreachable earlier (retry Commons downloader). Training data source still undecided.
- Next: halfFace baseline eval, stress test, training data, Colab notebook (user runs it), README/requirements, report due 10-22.
