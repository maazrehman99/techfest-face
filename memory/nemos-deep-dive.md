---
name: nemos-deep-dive
description: "NED University CSIT Tech Fest '26 \"Nemo's Deep Dive\" DL competition; Round 1 on 17 Oct 2026, rulebook in Downloads"
metadata:
  node_type: memory
  type: project
  originSessionId: da6b614f-2192-4078-9ddc-a9b2f02a23c1
  modified: 2026-10-17T06:38:50.863Z
---

User wants to win Nemo's Deep Dive (AI Alliance, NED CSIT). Rulebook: ~/Downloads/Nemos Deep Dive - Rulebook.pdf.

- Round 1 "Sonar Sweep": 2026-10-17, CSIT lab, 3h, team of 2-3, Kaggle notebooks only, no phones/internet, provided PCs. Unknown tabular dataset (classification or regression). Handwritten answer sheets (skew, missing values, imputation, split strategy, model choice, preprocessing). Score = model performance + reasoning.
- Top 10 go to Round 2 "Every Fish Has a Face": week-long online fine-tuning of pretrained face-embedding model on Pakistani dataset.
- Practice dataset suggested: Kaggle Titanic train.csv.

**Why:** user is competing; approach counts as much as score.
**How to apply:** 2026-10-17 (competition day): wrote ~/nemo_round1_cheatsheet.md (time plan, sklearn pipeline skeleton, answer-sheet phrases). Built ~/titanic_rehearsal.ipynb (untested on real Titanic data) and ~/helper.ipynb (generic clf/reg, config cell, find() name-search, answer-sheet draft; tested on synthetic data both modes). Rulebook: every member needs a Kaggle account, Kaggle is the only allowed site -> plan: import helper.ipynb into own Kaggle account so it can be opened in the lab (unconfirmed with organizers whether prepared code is allowed). Next session: ask how Round 1 went / whether they advanced to Round 2, then prep Round 2 (face-embedding fine-tuning).
