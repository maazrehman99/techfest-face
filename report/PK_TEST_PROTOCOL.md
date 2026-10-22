# Pakistani test set: protocol (fixed BEFORE any photo is scored)

Goal: a TEST-ONLY set of about 50 Pakistani people with 2 photos each (about 100 photos) to compare the original and the
fine-tuned model. Written 2026-10-22. Do not change these rules after scoring starts.

## Who may be included
- Real Pakistani public figures (cricketers, actors, politicians, journalists, scientists...), adults only.
- Not already in `data/pk_wiki/manifest.csv` (the 13 Wikimedia people) and not a halfFace / IMFDB person.
- Aim for a spread: at least 15 women, different regions, ages and professions.

## Which photos
- Exactly 2 photos per person, taken on different occasions (different day, clothes or background).
- One clear, correctly named face per photo (check the caption or article, never guess the identity).
- Prefer a mix of conditions: about a third with a covering (dupatta, hijab, turban, cap), beard, glasses or mask; some low-resolution or
  poor-light photos. The mix is chosen from what the photo looks like, never from model scores.
- Free licence or clearly allowed use; record the page link, licence and author for every photo.

## Rules while collecting
- Nobody runs either model on candidate photos until the list is frozen. Do not drop or swap a photo because of its score.
- Photos stay local (`data/pk_test/`, git-ignored), used only for this evaluation, deleted after the competition.

## Scoring (run once, report whatever comes out)
`python tools/check_pk_test.py` (must pass), then
`python tools/eval_pk_test.py --dir data/pk_test --tags data/pk_test/tags.csv --ft runs/ft1/finetuned.onnx --thr 0.290 --thr-ft 0.350`.
Thresholds are the fixed validation values, not tuned on this set. Report both models with the bootstrap interval,
including covering vs plain pairs. If the fine-tuned model is not better, say so.
